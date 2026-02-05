"""
LLM Inference-based suffix attack learner for ablation studies (no GRPO training).

This learner uses the same inference workflow as TRLSuffixLearner but without
any model weight updates. Instead, it incorporates feedback and experience
history into the prompt itself, enabling prompt-based learning.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from agentdojo.agent_pipeline import AgentPipeline
from agentdojo.attacks.base_attacks import BaseAttack
from agentdojo.base_tasks import BaseInjectionTask, BaseUserTask
from agentdojo.task_suite.task_suite import TaskSuite
from rlpi.attack.learners.base import AdaptiveAttackLearner
from rlpi.attack.learners.common.feedback_utils import (
    compute_adaptive_reward,
    compute_gpt_feedback_for_iteration,
)
from rlpi.attack.learners.common.stats_utils import (
    log_reward_stats,
    save_reward_stats,
)
from rlpi.attack.learners.llm_inference.utils import (
    create_enhanced_adversarial_prompt,
)
from rlpi.attack.learners.trl_suffix.evaluator import SuffixEvaluator
from rlpi.attack.learners.trl_suffix.utils import (
    build_gpt_config_dict,
    format_trl_suffix_prompt,
    generate_suffix_from_policy,
    save_experience_to_file,
)
from rlpi.attack.templates.task_modifier import InjectionTaskModifier

logger = logging.getLogger("LLMInferenceLearner")


class LLMInferenceLearner(AdaptiveAttackLearner):
    """LLM Inference-based suffix attack learner (ablation study - no GRPO training).

    This learner uses prompt-based learning by incorporating experience history
    and feedback into the generation prompt, without any model weight updates.
    """

    def __init__(
        self,
        attack_model_name: str,
        feedback_model: Optional[str] = None,
        victim_model_name: str = "the AI assistant",
        user_name: str = "User",
        device: str = "cuda",
        seed: int = 42,
        # Generation parameters
        max_suffix_length: int = 20,
        temperature: float = 0.7,
        top_p: float = 0.9,
        # Experience-based prompt parameters
        use_experience_examples: bool = True,
        num_experience_examples: int = 3,
        include_gpt_reasoning: bool = True,
        # Other parameters
        verbose: bool = True,
        log_dir: Optional[str] = None,
        save_outputs: bool = True,
        save_experience_frequency: int = 1,
        # GPT feedback parameters
        gpt_enabled: bool = True,
        gpt_sparse_threshold: float = 0.1,
        gpt_medium_threshold: float = 0.3,
        gpt_dense_threshold: float = 0.5,
        gpt_sparse_weight: float = 0.8,
        gpt_medium_weight: float = 0.5,
        gpt_transition_weight: float = 0.3,
        gpt_dense_weight: float = 0.1,
        **kwargs,
    ):
        """Initialize LLM Inference Learner.

        Args:
            attack_model_name: Model name for attack policy that generates adversarial suffixes
                             (e.g., "Qwen/Qwen2-1.5B").
            feedback_model: Model identifier for GPT feedback (e.g., "gpt-4o-2024-08-06").
            victim_model_name: Display name of the victim model being attacked
                             (e.g., "GPT-4", "Claude"). Used in prompts.
            use_experience_examples: Whether to use experience-based examples in prompt
            num_experience_examples: Number of top examples to include from experience
            include_gpt_reasoning: Whether to include GPT reasoning for best suffix in prompt
            save_outputs: Whether to save training outputs to files.
            save_experience_frequency: Frequency for saving experiences to file.
                                     1 = save all experiences (default),
                                     N = save every Nth experience,
                                     0 = disable experience saving.

        All parameters are typically provided via Hydra config.
        See src/rlpi/agentdojo/config/learner/llm_inference.yaml for defaults.
        """
        super().__init__()

        # Store all config parameters
        self._store_config_params(locals())

        # Initialize components
        self._init_state_variables()
        self._init_model_and_tokenizer()
        self._init_output_files()

    def _store_config_params(self, params: Dict[str, Any]) -> None:
        """Store all configuration parameters as instance variables."""
        # Remove special keys
        params.pop("self", None)
        params.pop("kwargs", None)
        params.pop("__class__", None)

        # Store each param
        for key, value in params.items():
            setattr(self, key, value)

        # Set default log_dir if not provided
        log_dir: Optional[str] = getattr(self, "log_dir", None)
        if log_dir is None:
            self.log_dir = os.getcwd()

    def _init_state_variables(self) -> None:
        """Initialize state tracking variables."""
        self.experience_count = (
            0  # Track number of experiences for ID generation
        )

        # Current task pair (set via set_current_user_task_obj and modify_tasks)
        self.current_user_task: Optional[BaseUserTask] = None

        # Current iteration state (single task only)
        self.current_prompt: Optional[str] = None
        self.current_suffix: Optional[str] = None

        # Evaluation infrastructure (set later via set_evaluation_infrastructure)
        self.evaluator: Optional[SuffixEvaluator] = None
        self.injection_task: Optional[BaseInjectionTask] = None

        # Suffix tracking: store best
        self.best_suffix: Optional[str] = None
        self.best_reward: float = -float("inf")

        # Early stopping flag
        self.early_stopped: bool = False

        # Experience history for prompt-based learning
        self.experience_history: List[Dict[str, Any]] = []

    def _init_model_and_tokenizer(self) -> None:
        """Load policy model and tokenizer."""
        logger.info(f"Loading attack model: {self.attack_model_name}")

        self.policy = AutoModelForCausalLM.from_pretrained(
            self.attack_model_name,
            torch_dtype=torch.float16,
            device_map="auto",
        )

        self.tokenizer = AutoTokenizer.from_pretrained(self.attack_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        set_seed(self.seed)
        logger.info(f"Model loaded on device: {self.device}")

    def _init_output_files(self) -> None:
        """Initialize output file paths if saving enabled."""
        if not self.save_outputs:
            return

        # Create output directory
        outputs_dir = os.path.join(self.log_dir, "llm_inference_outputs")
        os.makedirs(outputs_dir, exist_ok=True)

        # Initialize tracking files (no GRPO training files needed)
        self.experience_file = os.path.join(
            outputs_dir, "experience_history.jsonl"
        )
        self.stats_file = Path(self.log_dir) / "reward_stats.jsonl"

        logger.info(f"Output files initialized in {self.log_dir}")

    def _get_gpt_config(self) -> Dict[str, Any]:
        """Build GPT config dictionary."""
        return build_gpt_config_dict(
            gpt_enabled=self.gpt_enabled,
            feedback_model=self.feedback_model,
            gpt_sparse_threshold=self.gpt_sparse_threshold,
            gpt_medium_threshold=self.gpt_medium_threshold,
            gpt_dense_threshold=self.gpt_dense_threshold,
            gpt_sparse_weight=self.gpt_sparse_weight,
            gpt_medium_weight=self.gpt_medium_weight,
            gpt_transition_weight=self.gpt_transition_weight,
            gpt_dense_weight=self.gpt_dense_weight,
        )

    def set_evaluation_infrastructure(
        self,
        pipeline: AgentPipeline,
        suite: TaskSuite,
        attacker: BaseAttack,
        injection_tasks: List[BaseInjectionTask],
    ):
        """
        Inject the evaluation infrastructure for real-time reward evaluation.

        This must be called BEFORE inference starts, typically right after
        the learner is created in adaptive_agentdojo.py.

        Args:
            pipeline: The agent pipeline (victim model)
            suite: The task suite
            attacker: The attack strategy
            injection_tasks: List with single injection task (only supports one)
        """
        # LLM inference learner only supports single injection task
        assert len(injection_tasks) == 1, (
            f"LLM inference learner only supports exactly 1 injection task, "
            f"but got {len(injection_tasks)} injection tasks."
        )

        self.evaluator = SuffixEvaluator(
            pipeline=pipeline,
            suite=suite,
            attacker=attacker,
            user_name=self.user_name,
            victim_model_name=self.victim_model_name,
            logdir=Path(self.log_dir),
        )

        # Store single injection task
        self.injection_task = injection_tasks[0]

        logger.info(
            f"Evaluation infrastructure set for single injection task: {self.injection_task.ID}"
        )

    def set_current_user_task_obj(self, user_task: BaseUserTask):
        """Set the current user task object for evaluation.

        This should be called BEFORE modify_tasks() to ensure the
        prompt has the correct user task context.

        Args:
            user_task: The BaseUserTask object for the current iteration
        """
        self.current_user_task = user_task
        logger.info(f"Set current user_task: {user_task.ID}")

    def modify_tasks(
        self, tasks: list[BaseInjectionTask], user_task: str | None = None
    ) -> list[InjectionTaskModifier]:
        """Generate suffix for single injection task in current iteration.

        Args:
            tasks: List with single injection task (only supports one)
            user_task: User task string for prompt generation (required)

        Returns:
            List with single InjectionTaskModifier
        """
        if user_task is None:
            raise ValueError(
                "user_task is required for LLM inference learner. "
                "It should be set via set_current_user_task_obj() and passed to modify_tasks()."
            )

        task = tasks[0]

        # Generate prompt with experience-based examples and best suffix context
        if self.use_experience_examples:
            prompt = create_enhanced_adversarial_prompt(
                original_goal=task._original_goal,
                user_task=user_task,
                best_suffix=self.best_suffix,
                experience_history=self.experience_history,
                top_k_examples=self.num_experience_examples,
                include_gpt_reasoning=self.include_gpt_reasoning,
            )
        else:
            # Fallback to basic prompt (for comparison)
            from rlpi.attack.learners.trl_suffix.utils import (
                create_adversarial_prompt,
            )

            prompt = create_adversarial_prompt(
                task._original_goal, user_task, best_suffix=self.best_suffix
            )

        suffix = generate_suffix_from_policy(
            policy=self.policy,
            tokenizer=self.tokenizer,
            prompt=prompt,
            device=self.device,
            max_suffix_length=self.max_suffix_length,
            temperature=self.temperature,
            top_p=self.top_p,
            verbose=self.verbose,
        )

        if self.verbose and self.best_suffix is not None:
            logger.info(
                f"Generating with best suffix context (reward={self.best_reward:.3f})"
            )

        # Store current iteration state
        self.current_prompt = prompt
        self.current_suffix = suffix

        if self.current_user_task is None:
            raise ValueError(
                "current_user_task must be set before calling modify_tasks"
            )

        logger.info(
            f"Generated suffix for (user_task={self.current_user_task.ID}, "
            f"injection_task={task.ID})"
        )

        # Format goal and create modifier
        formatted_goal = format_trl_suffix_prompt(
            malicious_goal=task._original_goal,
            suffix=suffix,
            user_name=self.user_name,
            model_name=self.victim_model_name,
        )

        modifier = InjectionTaskModifier(task)
        modifier.set_formatted_goal(formatted_goal)

        if self.verbose:
            logger.info(f"Generated suffix: {suffix}")

        return [modifier]

    def update_scores(
        self,
        success_rate: float,
        utility_score: float | None = None,
        user_task_str: str | None = None,
    ):
        """Update experience history and track best suffix.

        Args:
            success_rate: Success rate for the single task
            utility_score: Utility score for the single task
            user_task_str: User task string for GPT comparison (optional, uses default if None)
        """
        # Check for early stopping condition: utility = 1.0 and security = 1.0
        utility = utility_score if utility_score is not None else 0.0
        if utility == 1.0 and success_rate == 1.0:
            self.early_stopped = True
            logger.info(
                f"Early stopping triggered: utility={utility:.1f}, security={success_rate:.1f}"
            )
            # Still save the experience (suffix already saved via modify_tasks -> benchmark)
            # But return early
            return

        # Use default if not provided
        if user_task_str is None:
            user_task_str = "the user's task"

        # Compute GPT feedback for current iteration
        prob_1, prob_0, is_better, reasoning = self._compute_gpt_feedback(
            user_task_str
        )

        # When GPT is disabled, pass None for gpt_score instead of 0.0 for clarity
        gpt_score = prob_1 if self.gpt_enabled else None

        # Add experiences with computed rewards (need reward to track best)
        current_reward = self._add_experiences_with_rewards(
            success_rate,
            utility_score,
            prob_1,
            prob_0,
            is_better,
            reasoning,
            gpt_score,
        )

        # Update suffix tracking with current reward (to track best)
        self._update_suffix_tracking(current_reward)

        # Save outputs and log stats
        self._save_outputs()

    def _compute_gpt_feedback(
        self, user_task_str: str
    ) -> Tuple[float, float, bool, str]:
        """Compute GPT feedback score and comparison result for current suffix.

        Args:
            user_task_str: User task string for GPT comparison prompt

        Returns:
            Tuple of (prob_1, prob_0, is_better, reasoning)
        """
        if self.current_suffix is None or self.injection_task is None:
            return 0.0, 0.0, False, ""

        # Build GPT config with user_task_str for comparison prompt
        gpt_config = self._get_gpt_config()
        gpt_config["user_task_str"] = user_task_str

        # Use BEST suffix for comparison (not last)
        return compute_gpt_feedback_for_iteration(
            current_suffix=self.current_suffix,
            previous_suffix=self.best_suffix,
            injection_goal=self.injection_task._original_goal,
            gpt_config=gpt_config,  # Use the config with user_task_str, not a new one
            verbose=self.verbose,
        )

    def _update_suffix_tracking(self, current_reward: float) -> None:
        """Update suffix storage for next iteration comparison.

        Args:
            current_reward: Reward for current suffix (to determine if it's the best)
        """
        # Update best suffix if current is better
        if current_reward > self.best_reward:
            self.best_reward = current_reward
            self.best_suffix = self.current_suffix
            if self.verbose:
                logger.info(
                    f"Updated best suffix: reward {self.best_reward:.3f}"
                )

    def _add_experiences_with_rewards(
        self,
        success_rate: float,
        utility_score: Optional[float],
        prob_1: float,
        prob_0: float,
        is_better: bool,
        gpt_reasoning: str,
        gpt_score: Optional[float] = None,
    ) -> float:
        """Add experience with computed reward to history.

        Works with single task - uses current_* instance attributes.

        Args:
            gpt_score: GPT score to use for reward computation. If None, will use prob_1
                      (for backward compatibility) or None if GPT is disabled.

        Returns:
            The computed reward value (needed for tracking best suffix)
        """
        # Type guards for current state
        if (
            self.injection_task is None
            or self.current_prompt is None
            or self.current_suffix is None
            or self.current_user_task is None
        ):
            raise RuntimeError(
                "Current task state not set. Call modify_tasks() first."
            )

        # Compute reward for single task
        individual_success = success_rate
        individual_utility = (
            utility_score if utility_score is not None else 0.0
        )

        # Use provided gpt_score if available, otherwise fall back to prob_1 for backward compatibility
        # When GPT is disabled, gpt_score should be None (passed from update_scores)
        if gpt_score is None:
            gpt_score = prob_1 if self.gpt_enabled else None

        reward = compute_adaptive_reward(
            security_score=individual_success,
            utility_score=individual_utility,
            gpt_score=gpt_score,
            gpt_config=self._get_gpt_config(),
        )

        # Increment experience counter
        self.experience_count += 1

        logger.info(
            f"Experience #{self.experience_count} with reward={reward:.3f} "
            f"(GPT: {'BETTER' if is_better else 'NOT BETTER'} than best) "
            f"for (user_task={self.current_user_task.ID}, "
            f"injection_task={self.injection_task.ID})"
        )

        # Add to experience history for prompt-based learning
        experience_entry = {
            "suffix": self.current_suffix,
            "reward": reward,
            "success_rate": individual_success,
            "utility_score": individual_utility,
            "prob_0": prob_0,
            "prob_1": prob_1,
            "is_better": is_better,
            "gpt_reasoning": gpt_reasoning,
        }
        self.experience_history.append(experience_entry)

        # Save experience to file
        # save_experience_frequency: 1 = save all, N = save every Nth, 0 = disable
        if self.save_outputs and self.save_experience_frequency > 0:
            if self.experience_count % self.save_experience_frequency == 0:
                # Regenerate formatted_goal when saving
                formatted_goal = format_trl_suffix_prompt(
                    malicious_goal=self.injection_task._original_goal,
                    suffix=self.current_suffix,
                    user_name=self.user_name,
                    model_name=self.victim_model_name,
                )
                save_experience_to_file(
                    self.experience_file,
                    self.experience_count,
                    self.current_prompt,
                    self.current_suffix,
                    reward,
                    individual_success,
                    individual_utility,
                    formatted_goal,
                    prob_0=prob_0,
                    prob_1=prob_1,
                    task_id=self.injection_task.ID,
                    user_task=self.current_user_task.ID,
                    is_better_than_previous=is_better,
                    gpt_reasoning=gpt_reasoning,
                )

        return reward

    def get_training_queries_used(self) -> int:
        """Get number of queries used during training.

        For LLM inference learner, there is no training, so this always returns 0.

        Returns:
            0 (no training queries used)
        """
        return 0

    def estimate_training_queries(self) -> int:
        """Estimate queries that will be used in next training session.

        For LLM inference learner, there is no training, so this always returns 0.

        Returns:
            0 (no training queries)
        """
        return 0

    def _save_outputs(self) -> None:
        """Save outputs and log statistics."""
        # Log and save statistics after each update_scores
        self._log_and_save_stats()

    def _log_and_save_stats(self) -> None:
        """Log and save reward function and evaluator statistics."""
        if self.evaluator is None:
            return

        # Get evaluator stats
        evaluator_stats = self.evaluator.get_eval_stats()

        # Create dummy reward stats (since we don't have a reward function)
        reward_stats = {
            "total_calls": 0,
            "total_completions": 0,
            "new_evaluations": 0,
            "cached_evaluations": 0,
        }

        # Log to console
        if self.verbose:
            log_reward_stats(
                reward_stats, evaluator_stats, prefix="[UPDATE_SCORES]"
            )

        # Save to file
        if self.save_outputs:
            save_reward_stats(
                self.stats_file,
                reward_stats,
                evaluator_stats,
                training_session=0,  # No training sessions
            )
