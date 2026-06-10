"""
TRL-based suffix attack learner using GRPO.
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl import GRPOTrainer

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
    log_stats_summary,
    save_reward_stats,
)
from rlpi.attack.learners.trl_suffix.evaluator import SuffixEvaluator
from rlpi.attack.learners.trl_suffix.reward_utils import (
    create_experience_based_reward_function,
)
from rlpi.attack.learners.trl_suffix.utils import (
    build_gpt_config_dict,
    create_adversarial_prompt,
    create_grpo_config,
    execute_grpo_training_session,
    format_trl_suffix_prompt,
    generate_suffix_from_policy,
    init_output_files,
    prepare_training_prompts_from_current,
    save_experience_to_file,
)
from rlpi.attack.templates.task_modifier import InjectionTaskModifier

logger = logging.getLogger("TRLSuffixLearner")


class TRLSuffixLearner(AdaptiveAttackLearner):
    """TRL-based suffix attack learner using GRPO."""

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
        # GRPO training parameters
        min_experiences_for_training: int = 10,
        grpo_num_generations: int = 8,
        grpo_num_iterations: int = 1,
        grpo_learning_rate: float = 1e-7,
        grpo_per_device_train_batch_size: int = 2,
        grpo_gradient_accumulation_steps: int = 2,
        grpo_warmup_steps: int = 20,
        # Stability parameters (to prevent gradient explosion)
        grpo_max_grad_norm: float = 0.1,
        grpo_beta: float = 10.0,
        grpo_adam_epsilon: float = 1e-5,
        grpo_adam_beta2: float = 0.98,
        # Real evaluation parameters
        grpo_eval_num_generalization_tasks: int = 0,
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
        """Initialize TRL Suffix Learner.

        Args:
            attack_model_name: Model name for attack policy that generates adversarial suffixes
                             (e.g., "Qwen/Qwen2-1.5B").
            feedback_model: Model identifier for GPT feedback (e.g., "gpt-4o-2024-08-06").
            victim_model_name: Display name of the victim model being attacked
                             (e.g., "GPT-4", "Claude"). Used in prompts.
            save_outputs: Whether to save training outputs to files.
            save_experience_frequency: Frequency for saving experiences to file.
                                     1 = save all experiences (default),
                                     N = save every Nth experience,
                                     0 = disable experience saving.

        All parameters are typically provided via Hydra config.
        See src/rlpi/agentdojo/config/learner/trl_suffix.yaml for defaults.
        """
        super().__init__(initial_tokens=[], exploration_rate=0.0)

        # Store all config parameters
        self._store_config_params(locals())

        # Initialize components
        self._init_state_variables()
        self._init_model_and_tokenizer()
        self._init_grpo_trainer()  # Only initializes config, not trainer
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
        self.training_count = 0
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

    def _init_model_and_tokenizer(self) -> None:
        """Load policy model and tokenizer."""
        logger.info(f"Loading attack model: {self.attack_model_name}")

        self.policy = AutoModelForCausalLM.from_pretrained(
            self.attack_model_name,
            torch_dtype=torch.float16,
            device_map="auto",
        )

        # Ensure model parameters are trainable (required for gradient checkpointing)
        # By default, parameters should have requires_grad=True, but we ensure it explicitly
        for param in self.policy.parameters():
            param.requires_grad = True

        self.tokenizer = AutoTokenizer.from_pretrained(self.attack_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Ensure pad_token_id is set (required for GRPO generation)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Ensure model's generation config has pad_token_id set
        # This prevents CUDA device-side assert errors during generation
        if hasattr(self.policy, "generation_config"):
            if self.policy.generation_config.pad_token_id is None:
                self.policy.generation_config.pad_token_id = (
                    self.tokenizer.pad_token_id
                )
            if self.policy.generation_config.eos_token_id is None:
                self.policy.generation_config.eos_token_id = (
                    self.tokenizer.eos_token_id
                )

        set_seed(self.seed)
        logger.info(f"Model loaded on device: {self.device}")

    def _init_grpo_trainer(self) -> None:
        """Initialize GRPO-related directories and reward function placeholder."""
        # Reward function will be set later in set_evaluation_infrastructure
        self.reward_function: Optional[
            Callable[[List[str], List[str]], List[float]]
        ] = None

        logger.info(
            "GRPO initialization complete (trainer will be created per training session)"
        )

    def _create_grpo_trainer_with_reward_function(
        self,
        reward_function: Callable[[List[str], List[str]], List[float]],
        training_dataset: Dataset,
        max_steps: int,
    ) -> GRPOTrainer:
        """Create a new GRPO trainer instance with the specified reward function.

        We create a fresh trainer for each training session to ensure:
        1. The reward function is properly registered (GRPO requires it at init time)
        2. No state is carried over between training sessions
        3. The trainer configuration matches the current training requirements

        Args:
            reward_function: The reward function to use for training
            training_dataset: The actual dataset to train on
            max_steps: The calculated max steps for this training session

        Returns:
            New GRPOTrainer instance with reward function configured
        """
        # Create a new config with the calculated max_steps
        grpo_config = create_grpo_config(
            output_dir=os.path.join(self.log_dir, "grpo_output"),
            grpo_per_device_train_batch_size=self.grpo_per_device_train_batch_size,
            grpo_learning_rate=self.grpo_learning_rate,
            grpo_num_generations=self.grpo_num_generations,
            grpo_num_iterations=self.grpo_num_iterations,
            grpo_gradient_accumulation_steps=self.grpo_gradient_accumulation_steps,
            grpo_warmup_steps=self.grpo_warmup_steps,
            max_suffix_length=self.max_suffix_length,
            max_steps=max_steps,  # Set max_steps at config creation time
            max_grad_norm=self.grpo_max_grad_norm,
            beta=self.grpo_beta,
            adam_epsilon=self.grpo_adam_epsilon,
            adam_beta2=self.grpo_adam_beta2,
        )

        # Ensure model is ready for training
        self.policy.train()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if self.verbose:
            logger.info("Model ready for GRPO training")

        # Create new trainer with the actual reward function and dataset
        grpo_trainer = GRPOTrainer(
            model=self.policy,
            reward_funcs=[reward_function],
            train_dataset=training_dataset,
            args=grpo_config,
            processing_class=self.tokenizer,
        )

        logger.info(
            f"Created new GRPO trainer instance with reward function: "
            f"{type(reward_function).__name__}, dataset size: {len(training_dataset)}, "
            f"max_steps: {max_steps}"
        )

        return grpo_trainer

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

    def _build_reward_function(
        self,
    ) -> Callable[[List[str], List[str]], List[float]]:
        """Build reward function with current state.

        This is the single source of truth for reward function creation.
        Called from:
        1. set_evaluation_infrastructure() - initial setup
        2. _train_with_grpo() - before each training session

        Raises:
            RuntimeError: If evaluator not set via set_evaluation_infrastructure()
        """
        if self.evaluator is None:
            raise RuntimeError(
                "Cannot create reward function without evaluator. "
                "Call set_evaluation_infrastructure() first."
            )

        # Set GRPO eval file path if saving enabled
        grpo_eval_file = None
        if self.save_outputs and hasattr(self, "grpo_eval_file"):
            grpo_eval_file = self.grpo_eval_file

        return create_experience_based_reward_function(
            evaluator=self.evaluator,
            current_user_task=self.current_user_task,
            injection_task=self.injection_task,
            feedback_model=self.feedback_model,
            gpt_config=self._get_gpt_config(),
            verbose=self.verbose,
            training_session_id=self.training_count,
            grpo_eval_file=grpo_eval_file,
        )

    def _init_output_files(self) -> None:
        """Initialize output file paths if saving enabled."""
        if not self.save_outputs:
            return

        paths = init_output_files(self.log_dir)
        for key, path in paths.items():
            setattr(self, key, path)

        # Add stats file
        self.stats_file = Path(self.log_dir) / "reward_stats.jsonl"

        # Add GRPO evaluations file (for GPT feedback during training)
        self.grpo_eval_file = str(
            Path(self.grpo_outputs_dir) / "grpo_evaluations.jsonl"
        )

        logger.info(f"Output files initialized in {self.log_dir}")

    def set_evaluation_infrastructure(
        self,
        pipeline: AgentPipeline,
        suite: TaskSuite,
        attacker: BaseAttack,
        injection_tasks: List[BaseInjectionTask],
    ):
        """
        Inject the evaluation infrastructure for real-time reward evaluation.

        This must be called BEFORE training starts, typically right after
        the learner is created in adaptive_agentdojo.py.

        Args:
            pipeline: The agent pipeline (victim model)
            suite: The task suite
            attacker: The attack strategy
            injection_tasks: List with single injection task (TRL suffix only supports one)
        """
        # TRL suffix learner only supports single injection task
        assert len(injection_tasks) == 1, (
            f"TRL suffix learner only supports exactly 1 injection task, "
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

        # Create reward function now that we have all required components
        self.reward_function = self._build_reward_function()

    def set_current_user_task_obj(self, user_task: BaseUserTask):
        """Set the current user task object for evaluation.

        This should be called BEFORE modify_tasks() to ensure the
        reward function has the correct user task context.

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
            tasks: List with single injection task (TRL suffix only supports one)
            user_task: User task string for prompt generation

        Returns:
            List with single InjectionTaskModifier
        """

        task = tasks[0]

        # Generate prompt with best suffix context (like self-transfer)
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
        """Update experience history and trigger training if ready.

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
            # But skip training and return early
            return

        # Use default if not provided
        if user_task_str is None:
            user_task_str = "the user's task"

        # Compute GPT feedback for current iteration
        prob_1, prob_0, is_better, reasoning = self._compute_gpt_feedback(
            user_task_str
        )

        # Add experiences with computed rewards (need reward to track best)
        current_reward = self._add_experiences_with_rewards(
            success_rate, utility_score, prob_1, prob_0, is_better, reasoning
        )

        # Update suffix tracking with current reward (to track best)
        self._update_suffix_tracking(current_reward)

        # Save outputs and check training trigger
        self._save_and_check_training_trigger()

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
            gpt_config=self._get_gpt_config(),
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
    ) -> float:
        """Add experience with computed reward to history.

        Works with single task - uses current_* instance attributes.

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

        # Compute reward for single task (use prob_1 as gpt_score)
        individual_success = success_rate
        individual_utility = (
            utility_score if utility_score is not None else 0.0
        )

        reward = compute_adaptive_reward(
            security_score=individual_success,
            utility_score=individual_utility,
            gpt_score=prob_1,
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

        # Save experience with comparison result
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

        Returns:
            Number of queries used by evaluator (0 if evaluator not set)
        """
        if self.evaluator is None:
            return 0
        return self.evaluator.get_queries_used()

    def estimate_training_queries(self) -> int:
        """Estimate queries that will be used in next training session.

        Returns:
            Estimated number of queries: num_prompts * grpo_num_generations
            For current implementation: 1 prompt * grpo_num_generations
        """
        # For current implementation: 1 prompt * grpo_num_generations
        return self.grpo_num_generations

    def _save_and_check_training_trigger(
        self,
    ) -> None:
        """Save outputs and check if we should trigger training."""
        # GPT scores are now saved directly in experience_history.jsonl
        # via save_experience_to_file(), so no separate GPT scores file needed

        # Check if we should train (with min_experiences=1, train after each experience)
        if self.experience_count >= self.min_experiences_for_training:
            logger.info(
                f"Triggering GRPO training after {self.experience_count} experiences"
            )
            self._train_with_grpo()
        elif self.verbose:
            logger.info(
                f"Not enough experiences yet: "
                f"{self.experience_count}/{self.min_experiences_for_training}"
            )

        # Log and save statistics after each update_scores
        self._log_and_save_stats()

    def _log_training_stats(self) -> None:
        """Log comprehensive statistics after GRPO training."""
        if self.reward_function is None or self.evaluator is None:
            return

        # Get stats from both components
        reward_stats = getattr(self.reward_function, "stats", None)
        if reward_stats is None:
            return

        evaluator_stats = self.evaluator.get_eval_stats()

        # Log detailed summary
        logger.info(f"\n{log_stats_summary(reward_stats, evaluator_stats)}")

        # Also log evaluator cache info
        if self.evaluator and self.verbose:
            self.evaluator.log_eval_stats()

    def _log_and_save_stats(self) -> None:
        """Log and save reward function and evaluator statistics."""
        if self.reward_function is None or self.evaluator is None:
            return

        # Get stats from both components
        reward_stats = getattr(self.reward_function, "stats", None)
        if reward_stats is None:
            logger.warning("Reward function has no stats attribute")
            return

        evaluator_stats = self.evaluator.get_eval_stats()

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
                training_session=self.training_count,
            )

    def _train_with_grpo(self):
        """Run GRPO training session with accumulated experiences."""
        self.training_count += 1
        training_start = datetime.now()

        logger.info(f"Starting GRPO training session #{self.training_count}")

        # Reset query counter at start of training to track queries for this session only
        if self.evaluator is not None and hasattr(
            self.evaluator, "reset_queries_used"
        ):
            self.evaluator.reset_queries_used()

        # Prepare training data
        prompts = prepare_training_prompts_from_current(
            current_prompt=self.current_prompt,
            experience_count=self.experience_count,
            training_count=self.training_count,
            training_file=self.training_file if self.save_outputs else None,
            save_outputs=self.save_outputs,
            tokenizer=self.tokenizer,  # Pass tokenizer for validation
        )

        # Rebuild reward function with updated training_count for logging
        if self.current_user_task is None or self.injection_task is None:
            raise RuntimeError(
                "Cannot train: current_user_task or injection_task not set"
            )
        self.reward_function = self._build_reward_function()

        # Check if reward function has stats attribute (confirms it's our function)
        if hasattr(self.reward_function, "stats"):
            logger.info(
                f"Reward function stats before training: {self.reward_function.stats}"
            )

        # Prepare the dataset
        training_dataset = Dataset.from_dict({"prompt": prompts})

        # Calculate max_steps based on the training configuration
        batch_size = self.grpo_per_device_train_batch_size
        total_completions_per_iter = len(prompts) * self.grpo_num_generations
        steps_per_iter = max(
            1, (total_completions_per_iter + batch_size - 1) // batch_size
        )
        max_steps = steps_per_iter * self.grpo_num_iterations

        # Create a new GRPO trainer instance with the reward function
        # This is critical: GRPO may require the reward function to be set at initialization
        # rather than dynamically, so we create a fresh trainer instance for each session
        grpo_trainer = self._create_grpo_trainer_with_reward_function(
            reward_function=self.reward_function,
            training_dataset=training_dataset,
            max_steps=max_steps,
        )

        logger.info(
            f"Reward function registered in new trainer: {len(grpo_trainer.reward_funcs)} function(s), "
            f"type: {type(grpo_trainer.reward_funcs[0])}"
        )

        # Configure and execute training
        execute_grpo_training_session(
            grpo_trainer=grpo_trainer,
            grpo_config=grpo_trainer.args,
            training_count=self.training_count,
            training_start=training_start,
            metrics_file=self.metrics_file if self.save_outputs else None,
            save_outputs=self.save_outputs,
            verbose=self.verbose,
            reward_function=self.reward_function,
            evaluator=self.evaluator,
        )

        logger.info(
            f"GRPO training #{self.training_count} completed in "
            f"{(datetime.now() - training_start).total_seconds():.1f}s"
        )

        # Check if early stopping was triggered during GRPO training
        if hasattr(
            self.reward_function, "early_stop_info"
        ) and self.reward_function.early_stop_info.get("triggered", False):
            self.early_stopped = True
            early_stop_info = self.reward_function.early_stop_info
            logger.info(
                f"Early stopping detected after GRPO training: "
                f"utility={early_stop_info.get('utility', 'N/A'):.1f}, "
                f"security={early_stop_info.get('security', 'N/A'):.1f}. "
                f"Suffix already saved to grpo_evaluations.jsonl"
            )

        # Log comprehensive statistics after training
        self._log_training_stats()

    def save_model(
        self, path: str, queries_used: Optional[int] = None
    ) -> None:
        """Save model and learner state to checkpoint.

        Args:
            path: Path to save checkpoint (should be .pt file for model)
            queries_used: Optional total queries used (for tracking progress)
        """
        # Extract base path and create checkpoint directory structure
        checkpoint_dir = Path(path).parent
        checkpoint_name = Path(path).stem

        # Save model state dict
        model_path = path
        torch.save(
            {
                "policy_state_dict": self.policy.state_dict(),
                "model_config": {
                    "attack_model_name": self.attack_model_name,
                    "dtype": str(self.policy.dtype),
                },
            },
            model_path,
        )
        logger.info(f"Saved policy model to {model_path}")

        # Save learner state to JSON file (same name, different extension)
        state_path = checkpoint_dir / f"{checkpoint_name}_state.json"

        checkpoint_state = {
            "checkpoint_version": "1.0",
            "timestamp": datetime.now().isoformat(),
            "learner_state": {
                "experience_count": self.experience_count,
                "training_count": self.training_count,
                "best_suffix": self.best_suffix,
                "best_reward": float(self.best_reward),
                "early_stopped": self.early_stopped,
                "queries_used": (
                    queries_used if queries_used is not None else None
                ),
            },
            "model_info": {
                "attack_model_name": self.attack_model_name,
                "checkpoint_path": str(model_path),
            },
            "config_snapshot": {
                "max_suffix_length": self.max_suffix_length,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "grpo_num_generations": self.grpo_num_generations,
                "grpo_num_iterations": self.grpo_num_iterations,
                "grpo_learning_rate": self.grpo_learning_rate,
                "min_experiences_for_training": self.min_experiences_for_training,
            },
        }

        with open(state_path, "w") as f:
            json.dump(checkpoint_state, f, indent=2)
        logger.info(f"Saved checkpoint state to {state_path}")

    def load_model(self, path: str) -> None:
        """Load model and learner state from checkpoint.

        Args:
            path: Path to checkpoint model file (.pt) or state file (.json)
                  If .pt, will look for corresponding _state.json file
        """
        checkpoint_path = Path(path)

        # Determine if we have model file or state file
        if checkpoint_path.suffix == ".pt":
            model_path = checkpoint_path
            state_path = (
                checkpoint_path.parent / f"{checkpoint_path.stem}_state.json"
            )
        elif (
            checkpoint_path.suffix == ".json"
            and "_state.json" in checkpoint_path.name
        ):
            state_path = checkpoint_path
            # Extract model path from state file name
            model_name = checkpoint_path.stem.replace("_state", "")
            model_path = checkpoint_path.parent / f"{model_name}.pt"
        else:
            logger.error(f"Unrecognized checkpoint file format: {path}")
            return

        # Load model state dict
        if not model_path.exists():
            logger.warning(
                f"Model checkpoint not found at {model_path}, skipping model loading"
            )
        else:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.policy.load_state_dict(checkpoint["policy_state_dict"])
            logger.info(f"Loaded policy model from {model_path}")

            # Validate model config matches
            if "model_config" in checkpoint:
                saved_model_name = checkpoint["model_config"].get(
                    "attack_model_name"
                )
                if saved_model_name != self.attack_model_name:
                    logger.warning(
                        f"Model name mismatch: checkpoint has {saved_model_name}, "
                        f"but current config has {self.attack_model_name}"
                    )

        # Load learner state
        if not state_path.exists():
            logger.warning(
                f"Checkpoint state file not found at {state_path}. "
                "Only model weights were loaded. State variables will remain at defaults."
            )
            return

        with open(state_path, "r") as f:
            checkpoint_state = json.load(f)

        # Validate checkpoint version
        version = checkpoint_state.get("checkpoint_version", "0.0")
        if version != "1.0":
            logger.warning(
                f"Checkpoint version {version} may not be compatible with current code"
            )

        # Restore learner state
        learner_state = checkpoint_state.get("learner_state", {})
        self.experience_count = learner_state.get("experience_count", 0)
        self.training_count = learner_state.get("training_count", 0)
        self.best_suffix = learner_state.get("best_suffix", None)
        self.best_reward = float(
            learner_state.get("best_reward", -float("inf"))
        )
        self.early_stopped = learner_state.get("early_stopped", False)
        queries_used = learner_state.get("queries_used")

        logger.info(
            f"Loaded checkpoint state: experience_count={self.experience_count}, "
            f"training_count={self.training_count}, best_reward={self.best_reward:.3f}, "
            f"queries_used={queries_used}"
        )

        # Return queries_used if caller needs it (stored as attribute for access)
        self._checkpoint_queries_used = queries_used

        # Validate config snapshot if present
        config_snapshot = checkpoint_state.get("config_snapshot", {})
        if config_snapshot:
            mismatches = []
            for key, saved_value in config_snapshot.items():
                current_value = getattr(self, key, None)
                if current_value is not None and current_value != saved_value:
                    mismatches.append(
                        f"{key}: {saved_value} (saved) vs {current_value} (current)"
                    )

            if mismatches:
                logger.warning(
                    "Config mismatches detected (may affect training behavior):\n"
                    + "\n".join(mismatches)
                )
