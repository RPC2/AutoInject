"""
TRL-based suffix attack learner using GRPO with joint training on multiple task pairs.
"""

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
from rlpi.attack.learners.trl_suffix.utils import (
    build_gpt_config_dict,
    create_adversarial_prompt,
    create_grpo_config,
    create_joint_reward_function,
    create_universal_adversarial_prompt,
    execute_grpo_training_session,
    format_trl_suffix_prompt,
    generate_suffix_from_policy,
    init_output_files,
    prepare_training_prompts_from_all_pairs,
    save_experience_to_file,
)
from rlpi.attack.templates.task_modifier import InjectionTaskModifier

logger = logging.getLogger("TRLSuffixJointLearner")


class TRLSuffixJointLearner(AdaptiveAttackLearner):
    """TRL-based suffix attack learner with joint training on multiple task pairs."""

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
        # Real evaluation parameters
        grpo_eval_num_generalization_tasks: int = 0,
        # Other parameters
        verbose: bool = True,
        log_dir: Optional[str] = None,
        save_outputs: bool = True,
        save_experience_frequency: int = 1,
        # Universal suffix mode
        universal_suffix_mode: bool = False,  # If True, train one suffix for all pairs
        universal_reward_aggregation: str = "mean",  # "mean", "min", "max", "median"
        # Early stopping
        disable_early_stopping: bool = True,
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
        """Initialize TRL Suffix Joint Learner.

        Args:
            attack_model_name: Model name for attack policy (e.g., "Qwen/Qwen2-1.5B")
            feedback_model: Model identifier for GPT feedback
            victim_model_name: Display name of the victim model
            user_name: User name for prompt formatting
            device: Device to run on
            seed: Random seed
            max_suffix_length: Maximum suffix length in tokens
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            min_experiences_for_training: Minimum experiences before training
            grpo_num_generations: Suffixes per prompt per iteration
            grpo_num_iterations: Training iterations per session
            grpo_learning_rate: Policy learning rate
            grpo_per_device_train_batch_size: Batch size per device
            grpo_gradient_accumulation_steps: Gradient accumulation steps
            grpo_warmup_steps: Learning rate warmup steps
            verbose: Enable detailed logging
            log_dir: Directory for logs
            save_outputs: Whether to save training outputs
            save_experience_frequency: Frequency for saving experiences
            gpt_enabled: Enable GPT-based feedback
            gpt_*: GPT feedback configuration parameters
        """
        super().__init__()

        # Store all config parameters
        self._store_config_params(locals())

        # Initialize components
        self._init_state_variables()
        self._init_model_and_tokenizer()
        self._init_grpo_trainer()
        self._init_output_files()

    def _store_config_params(self, params: Dict[str, Any]) -> None:
        """Store all configuration parameters as instance variables."""
        params.pop("self", None)
        params.pop("kwargs", None)
        params.pop("__class__", None)

        for key, value in params.items():
            setattr(self, key, value)

        log_dir: Optional[str] = getattr(self, "log_dir", None)
        if log_dir is None:
            self.log_dir = os.getcwd()

    def _init_state_variables(self) -> None:
        """Initialize state tracking variables for multiple task pairs."""
        self.training_count = 0
        self.experience_count = 0

        # Multi-pair state: track experiences per (user_task_id, injection_task_id) pair
        self.experiences: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}

        # Current iteration state (can be set per pair)
        self.current_prompts: Dict[Tuple[str, str], str] = {}
        self.current_suffixes: Dict[Tuple[str, str], str] = {}

        # Evaluation infrastructure
        self.evaluator: Optional[SuffixEvaluator] = None
        self.task_pairs: List[Tuple[BaseUserTask, BaseInjectionTask]] = []

        # Best suffix tracking per pair
        self.best_suffixes: Dict[Tuple[str, str], Optional[str]] = {}
        self.best_rewards: Dict[Tuple[str, str], float] = {}

        # Universal suffix mode: track one suffix for all pairs
        self.universal_suffix: Optional[str] = None
        self.universal_reward: float = -float("inf")

        # Early stopping per pair
        self.early_stopped_pairs: set = set()
        self.early_stopped: bool = False

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

    def _init_grpo_trainer(self) -> None:
        """Initialize GRPO-related directories and reward function placeholder."""
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
        """Create a new GRPO trainer instance with the specified reward function."""
        grpo_config = create_grpo_config(
            output_dir=os.path.join(self.log_dir, "grpo_output"),
            grpo_per_device_train_batch_size=self.grpo_per_device_train_batch_size,
            grpo_learning_rate=self.grpo_learning_rate,
            grpo_num_generations=self.grpo_num_generations,
            grpo_num_iterations=self.grpo_num_iterations,
            grpo_gradient_accumulation_steps=self.grpo_gradient_accumulation_steps,
            grpo_warmup_steps=self.grpo_warmup_steps,
            max_suffix_length=self.max_suffix_length,
            max_steps=max_steps,
            max_grad_norm=getattr(self, "grpo_max_grad_norm", 0.1),
            beta=getattr(self, "grpo_beta", 10.0),
            adam_epsilon=getattr(self, "grpo_adam_epsilon", 1e-5),
            adam_beta2=getattr(self, "grpo_adam_beta2", 0.98),
        )

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
        """Build reward function for joint training on multiple pairs."""
        if self.evaluator is None:
            raise RuntimeError(
                "Cannot create reward function without evaluator. "
                "Call set_evaluation_infrastructure() first."
            )

        if not self.task_pairs:
            raise RuntimeError(
                "Cannot create reward function without task pairs. "
                "Call set_evaluation_infrastructure() first."
            )

        grpo_eval_file = None
        if self.save_outputs and hasattr(self, "grpo_eval_file"):
            grpo_eval_file = self.grpo_eval_file

        # Get prompt-to-pair mapping from current training prompts
        # This will be set during _train_with_grpo
        if not hasattr(self, "_current_prompt_to_pair_map"):
            raise RuntimeError(
                "Prompt-to-pair map not set. This should be set during training."
            )

        return create_joint_reward_function(
            evaluator=self.evaluator,
            task_pairs=self.task_pairs,
            prompt_to_pair_map=self._current_prompt_to_pair_map,
            feedback_model=self.feedback_model,
            gpt_config=self._get_gpt_config(),
            best_suffixes=self.best_suffixes,
            best_rewards=self.best_rewards,
            verbose=self.verbose,
            training_session_id=self.training_count,
            grpo_eval_file=grpo_eval_file,
            universal_mode=self.universal_suffix_mode,
            universal_reward_aggregation=self.universal_reward_aggregation,
        )

    def _init_output_files(self) -> None:
        """Initialize output file paths if saving enabled."""
        if not self.save_outputs:
            return

        paths = init_output_files(self.log_dir)
        for key, path in paths.items():
            setattr(self, key, path)

        self.stats_file = Path(self.log_dir) / "reward_stats.jsonl"
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
        """Inject the evaluation infrastructure for real-time reward evaluation.

        Args:
            pipeline: The agent pipeline (victim model)
            suite: The task suite
            attacker: The attack strategy
            injection_tasks: List of injection tasks (can be multiple for joint training)
        """
        self.evaluator = SuffixEvaluator(
            pipeline=pipeline,
            suite=suite,
            attacker=attacker,
            user_name=self.user_name,
            victim_model_name=self.victim_model_name,
            logdir=Path(self.log_dir),
        )

        # Store injection tasks (will be paired with user tasks later)
        # For now, we just store them - they'll be paired when task_pairs is set
        self._injection_tasks = injection_tasks

        logger.info(
            f"Evaluation infrastructure set for {len(injection_tasks)} injection task(s)"
        )

        # Note: task_pairs will be set by adaptive_agentdojo.py after this call

    def set_current_user_task_obj(self, user_task: BaseUserTask):
        """Set the current user task object for evaluation.

        This is called per iteration for each user task.
        For joint training, we track multiple user tasks.

        Args:
            user_task: The BaseUserTask object for the current iteration
        """
        # Store user task - will be paired with injection tasks in modify_tasks
        self._current_user_task = user_task
        logger.info(f"Set current user_task: {user_task.ID}")

    def modify_tasks(
        self, tasks: list[BaseInjectionTask], user_task: str | None = None
    ) -> list[InjectionTaskModifier]:
        """Generate suffixes for multiple injection tasks in current iteration.

        Args:
            tasks: List of injection tasks (can be multiple for joint training)
            user_task: User task string for prompt generation

        Returns:
            List of InjectionTaskModifiers (one per injection task)
        """
        if self._current_user_task is None:
            raise ValueError(
                "current_user_task must be set before calling modify_tasks"
            )

        modifiers = []

        if self.universal_suffix_mode:
            # Universal suffix mode: generate ONE suffix for all pairs
            # Include all user tasks and injection tasks in the prompt
            if tasks and self.task_pairs:
                prompt = create_universal_adversarial_prompt(
                    task_pairs=self.task_pairs,
                    best_suffix=self.universal_suffix,
                )

                # Generate ONE suffix
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

                if self.verbose:
                    logger.info(
                        f"Generated universal suffix (reward={self.universal_reward:.3f}): {suffix[:50]}..."
                    )

                # Store universal suffix
                self.universal_suffix = suffix

                # Use the same suffix for all tasks
                for injection_task in tasks:
                    pair_key = (self._current_user_task.ID, injection_task.ID)
                    self.current_prompts[pair_key] = prompt
                    self.current_suffixes[pair_key] = suffix

                    # Format goal and create modifier
                    formatted_goal = format_trl_suffix_prompt(
                        malicious_goal=injection_task._original_goal,
                        suffix=suffix,
                        user_name=self.user_name,
                        model_name=self.victim_model_name,
                    )

                    modifier = InjectionTaskModifier(injection_task)
                    modifier.set_formatted_goal(formatted_goal)
                    modifiers.append(modifier)
        else:
            # Conditional mode: generate different suffix per pair
            for injection_task in tasks:
                pair_key = (self._current_user_task.ID, injection_task.ID)

                # Get best suffix for this pair if available
                best_suffix = self.best_suffixes.get(pair_key)

                # Generate prompt with best suffix context
                prompt = create_adversarial_prompt(
                    injection_task._original_goal,
                    user_task,
                    best_suffix=best_suffix,
                )

                # Generate suffix from policy
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

                if self.verbose and best_suffix is not None:
                    best_reward = self.best_rewards.get(
                        pair_key, -float("inf")
                    )
                    logger.info(
                        f"Generating for pair {pair_key} with best suffix context "
                        f"(reward={best_reward:.3f})"
                    )

                # Store current iteration state for this pair
                self.current_prompts[pair_key] = prompt
                self.current_suffixes[pair_key] = suffix

                # Format goal and create modifier
                formatted_goal = format_trl_suffix_prompt(
                    malicious_goal=injection_task._original_goal,
                    suffix=suffix,
                    user_name=self.user_name,
                    model_name=self.victim_model_name,
                )

                modifier = InjectionTaskModifier(injection_task)
                modifier.set_formatted_goal(formatted_goal)
                modifiers.append(modifier)

                if self.verbose:
                    logger.info(
                        f"Generated suffix for pair {pair_key}: {suffix[:50]}..."
                    )

        return modifiers

    def update_scores(
        self,
        success_rate: float,
        utility_score: float | None = None,
        user_task_str: str | None = None,
        user_task_id: Optional[str] = None,
        injection_task_id: Optional[str] = None,
    ):
        """Update experience history for a specific task pair and trigger training if ready.

        Args:
            success_rate: Success rate for the task pair
            utility_score: Utility score for the task pair
            user_task_str: User task string for GPT comparison
            user_task_id: User task ID (required for joint training, except in universal mode)
            injection_task_id: Injection task ID (required for joint training, except in universal mode)
        """
        if self.universal_suffix_mode:
            # In universal mode, collect scores from all pairs
            # The workflow will call this for each pair, then call aggregate_universal_scores()
            if not hasattr(self, "_pending_universal_scores"):
                self._pending_universal_scores: Dict[
                    Tuple[str, str], Tuple[float, Optional[float]]
                ] = {}

            if user_task_id and injection_task_id:
                pair_key = (user_task_id, injection_task_id)
                self._pending_universal_scores[pair_key] = (
                    success_rate,
                    utility_score,
                )
                utility = utility_score if utility_score is not None else 0.0
                logger.info(
                    f"Collected universal suffix score for pair {pair_key}: "
                    f"security={success_rate:.3f}, utility={utility:.3f}"
                )
            return

        # Conditional mode: normal per-pair updates
        if user_task_id is None or injection_task_id is None:
            raise ValueError(
                "user_task_id and injection_task_id must be provided for joint training"
            )

        pair_key = (user_task_id, injection_task_id)

        # Check if this pair is already early stopped
        if pair_key in self.early_stopped_pairs:
            logger.info(
                f"Pair {pair_key} already early stopped, skipping update"
            )
            return

        # Check for early stopping condition
        if not self.disable_early_stopping:
            utility = utility_score if utility_score is not None else 0.0
            if utility == 1.0 and success_rate == 1.0:
                self.early_stopped_pairs.add(pair_key)
                logger.info(
                    f"Early stopping triggered for pair {pair_key}: "
                    f"utility={utility:.1f}, security={success_rate:.1f}"
                )
            # Still save the experience
            self._add_experiences_with_rewards(
                success_rate,
                utility_score,
                pair_key,
                user_task_str,
                0.0,
                0.0,
                False,
                "",
            )
            return

        # Use default if not provided
        if user_task_str is None:
            user_task_str = "the user's task"

        # Compute GPT feedback for current iteration
        prob_1, prob_0, is_better, reasoning = self._compute_gpt_feedback(
            user_task_str, pair_key
        )

        # Add experience with computed rewards
        current_reward = self._add_experiences_with_rewards(
            success_rate,
            utility_score,
            pair_key,
            user_task_str,
            prob_1,
            prob_0,
            is_better,
            reasoning,
        )

        # Update suffix tracking for this pair
        self._update_suffix_tracking(current_reward, pair_key)

        # Save outputs and check training trigger
        self._save_and_check_training_trigger()

    def _compute_gpt_feedback(
        self, user_task_str: str, pair_key: Tuple[str, str]
    ) -> Tuple[float, float, bool, str]:
        """Compute GPT feedback score for current suffix of a specific pair."""
        user_task_id, injection_task_id = pair_key

        # Get current suffix and injection task for this pair
        current_suffix = self.current_suffixes.get(pair_key)
        if current_suffix is None:
            return 0.0, 0.0, False, ""

        # Find the injection task object
        injection_task = None
        for ut, it in self.task_pairs:
            if ut.ID == user_task_id and it.ID == injection_task_id:
                injection_task = it
                break

        if injection_task is None:
            logger.warning(f"Injection task not found for pair {pair_key}")
            return 0.0, 0.0, False, ""

        # Build GPT config
        gpt_config = self._get_gpt_config()
        gpt_config["user_task_str"] = user_task_str

        # Use BEST suffix for comparison
        best_suffix = self.best_suffixes.get(pair_key)

        return compute_gpt_feedback_for_iteration(
            current_suffix=current_suffix,
            previous_suffix=best_suffix,
            injection_goal=injection_task._original_goal,
            gpt_config=gpt_config,
            verbose=self.verbose,
        )

    def _update_suffix_tracking(
        self, current_reward: float, pair_key: Tuple[str, str]
    ) -> None:
        """Update suffix storage for a specific pair."""
        best_reward = self.best_rewards.get(pair_key, -float("inf"))
        if current_reward > best_reward:
            self.best_rewards[pair_key] = current_reward
            self.best_suffixes[pair_key] = self.current_suffixes.get(pair_key)
            if self.verbose:
                logger.info(
                    f"Updated best suffix for pair {pair_key}: reward {current_reward:.3f}"
                )

    def _add_experiences_with_rewards(
        self,
        success_rate: float,
        utility_score: Optional[float],
        pair_key: Tuple[str, str],
        user_task_str: Optional[str] = None,
        prob_1: float = 0.0,
        prob_0: float = 0.0,
        is_better: bool = False,
        gpt_reasoning: str = "",
    ) -> float:
        """Add experience with computed reward to history for a specific pair."""
        user_task_id, injection_task_id = pair_key

        # Get current state for this pair
        current_prompt = self.current_prompts.get(pair_key)
        current_suffix = self.current_suffixes.get(pair_key)

        if current_prompt is None or current_suffix is None:
            raise RuntimeError(
                f"Current task state not set for pair {pair_key}. Call modify_tasks() first."
            )

        # Find the injection task object
        injection_task = None
        for ut, it in self.task_pairs:
            if ut.ID == user_task_id and it.ID == injection_task_id:
                injection_task = it
                break

        if injection_task is None:
            raise RuntimeError(f"Injection task not found for pair {pair_key}")

        # Compute reward
        reward = compute_adaptive_reward(
            security_score=success_rate,
            utility_score=utility_score if utility_score is not None else 0.0,
            gpt_score=prob_1,
            gpt_config=self._get_gpt_config(),
        )

        # Increment experience counter
        self.experience_count += 1

        # Initialize experience list for this pair if needed
        if pair_key not in self.experiences:
            self.experiences[pair_key] = []

        # Create experience entry
        experience = {
            "prompt": current_prompt,
            "suffix": current_suffix,
            "reward": reward,
            "success_rate": success_rate,
            "utility_score": (
                utility_score if utility_score is not None else 0.0
            ),
            "prob_1": prob_1,
            "prob_0": prob_0,
            "is_better": is_better,
            "gpt_reasoning": gpt_reasoning,
        }

        self.experiences[pair_key].append(experience)

        logger.info(
            f"Experience #{self.experience_count} for pair {pair_key} with reward={reward:.3f} "
            f"(GPT: {'BETTER' if is_better else 'NOT BETTER'} than best)"
        )

        # Save experience if enabled
        if self.save_outputs and self.save_experience_frequency > 0:
            if self.experience_count % self.save_experience_frequency == 0:
                formatted_goal = format_trl_suffix_prompt(
                    malicious_goal=injection_task._original_goal,
                    suffix=current_suffix,
                    user_name=self.user_name,
                    model_name=self.victim_model_name,
                )
                save_experience_to_file(
                    self.experience_file,
                    self.experience_count,
                    current_prompt,
                    current_suffix,
                    reward,
                    success_rate,
                    utility_score,
                    formatted_goal,
                    prob_0=prob_0,
                    prob_1=prob_1,
                    task_id=injection_task_id,
                    user_task=user_task_id,
                    is_better_than_previous=is_better,
                    gpt_reasoning=gpt_reasoning,
                )

        return reward

    def aggregate_universal_scores(self) -> None:
        """Aggregate scores from all pairs in universal suffix mode and update experience.

        This should be called after all pairs have been evaluated with the same universal suffix.
        """
        if not self.universal_suffix_mode:
            return

        if (
            not hasattr(self, "_pending_universal_scores")
            or not self._pending_universal_scores
        ):
            logger.warning("No pending universal scores to aggregate")
            return

        # Aggregate rewards across all pairs
        all_success_rates = [
            sr for sr, _ in self._pending_universal_scores.values()
        ]
        all_utility_scores = [
            us
            for _, us in self._pending_universal_scores.values()
            if us is not None
        ]

        if self.universal_reward_aggregation == "mean":
            aggregated_success = sum(all_success_rates) / len(
                all_success_rates
            )
            aggregated_utility = (
                sum(all_utility_scores) / len(all_utility_scores)
                if all_utility_scores
                else 0.0
            )
        elif self.universal_reward_aggregation == "min":
            aggregated_success = min(all_success_rates)
            aggregated_utility = (
                min(all_utility_scores) if all_utility_scores else 0.0
            )
        elif self.universal_reward_aggregation == "max":
            aggregated_success = max(all_success_rates)
            aggregated_utility = (
                max(all_utility_scores) if all_utility_scores else 0.0
            )
        elif self.universal_reward_aggregation == "median":
            import numpy as np

            aggregated_success = float(np.median(all_success_rates))
            aggregated_utility = (
                float(np.median(all_utility_scores))
                if all_utility_scores
                else 0.0
            )
        else:
            raise ValueError(
                f"Unknown aggregation method: {self.universal_reward_aggregation}"
            )

        logger.info(
            f"Aggregated universal suffix scores across {len(self._pending_universal_scores)} pairs: "
            f"security={aggregated_success:.3f} ({self.universal_reward_aggregation}), "
            f"utility={aggregated_utility:.3f} ({self.universal_reward_aggregation})"
        )

        # Use the universal suffix for all pairs in experience tracking
        if self.universal_suffix is None:
            logger.warning("Universal suffix not set, cannot save experience")
            self._pending_universal_scores = {}
            return

        # Compute GPT feedback (use first pair's context as representative)
        first_pair_key = list(self._pending_universal_scores.keys())[0]
        user_task_id, injection_task_id = first_pair_key

        # Find the user task and injection task for GPT feedback
        user_task_obj = None
        injection_task = None
        for ut, it in self.task_pairs:
            if ut.ID == user_task_id and it.ID == injection_task_id:
                user_task_obj = ut
                injection_task = it
                break

        if injection_task and user_task_obj:
            # Use the actual user task string from the first pair
            user_task_str = user_task_obj.PROMPT
            prob_1, prob_0, is_better, reasoning = self._compute_gpt_feedback(
                user_task_str, first_pair_key
            )
        else:
            prob_1, prob_0, is_better, reasoning = 0.0, 0.0, False, ""

        # Add experience with aggregated reward for all pairs
        # Store the universal suffix and aggregated scores
        current_reward = self._add_universal_experience(
            aggregated_success,
            aggregated_utility,
            prob_1,
            prob_0,
            is_better,
            reasoning,
        )

        # Update universal suffix tracking
        if current_reward > self.universal_reward:
            self.universal_reward = current_reward
            logger.info(
                f"Updated best universal suffix: reward {current_reward:.3f}"
            )

        # Clear pending scores
        self._pending_universal_scores = {}

        # Check training trigger
        self._save_and_check_training_trigger()

    def _add_universal_experience(
        self,
        success_rate: float,
        utility_score: Optional[float],
        prob_1: float,
        prob_0: float,
        is_better: bool,
        gpt_reasoning: str,
    ) -> float:
        """Add experience with aggregated reward for universal suffix mode."""
        if self.universal_suffix is None:
            raise RuntimeError("Universal suffix not set")

        # Use first pair's prompt as representative
        if not self.current_prompts:
            raise RuntimeError("No prompts available for universal experience")

        representative_prompt = list(self.current_prompts.values())[0]

        # Compute reward
        reward = compute_adaptive_reward(
            security_score=success_rate,
            utility_score=utility_score if utility_score is not None else 0.0,
            gpt_score=prob_1,
            gpt_config=self._get_gpt_config(),
        )

        # Increment experience counter
        self.experience_count += 1

        # Store experience for all pairs (use a special key)
        universal_key = ("__universal__", "__universal__")
        if universal_key not in self.experiences:
            self.experiences[universal_key] = []

        experience = {
            "prompt": representative_prompt,
            "suffix": self.universal_suffix,
            "reward": reward,
            "success_rate": success_rate,
            "utility_score": (
                utility_score if utility_score is not None else 0.0
            ),
            "prob_1": prob_1,
            "prob_0": prob_0,
            "is_better": is_better,
            "gpt_reasoning": gpt_reasoning,
            "evaluated_on_pairs": list(self._pending_universal_scores.keys()),
        }

        self.experiences[universal_key].append(experience)

        logger.info(
            f"Universal experience #{self.experience_count} with reward={reward:.3f} "
            f"(evaluated on {len(self._pending_universal_scores)} pairs)"
        )

        # Save experience if enabled
        if self.save_outputs and self.save_experience_frequency > 0:
            if self.experience_count % self.save_experience_frequency == 0:
                # Use first injection task for formatting
                first_pair_key = list(self._pending_universal_scores.keys())[0]
                _, injection_task_id = first_pair_key
                injection_task = None
                for ut, it in self.task_pairs:
                    if it.ID == injection_task_id:
                        injection_task = it
                        break

                if injection_task:
                    formatted_goal = format_trl_suffix_prompt(
                        malicious_goal=injection_task._original_goal,
                        suffix=self.universal_suffix,
                        user_name=self.user_name,
                        model_name=self.victim_model_name,
                    )
                    save_experience_to_file(
                        self.experience_file,
                        self.experience_count,
                        representative_prompt,
                        self.universal_suffix,
                        reward,
                        success_rate,
                        utility_score,
                        formatted_goal,
                        prob_0=prob_0,
                        prob_1=prob_1,
                        task_id=f"universal_{injection_task_id}",
                        user_task="universal",
                        is_better_than_previous=is_better,
                        gpt_reasoning=gpt_reasoning,
                    )

        return reward

    def get_training_queries_used(self) -> int:
        """Get number of queries used during training."""
        if self.evaluator is None:
            return 0
        return self.evaluator.get_queries_used()

    def estimate_training_queries(self) -> int:
        """Estimate queries that will be used in next training session."""
        # Estimate based on number of prompts and generations
        num_prompts = sum(len(exps) for exps in self.experiences.values())
        return num_prompts * self.grpo_num_generations

    def _save_and_check_training_trigger(self) -> None:
        """Save outputs and check if we should trigger training."""
        # Check if we should train
        total_experiences = sum(
            len(exps) for exps in self.experiences.values()
        )
        if total_experiences >= self.min_experiences_for_training:
            logger.info(
                f"Triggering GRPO training after {total_experiences} total experiences "
                f"across {len(self.experiences)} task pairs"
            )
            self._train_with_grpo()
        elif self.verbose:
            logger.info(
                f"Not enough experiences yet: "
                f"{total_experiences}/{self.min_experiences_for_training}"
            )

        # Log and save statistics
        self._log_and_save_stats()

    def _log_and_save_stats(self) -> None:
        """Log and save reward function and evaluator statistics."""
        if self.reward_function is None or self.evaluator is None:
            return

        reward_stats = getattr(self.reward_function, "stats", None)
        if reward_stats is None:
            return

        evaluator_stats = self.evaluator.get_eval_stats()

        if self.verbose:
            log_reward_stats(
                reward_stats, evaluator_stats, prefix="[UPDATE_SCORES]"
            )

        if self.save_outputs:
            save_reward_stats(
                self.stats_file,
                reward_stats,
                evaluator_stats,
                training_session=self.training_count,
            )

    def _train_with_grpo(self):
        """Run GRPO training session with accumulated experiences from all pairs."""
        self.training_count += 1
        training_start = datetime.now()

        logger.info(
            f"Starting GRPO joint training session #{self.training_count} "
            f"on {len(self.experiences)} task pairs"
        )

        # Reset query counter
        if self.evaluator is not None and hasattr(
            self.evaluator, "reset_queries_used"
        ):
            self.evaluator.reset_queries_used()

        # Prepare training data from all pairs
        prompts, prompt_to_pair_map = prepare_training_prompts_from_all_pairs(
            experiences=self.experiences,
            training_count=self.training_count,
            training_file=self.training_file if self.save_outputs else None,
            save_outputs=self.save_outputs,
            universal_mode=self.universal_suffix_mode,
        )

        if not prompts:
            logger.warning("No prompts available for training, skipping")
            return

        # Store prompt-to-pair map for reward function
        self._current_prompt_to_pair_map = prompt_to_pair_map

        # Rebuild reward function with updated training_count
        if not self.task_pairs:
            raise RuntimeError("Cannot train: no task pairs set")

        self.reward_function = self._build_reward_function()

        # Prepare the dataset
        training_dataset = Dataset.from_dict({"prompt": prompts})

        # Calculate max_steps
        batch_size = self.grpo_per_device_train_batch_size
        total_completions_per_iter = len(prompts) * self.grpo_num_generations
        steps_per_iter = max(
            1, (total_completions_per_iter + batch_size - 1) // batch_size
        )
        max_steps = steps_per_iter * self.grpo_num_iterations

        # Create GRPO trainer
        grpo_trainer = self._create_grpo_trainer_with_reward_function(
            reward_function=self.reward_function,
            training_dataset=training_dataset,
            max_steps=max_steps,
        )

        logger.info(
            f"Reward function registered in new trainer: {len(grpo_trainer.reward_funcs)} function(s)"
        )

        # Execute training
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
            f"GRPO joint training #{self.training_count} completed in "
            f"{(datetime.now() - training_start).total_seconds():.1f}s"
        )

        # Check for early stopping
        if (
            not self.disable_early_stopping
            and hasattr(self.reward_function, "early_stop_info")
            and self.reward_function.early_stop_info
        ):
            for (
                pair_key,
                stop_info,
            ) in self.reward_function.early_stop_info.items():
                if stop_info.get("triggered", False):
                    self.early_stopped_pairs.add(pair_key)
                    logger.info(
                        f"Early stopping detected for pair {pair_key} after GRPO training"
                    )

        # Log statistics
        self._log_training_stats()

    def _log_training_stats(self) -> None:
        """Log comprehensive statistics after GRPO training."""
        if self.reward_function is None or self.evaluator is None:
            return

        reward_stats = getattr(self.reward_function, "stats", None)
        if reward_stats is None:
            return

        evaluator_stats = self.evaluator.get_eval_stats()

        logger.info(f"\n{log_stats_summary(reward_stats, evaluator_stats)}")

        if self.evaluator and self.verbose:
            self.evaluator.log_eval_stats()

    def save_model(self, path: str) -> None:
        """Save model and training state for transfer learning.

        Args:
            path: Path to save the model checkpoint
        """
        save_dict = {
            "policy_state_dict": self.policy.state_dict(),
            "tokenizer_config": (
                self.tokenizer.get_config()
                if hasattr(self.tokenizer, "get_config")
                else None
            ),
            "tokenizer_name": self.attack_model_name,
            "trained_pairs": [
                {"user_task_id": ut, "injection_task_id": it}
                for ut, it in self.experiences.keys()
            ],
            "experience_counts": {
                f"{ut}_{it}": len(exps)
                for (ut, it), exps in self.experiences.items()
            },
            "best_suffixes": {
                f"{ut}_{it}": suffix
                for (ut, it), suffix in self.best_suffixes.items()
                if suffix is not None
            },
            "best_rewards": {
                f"{ut}_{it}": reward
                for (ut, it), reward in self.best_rewards.items()
            },
            "universal_suffix": self.universal_suffix,
            "universal_reward": self.universal_reward,
            "universal_suffix_mode": self.universal_suffix_mode,
            "training_count": self.training_count,
            "experience_count": self.experience_count,
            "training_config": {
                "max_suffix_length": self.max_suffix_length,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "grpo_learning_rate": self.grpo_learning_rate,
                "grpo_num_generations": self.grpo_num_generations,
                "grpo_num_iterations": self.grpo_num_iterations,
            },
        }

        torch.save(save_dict, path)
        logger.info(f"Saved joint training model and state to {path}")

    def load_model(self, path: str) -> None:
        """Load model and training state from checkpoint.

        Args:
            path: Path to load the model checkpoint from
        """
        if not os.path.exists(path):
            logger.error(f"Model file not found at {path}")
            return

        checkpoint = torch.load(path, map_location=self.device)

        # Load policy state
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        logger.info("Loaded policy state dict")

        # Load tokenizer if config available
        if checkpoint.get("tokenizer_config") is not None:
            # Tokenizer config loading would go here if needed
            pass

        # Restore training state (optional - for analysis/debugging)
        if "trained_pairs" in checkpoint:
            logger.info(
                f"Model was trained on {len(checkpoint['trained_pairs'])} task pairs"
            )
            for pair_info in checkpoint["trained_pairs"]:
                logger.info(
                    f"  - {pair_info['user_task_id']} / {pair_info['injection_task_id']}"
                )

        if "best_suffixes" in checkpoint:
            # Convert back to tuple keys
            for key, suffix in checkpoint["best_suffixes"].items():
                ut, it = key.split("_", 1)
                self.best_suffixes[(ut, it)] = suffix
            logger.info(f"Restored {len(self.best_suffixes)} best suffixes")

        logger.info(f"Loaded model from {path}")
