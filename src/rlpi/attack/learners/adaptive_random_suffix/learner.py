"""
Adaptive Random Suffix Attack Learner.

This learner implements a random token-level baseline for adversarial suffix
generation, inspired by the llm-adaptive-attacks repository. It uses random
token substitution with self-transfer of successful suffixes across iterations.
"""

import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from transformers import AutoTokenizer, set_seed

from agentdojo.base_tasks import BaseInjectionTask, BaseUserTask
from rlpi.attack.learners.base import AdaptiveAttackLearner
from rlpi.attack.learners.common.feedback_utils import (
    compute_adaptive_reward,
    compute_gpt_feedback_for_iteration,
)
from rlpi.attack.learners.common.token_utils import (
    clean_generated_suffix,
    is_valid_adversarial_token,
)
from rlpi.attack.learners.trl_suffix.utils import (
    build_gpt_config_dict,
    format_trl_suffix_prompt,
    log_experience_state,
    save_experience_to_file,
)
from rlpi.attack.templates.task_modifier import InjectionTaskModifier

logger = logging.getLogger("AdaptiveRandomSuffixLearner")


class AdaptiveRandomSuffixLearner(AdaptiveAttackLearner):
    """
    Baseline learner that generates random suffixes using token-level random search.
    Uses successful suffixes as initialization for future iterations (self-transfer).
    """

    def __init__(
        self,
        attack_model_name: str,
        victim_model_name: str = "the AI assistant",
        user_name: str = "User",
        device: str = "cpu",  # No GPU needed for random generation
        seed: int = 42,
        # Generation parameters
        max_suffix_length: int = 50,
        min_suffix_length: int = 5,
        mutation_rate: float = 0.5,  # Fraction of tokens to mutate
        # Self-transfer parameters
        use_self_transfer: bool = True,
        # Other parameters
        verbose: bool = True,
        log_dir: Optional[str] = None,
        save_outputs: bool = True,
        # GPT feedback parameters (same as TRL suffix)
        gpt_enabled: bool = True,
        gpt_sparse_threshold: float = 0.1,
        gpt_medium_threshold: float = 0.3,
        gpt_dense_threshold: float = 0.5,
        gpt_sparse_weight: float = 0.8,
        gpt_medium_weight: float = 0.5,
        gpt_transition_weight: float = 0.3,
        gpt_dense_weight: float = 0.1,
        feedback_model: Optional[str] = None,  # For GPT feedback
        **kwargs,
    ):
        """Initialize Adaptive Random Suffix Learner.

        Args:
            attack_model_name: Model name for tokenizer used to generate adversarial suffixes
                             (e.g., "Qwen/Qwen2-1.5B"). This is the attack policy model.
            victim_model_name: Display name of the victim model being attacked
                             (e.g., "GPT-4", "Claude"). Used in prompts.
            feedback_model: Model identifier for GPT feedback (e.g., "gpt-4o-2024-08-06").

        All parameters are typically provided via Hydra config.
        See src/rlpi/agentdojo/config/learner/adaptive_random_suffix.yaml for defaults.
        """
        super().__init__(initial_tokens=[], exploration_rate=0.0)

        # Store all config parameters
        self._store_config_params(locals())

        # Initialize components
        self._init_state_variables()
        self._init_tokenizer()
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
        # Experience tracking (3-tuple: prompt, suffix, reward)
        self.all_experiences: List[Tuple[str, str, float]] = []

        # Current task pair (set via set_current_user_task_obj and modify_tasks)
        self.current_user_task: Optional[BaseUserTask] = None
        self.current_injection_task: Optional[BaseInjectionTask] = None

        # Current iteration state
        self.current_prompt: Optional[str] = None
        self.current_suffix: Optional[str] = None
        self.current_formatted_goal: Optional[str] = None
        self.current_modifier: Optional[InjectionTaskModifier] = None

        # Suffix tracking for self-transfer
        self.last_suffix: Optional[str] = None
        self.best_suffix: Optional[str] = None
        self.best_reward: float = -float("inf")

        # Early stopping flag
        self.early_stopped: bool = False

        # Random number generator for reproducibility
        self.rng = np.random.RandomState(self.seed)

    def _init_tokenizer(self) -> None:
        """Load tokenizer (no model needed)."""
        logger.info(f"Loading tokenizer: {self.attack_model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.attack_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.vocab_size = len(self.tokenizer)

        set_seed(self.seed)
        logger.info(f"Tokenizer loaded with vocab size: {self.vocab_size}")

        # Build list of valid tokens for sampling (cache for efficiency)
        logger.info("Building valid token cache...")
        self.valid_token_ids = self._build_valid_token_cache()
        logger.info(
            f"Valid token cache built: {len(self.valid_token_ids)} tokens"
        )

    def _build_valid_token_cache(self) -> List[int]:
        """Build cache of valid token IDs to sample from.

        Uses shared token validation logic from token_utils module.
        """
        valid_ids = []
        for token_id in range(self.vocab_size):
            decoded = self.tokenizer.decode([token_id])
            if is_valid_adversarial_token(decoded, self.tokenizer):
                valid_ids.append(token_id)
        return valid_ids

    def _init_output_files(self) -> None:
        """Initialize output file paths if saving enabled."""
        # Create subdirectory for outputs
        output_dir = os.path.join(self.log_dir, "random_suffix_outputs")
        os.makedirs(output_dir, exist_ok=True)

        self.output_dir = output_dir
        self.experience_file = os.path.join(
            output_dir, "experience_history.jsonl"
        )
        self.summary_file = os.path.join(output_dir, "summary.json")
        self.suffix_history_file = os.path.join(
            output_dir, "suffix_history.jsonl"
        )

        logger.info(f"Output files initialized in {output_dir}")

    def set_current_user_task_obj(self, user_task: BaseUserTask):
        """Set the current user task object for evaluation.

        Args:
            user_task: The BaseUserTask object for the current iteration
        """
        self.current_user_task = user_task
        logger.info(f"Set current user_task: {user_task.ID}")

    def modify_tasks(
        self, tasks: List[BaseInjectionTask], user_task: str | None = None
    ) -> List[InjectionTaskModifier]:
        """Generate random suffix for single task pair.

        Args:
            tasks: List containing exactly 1 injection task
            user_task: User task string (unused, kept for API compatibility)

        Returns:
            List containing single task modifier
        """

        # Ensure user_task was set before modify_tasks() is called
        if self.current_user_task is None:
            raise RuntimeError(
                "Must call set_current_user_task_obj() before modify_tasks(). "
                "This is required for correct user_task tracking."
            )

        # Extract single task
        task = tasks[0]
        self.current_injection_task = task

        logger.info(
            f"modify_tasks called with user_task={self.current_user_task.ID}, "
            f"injection_task={task.ID}"
        )

        # Generate or retrieve suffix for this task
        suffix = self._generate_or_retrieve_suffix()

        # Store current suffix
        self.current_suffix = suffix

        # Format into prompt (same as TRL suffix)
        formatted_goal = format_trl_suffix_prompt(
            malicious_goal=task._original_goal,
            suffix=suffix,
            user_name=self.user_name,
            model_name=self.victim_model_name,
        )

        # Store current state
        self.current_prompt = task._original_goal
        self.current_formatted_goal = formatted_goal

        # Create modifier
        modifier = InjectionTaskModifier(task)
        modifier.set_formatted_goal(formatted_goal)
        self.current_modifier = modifier

        if self.verbose:
            logger.info(f"Generated suffix for task {task.ID}: {suffix}")

        return [modifier]

    def _generate_or_retrieve_suffix(self) -> str:
        """Generate suffix using self-transfer if available."""
        if self.current_user_task is None:
            raise RuntimeError(
                "current_user_task must be set before calling _generate_or_retrieve_suffix"
            )
        if self.current_injection_task is None:
            raise RuntimeError(
                "current_injection_task must be set before calling _generate_or_retrieve_suffix"
            )

        user_task_id = self.current_user_task.ID
        task_id = self.current_injection_task.ID

        if self.use_self_transfer and self.best_suffix is not None:
            # Start from previous best suffix
            suffix = self._mutate_suffix(self.best_suffix)
            if self.verbose:
                logger.info(
                    f"Using self-transfer for ({user_task_id}, {task_id}), "
                    f"mutating previous best (reward={self.best_reward:.3f})"
                )
        else:
            # Pure random generation
            suffix = self._generate_random_suffix()
            if self.verbose:
                logger.info(
                    f"Generating pure random suffix for ({user_task_id}, {task_id})"
                )

        return suffix

    def _generate_random_suffix(self) -> str:
        """Generate completely random suffix."""
        # Sample length
        length = self.rng.randint(
            self.min_suffix_length, self.max_suffix_length + 1
        )

        # Sample random valid tokens
        sampled_ids = self.rng.choice(
            self.valid_token_ids, size=length, replace=True
        )

        # Decode to string
        suffix = self.tokenizer.decode(sampled_ids, skip_special_tokens=True)

        # Clean suffix to avoid YAML parsing errors
        return clean_generated_suffix(suffix)

    def _mutate_suffix(self, suffix: str) -> str:
        """Apply random mutations to existing suffix."""
        # Tokenize current suffix
        tokens = self.tokenizer.encode(suffix, add_special_tokens=False)

        # Calculate number of positions to mutate
        n_mutations = max(1, int(len(tokens) * self.mutation_rate))

        # Randomly select positions
        positions = self.rng.choice(len(tokens), n_mutations, replace=False)

        # Replace with random valid tokens
        for pos in positions:
            tokens[pos] = self._sample_random_valid_token()

        # Decode to string
        mutated = self.tokenizer.decode(tokens, skip_special_tokens=True)

        # Clean suffix to avoid YAML parsing errors
        return clean_generated_suffix(mutated)

    def _sample_random_valid_token(self) -> int:
        """Sample a random valid token ID."""
        return self.rng.choice(self.valid_token_ids)

    def update_scores(
        self,
        success_rate: float,
        utility_score: float | None = None,
        per_task_scores: Dict[str, Tuple[float, float]] | None = None,
    ):
        """Update suffix tracking and experience history based on performance.

        Args:
            success_rate: Success rate for the single task pair
            utility_score: Utility score for the single task pair
            per_task_scores: Ignored (kept for API compatibility with base class)
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

        # Ensure state is properly set
        if self.current_injection_task is None:
            raise RuntimeError(
                "current_injection_task must be set before update_scores"
            )
        if self.current_user_task is None:
            raise RuntimeError(
                "current_user_task must be set before update_scores"
            )
        if self.current_prompt is None or self.current_suffix is None:
            raise RuntimeError(
                "current_prompt and current_suffix must be set before update_scores"
            )
        if self.current_formatted_goal is None:
            raise RuntimeError(
                "current_formatted_goal must be set before update_scores"
            )

        # Compute GPT feedback for current suffix
        gpt_config_dict = build_gpt_config_dict(
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

        # Add user_task_str to config for GPT feedback (if enabled)
        if self.current_user_task and hasattr(
            self.current_user_task, "PROMPT"
        ):
            gpt_config_dict["user_task_str"] = self.current_user_task.PROMPT
        else:
            gpt_config_dict["user_task_str"] = "the user's task"

        prob_1, prob_0, is_better, reasoning = (
            compute_gpt_feedback_for_iteration(
                current_suffix=self.current_suffix,
                previous_suffix=self.best_suffix,
                injection_goal=self.current_injection_task._original_goal,
                gpt_config=gpt_config_dict,
                verbose=self.verbose,
            )
        )

        # Compute reward: use adaptive reward that combines security, utility, and GPT feedback
        # When GPT is disabled, this returns: 0.7 * security + 0.3 * utility
        individual_success = success_rate
        individual_utility = (
            utility_score if utility_score is not None else 0.0
        )

        # Use adaptive reward computation (handles GPT enabled/disabled cases)
        reward = compute_adaptive_reward(
            security_score=success_rate,
            utility_score=utility_score,
            gpt_score=prob_1 if self.gpt_enabled else None,
            gpt_config=gpt_config_dict,
        )

        task_id = self.current_injection_task.ID

        # Update best suffix if this one is better
        if reward > self.best_reward:
            self.best_suffix = self.current_suffix
            self.best_reward = reward
            if self.verbose:
                logger.info(
                    f"Updated best suffix for ({self.current_user_task.ID}, {task_id}): "
                    f"reward {self.best_reward:.3f}"
                )

        # Update last suffix for next iteration comparison
        self.last_suffix = self.current_suffix

        # Add 3-tuple experience
        self.all_experiences.append(
            (self.current_prompt, self.current_suffix, reward)
        )

        # Save experience to file
        save_experience_to_file(
            self.experience_file,
            len(self.all_experiences),
            self.current_prompt,
            self.current_suffix,
            reward,
            individual_success,
            individual_utility,
            self.current_formatted_goal,
            prob_0=prob_0,
            prob_1=prob_1,
            task_id=task_id,
            user_task=self.current_user_task.ID,
            is_better_than_previous=is_better,
            gpt_reasoning=reasoning,
        )

        # Save summary
        self._save_summary()

        # Log experience state
        log_experience_state(
            self.all_experiences,
            1,  # Single task only
            self.verbose,
        )

    def _save_summary(self) -> None:
        """Save comprehensive summary for single task pair."""
        import json

        summary = {
            "last_updated": datetime.now().isoformat(),
            "total_experiences": len(self.all_experiences),
            "config": {
                "max_suffix_length": self.max_suffix_length,
                "min_suffix_length": self.min_suffix_length,
                "mutation_rate": self.mutation_rate,
                "use_self_transfer": self.use_self_transfer,
                "gpt_enabled": self.gpt_enabled,
            },
        }

        # Add current task pair info if available
        if self.current_user_task and self.current_injection_task:
            summary["task_pair"] = {
                "user_task_id": self.current_user_task.ID,
                "injection_task_id": self.current_injection_task.ID,
            }

        # Add best suffix info
        if self.best_suffix is not None:
            summary["best_suffix"] = {
                "reward": self.best_reward,
                "suffix_preview": (
                    self.best_suffix[:50] + "..."
                    if len(self.best_suffix) > 50
                    else self.best_suffix
                ),
            }

        # Add reward statistics
        if self.all_experiences:
            all_rewards = [exp[2] for exp in self.all_experiences]
            summary["reward_stats"] = {
                "mean": float(np.mean(all_rewards)),
                "std": float(np.std(all_rewards)),
                "min": float(np.min(all_rewards)),
                "max": float(np.max(all_rewards)),
                "recent_mean": float(np.mean(all_rewards[-20:])),
            }

        with open(self.summary_file, "w") as f:
            json.dump(summary, f, indent=2)
