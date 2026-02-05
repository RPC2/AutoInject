"""
Reward function utilities for TRL suffix learner.
"""

import json
import logging
from datetime import datetime

# Import SuffixEvaluator for type hints (avoid circular import at module level)
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from agentdojo.base_tasks import BaseInjectionTask, BaseUserTask
from rlpi.attack.learners.common.feedback_utils import (
    compare_suffix_with_previous,
    compute_adaptive_reward,
)
from rlpi.attack.learners.common.token_utils import clean_generated_suffix

if TYPE_CHECKING:
    from rlpi.attack.learners.trl_suffix.evaluator import SuffixEvaluator

logger = logging.getLogger("TRLSuffixRewards")


def extract_injection_goal_from_prompt(prompt: str) -> str:
    """Extract the injection goal from the adversarial prompt.

    The prompt format is:
    Examples of adversarial suffixes:
    ...
    Generate a non-sensical adversarial suffix using random tokens...
    Original Goal: <GOAL_HERE>
    User Task: <USER_TASK>
    Suffix (use brackets, symbols, repeated tokens):
    """
    try:
        # Look for "Original Goal:" line
        lines = prompt.split("\n")
        for line in lines:
            if line.startswith("Original Goal:"):
                return line.replace("Original Goal:", "").strip()
        # Fallback: return empty string if not found
        return ""
    except Exception:
        return ""


def rank_normalize_rewards(rewards: List[float]) -> List[float]:
    """Convert rewards to normalized ranks.

    This function converts a list of rewards to their normalized ranks,
    where ranks are normalized to the range [-1, 1]. The lowest reward
    gets rank -1 and the highest reward gets rank 1.

    Args:
        rewards: List of reward values to rank-normalize

    Returns:
        List of normalized rank values in [-1, 1] range

    Examples:
        >>> rank_normalize_rewards([1.0, 2.0, 3.0])
        [-1.0, 0.0, 1.0]
        >>> rank_normalize_rewards([0.5, 0.5, 1.0])
        [-1.0, -1.0, 1.0]
    """
    if not rewards:
        return []
    if len(rewards) == 1:
        # Single reward: return 0.0 (neutral rank)
        return [0.0]

    # Convert to numpy array for efficient ranking
    rewards_array = np.array(rewards)
    # Get ranks (argsort of argsort gives ranks from 0 to n-1)
    ranks = np.argsort(np.argsort(rewards_array)).astype(float)
    # Normalize to [-1, 1]
    n = len(rewards)
    normalized_ranks = (ranks / (n - 1)) * 2 - 1
    return normalized_ranks.tolist()


def create_experience_based_reward_function(
    evaluator: Optional["SuffixEvaluator"],
    current_user_task: Optional[BaseUserTask],
    injection_task: Optional[BaseInjectionTask],
    feedback_model: str,
    gpt_config: Dict[str, Any],
    verbose: bool = True,
    training_session_id: int = 0,
    grpo_eval_file: Optional[str] = None,
) -> Callable[[List[str], List[str]], List[float]]:
    """Create reward function that ALWAYS evaluates on real tasks.

    Simplified for single task pair operation.

    Args:
        evaluator: SuffixEvaluator instance for real evaluation (REQUIRED)
        current_user_task: The single user task for this learner session
        injection_task: The single injection task for this learner session
        feedback_model: Model for GPT-based suffix quality evaluation
        gpt_config: GPT configuration dict
        verbose: Whether to log verbose information
        training_session_id: Training session number for organizing logs
        grpo_eval_file: Optional file path to save GRPO evaluation details

    Returns:
        Reward function that maps (completions, prompts) to rewards
    """

    # Track GRPO iteration for logging purposes
    iteration_counter = {"count": 0}

    # Track best suffix dynamically during GRPO training
    best_suffix_tracker: Dict[str, Optional[str] | float] = {
        "suffix": None,
        "reward": -float("inf"),
    }

    # Track early stopping (utility=1.0 and security=1.0)
    early_stop_info: Dict[str, Any] = {
        "triggered": False,
        "suffix": None,
        "utility": None,
        "security": None,
    }

    # Track reward function call statistics
    reward_stats = {
        "total_calls": 0,  # Total times reward function called
        "total_completions": 0,  # Total completions evaluated
        "new_evaluations": 0,  # Number of new evaluations
    }

    # Extract preloaded model/tokenizer from evaluator if available
    preloaded_model, preloaded_tokenizer = None, None
    if evaluator is not None:
        (
            preloaded_model,
            preloaded_tokenizer,
        ) = evaluator.get_preloaded_model_and_tokenizer()

    def experience_based_reward_function(
        completions: List[str], prompts: List[str], **kwargs
    ) -> List[float]:

        # Track function call
        reward_stats["total_calls"] += 1
        reward_stats["total_completions"] += len(completions)

        logger.info(
            f"[REWARD_FN] Call #{reward_stats['total_calls']}: "
            f"Evaluating {len(completions)} completions "
            f"(Total evaluated so far: {reward_stats['total_completions']})"
        )

        rewards = []

        for prompt, completion in zip(prompts, completions):
            # Get current iteration for logging (increments for each completion)
            within_session_iter = iteration_counter["count"]
            iteration_counter["count"] += 1

            # CRITICAL: Clean the completion before evaluation
            cleaned_completion = clean_generated_suffix(completion)

            # New evaluation required
            reward_stats["new_evaluations"] += 1

            # Format iteration as "session{N}_iter{M}" for unique directory naming
            iteration_label = (
                f"session{training_session_id}_iter{within_session_iter}"
            )

            # Evaluate with best suffix context
            current_best_suffix = best_suffix_tracker["suffix"]
            reward, gpt_details = _evaluate_on_real_tasks(
                completion=cleaned_completion,
                evaluator=evaluator,
                user_task=current_user_task,
                injection_task=injection_task,
                feedback_model=feedback_model,
                gpt_config=gpt_config,
                verbose=verbose,
                iteration=iteration_label,
                best_suffix=(
                    current_best_suffix
                    if isinstance(current_best_suffix, str)
                    else None
                ),
                preloaded_model=preloaded_model,
                preloaded_tokenizer=preloaded_tokenizer,
            )
            rewards.append(reward)

            # Check for early stopping condition (utility=1.0 and security=1.0)
            utility = gpt_details.get("utility", 0.0)
            security = gpt_details.get("security", 0.0)
            if (
                utility == 1.0
                and security == 1.0
                and not early_stop_info["triggered"]
            ):
                early_stop_info["triggered"] = True
                early_stop_info["suffix"] = cleaned_completion
                early_stop_info["utility"] = utility
                early_stop_info["security"] = security
                if verbose:
                    logger.info(
                        f"[EARLY_STOP] Triggered during GRPO training: "
                        f"utility={utility:.1f}, security={security:.1f}"
                    )

            # Save GRPO evaluation details to file if path provided
            if grpo_eval_file is not None:
                _save_grpo_eval_to_file(
                    file_path=grpo_eval_file,
                    training_session_id=training_session_id,
                    iteration_label=iteration_label,
                    prompt=prompt,
                    suffix=cleaned_completion,
                    reward=reward,
                    gpt_details=gpt_details,
                    user_task_id=(
                        current_user_task.ID if current_user_task else None
                    ),
                    injection_task_id=(
                        injection_task.ID if injection_task else None
                    ),
                )

            # Update best suffix tracker if current is better
            current_best_reward = best_suffix_tracker["reward"]
            if (
                isinstance(current_best_reward, (int, float))
                and reward > current_best_reward
            ):
                best_suffix_tracker["reward"] = reward
                best_suffix_tracker["suffix"] = cleaned_completion
                if verbose:
                    logger.info(
                        f"[GRPO] Updated best suffix in training session: "
                        f"reward={reward:.3f}"
                    )

        # Log summary statistics for this call
        logger.info(
            f"[REWARD_FN] Call #{reward_stats['total_calls']} complete: "
            f"Returned {len(rewards)} rewards "
            f"(Evaluations: {reward_stats['new_evaluations']})"
        )

        return rewards

    # Attach stats dictionary and early stop info to function for external access
    experience_based_reward_function.stats = reward_stats  # type: ignore
    experience_based_reward_function.early_stop_info = early_stop_info  # type: ignore

    return experience_based_reward_function


def _evaluate_on_real_tasks(
    completion: str,
    evaluator: Optional["SuffixEvaluator"],
    user_task: Optional[BaseUserTask],
    injection_task: Optional[BaseInjectionTask],
    feedback_model: Optional[str],
    gpt_config: Dict[str, Any],
    verbose: bool,
    iteration: Optional[str] = None,
    best_suffix: Optional[str] = None,
    preloaded_model: Optional[Any] = None,
    preloaded_tokenizer: Optional[Any] = None,
) -> Tuple[float, Dict[str, Any]]:
    """
    Evaluate suffix on single task pair. Always returns a valid reward.
    Raises an error if evaluation prerequisites are not met.

    Simplified for single task pair operation.

    Args:
        iteration: GRPO iteration label for logging (e.g., "session1_iter0")

    Returns:
        Tuple of (reward, gpt_details) where gpt_details contains:
            - prob_0, prob_1, is_better, reasoning from GPT comparison
            - utility, security scores
    """
    # Validate prerequisites
    if evaluator is None:
        raise ValueError(
            "Evaluator is None - call set_evaluation_infrastructure() first"
        )
    if user_task is None:
        raise ValueError(
            "User task is None - set current_user_task before training"
        )
    if injection_task is None:
        raise ValueError(
            "Injection task is None - call set_evaluation_infrastructure() first"
        )

    logger.info(
        f"[REAL_EVAL] Evaluating new generation with "
        f"(user_task={user_task.ID}, injection_task={injection_task.ID})"
    )

    utility, security = evaluator.evaluate_suffix(
        suffix=completion,
        user_task=user_task,
        injection_task=injection_task,
        attack_type="grpo_eval",
        iteration=iteration,
    )

    logger.info(
        f"[REAL_EVAL] Evaluation complete: utility={utility:.3f}, security={security:.3f}"
    )

    # Get optional GPT score using comparison (with full details)
    (
        gpt_score,
        prob_0,
        prob_1,
        is_better,
        reasoning,
    ) = _get_gpt_score_if_enabled(
        completion=completion,
        injection_goal=injection_task.task.GOAL,
        feedback_model=feedback_model,
        gpt_config=gpt_config,
        best_suffix=best_suffix,
        user_task_str=user_task.ID,
        preloaded_model=preloaded_model,
        preloaded_tokenizer=preloaded_tokenizer,
    )

    # Compute final reward
    reward = compute_adaptive_reward(
        security_score=security,
        utility_score=utility,
        gpt_score=gpt_score,
        gpt_config=gpt_config,
    )

    # Clamp reward
    reward = max(-2.0, min(2.0, reward))

    if verbose:
        logger.info(
            f"REAL reward: {reward:.3f} (sec: {security:.3f}, util: {utility:.3f}) "
            f"on injection_task={injection_task.ID}, user_task={user_task.ID}"
        )

    # Package GPT details for logging
    gpt_details = {
        "prob_0": prob_0,
        "prob_1": prob_1,
        "is_better": is_better,
        "reasoning": reasoning,
        "utility": utility,
        "security": security,
    }

    return reward, gpt_details


def _get_gpt_score_if_enabled(
    completion: str,
    injection_goal: str,
    feedback_model: Optional[str],
    gpt_config: Dict[str, Any],
    best_suffix: Optional[str] = None,
    user_task_str: str = "the user's task",
    preloaded_model: Optional[Any] = None,
    preloaded_tokenizer: Optional[Any] = None,
) -> Tuple[Optional[float], float, float, bool, str]:
    """Get GPT quality score using comparison-based feedback.

    Args:
        completion: Current suffix to evaluate
        injection_goal: Injection goal string
        feedback_model: Model to use for GPT feedback
        gpt_config: GPT configuration dict
        best_suffix: Best suffix seen so far (for comparison)
        user_task_str: User task description for comparison prompt

    Returns:
        Tuple of (gpt_score, prob_0, prob_1, is_better, reasoning)
        - gpt_score: Comparison score (0.0-1.0) or None if GPT disabled
        - prob_0: Probability that previous suffix is better
        - prob_1: Probability that new suffix is better
        - is_better: Boolean indicating if new is better
        - reasoning: GPT's detailed explanation
    """
    if not (gpt_config["enabled"] and feedback_model):
        return None, 0.0, 0.0, False, ""

    # If no best suffix yet, return neutral score
    if best_suffix is None:
        return 0.0, 0.0, 0.0, False, ""

    try:
        prob_1, prob_0, is_better, reasoning = compare_suffix_with_previous(
            current_suffix=completion,
            previous_suffix=best_suffix,
            user_task=user_task_str,
            injection_goal=injection_goal,
            model=feedback_model,
            verbose=False,  # Don't spam logs during GRPO
            preloaded_model=preloaded_model,
            preloaded_tokenizer=preloaded_tokenizer,
        )
        return (
            prob_1,
            prob_0,
            prob_1,
            is_better,
            reasoning,
        )  # Use prob_1 as gpt_score
    except Exception:
        return None, 0.0, 0.0, False, ""


def _save_grpo_eval_to_file(
    file_path: str,
    training_session_id: int,
    iteration_label: str,
    prompt: str,
    suffix: str,
    reward: float,
    gpt_details: Dict[str, Any],
    user_task_id: Optional[str] = None,
    injection_task_id: Optional[str] = None,
):
    """Save GRPO evaluation details to JSONL file.

    Args:
        file_path: Path to the GRPO evaluations file
        training_session_id: Training session number
        iteration_label: Iteration label (e.g., "session1_iter0")
        prompt: Original prompt/goal
        suffix: Generated suffix
        reward: Computed reward
        gpt_details: Dictionary with GPT feedback details
        user_task_id: User task ID
        injection_task_id: Injection task ID
    """
    entry = {
        "timestamp": datetime.now().isoformat(),
        "training_session_id": training_session_id,
        "iteration_label": iteration_label,
        "prompt": prompt,
        "suffix": suffix,
        "reward": float(reward),
        "success_rate": float(gpt_details.get("security", 0.0)),
        "utility_score": float(gpt_details.get("utility", 0.0)),
        "prob_0": float(gpt_details.get("prob_0", 0.0)),
        "prob_1": float(gpt_details.get("prob_1", 0.0)),
        "is_better_than_previous": bool(gpt_details.get("is_better", False)),
        "gpt_reasoning": gpt_details.get("reasoning", ""),
        "user_task_id": user_task_id,
        "injection_task_id": injection_task_id,
    }

    with open(file_path, "a") as f:
        json.dump(entry, f)
        f.write("\n")
