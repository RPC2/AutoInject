"""Statistics tracking and logging utilities for reward functions and evaluators."""

import json
import logging
from pathlib import Path
from typing import Dict, Optional, Union

logger = logging.getLogger("RewardStats")


def log_reward_stats(
    reward_stats: Dict[str, int],
    evaluator_stats: Optional[Dict[str, int]] = None,
    prefix: str = "[STATS]",
) -> None:
    """Log comprehensive reward function and evaluator statistics.

    Args:
        reward_stats: Dictionary with reward function statistics
        evaluator_stats: Optional dictionary with evaluator statistics
        prefix: Log message prefix
    """
    # Log reward function stats
    logger.info(
        f"{prefix} Reward Function: "
        f"Total calls: {reward_stats['total_calls']}, "
        f"Total completions: {reward_stats['total_completions']}, "
        f"Evaluations: {reward_stats['new_evaluations']}"
    )

    # Log evaluator stats if provided
    if evaluator_stats:
        logger.info(
            f"{prefix} Evaluator: "
            f"Total calls: {evaluator_stats['total_evaluate_suffix_calls']}, "
            f"Real evaluations: {evaluator_stats['real_evaluations']}"
        )


def save_reward_stats(
    output_file: Path,
    reward_stats: Dict[str, int],
    evaluator_stats: Optional[Dict[str, int]] = None,
    training_session: Optional[int] = None,
) -> None:
    """Save reward function and evaluator statistics to JSON file.

    Args:
        output_file: Path to output JSON file
        reward_stats: Dictionary with reward function statistics
        evaluator_stats: Optional dictionary with evaluator statistics
        training_session: Optional training session number
    """
    stats: Dict[str, Union[int, Dict[str, int], None]] = {
        "training_session": training_session,
        "reward_function": reward_stats.copy(),
    }

    if evaluator_stats:
        stats["evaluator"] = evaluator_stats.copy()

    # Append to file (one JSON object per line)
    with open(output_file, "a") as f:
        f.write(json.dumps(stats) + "\n")

    logger.info(f"[STATS] Saved statistics to {output_file}")


def log_stats_summary(
    reward_stats: Dict[str, int],
    evaluator_stats: Optional[Dict[str, int]] = None,
) -> str:
    """Generate a human-readable summary of statistics.

    Args:
        reward_stats: Dictionary with reward function statistics
        evaluator_stats: Optional dictionary with evaluator statistics

    Returns:
        Formatted string with statistics summary
    """
    lines = ["=" * 60, "REWARD FUNCTION & EVALUATOR STATISTICS", "=" * 60]

    # Reward function section
    lines.append("\nReward Function:")
    lines.append(f"  Total calls: {reward_stats['total_calls']}")
    lines.append(f"  Total completions: {reward_stats['total_completions']}")
    lines.append(f"  Evaluations: {reward_stats['new_evaluations']}")

    # Evaluator section
    if evaluator_stats:
        lines.append("\nEvaluator:")
        lines.append(
            f"  Total evaluate_suffix calls: {evaluator_stats['total_evaluate_suffix_calls']}"
        )
        lines.append(
            f"  Real evaluations: {evaluator_stats['real_evaluations']}"
        )

    lines.append("=" * 60)
    return "\n".join(lines)
