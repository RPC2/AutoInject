"""
Utility functions for Adaptive Random Suffix learner.

Most utilities are reused from trl_suffix learner for consistency.
This module contains only learner-specific utilities.
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict

logger = logging.getLogger("AdaptiveRandomSuffixUtils")


def save_suffix_history(
    filepath: str, suffix_history: Dict[str, Dict[str, Any]]
):
    """Save suffix history to JSONL file.

    Args:
        filepath: Path to save the suffix history
        suffix_history: Dict mapping task_id to {suffix, reward, security, utility}
    """
    with open(filepath, "a") as f:
        entry = {
            "timestamp": datetime.now().isoformat(),
            "num_tasks": len(suffix_history),
            "history": {
                task_id: {
                    "suffix": hist["suffix"],
                    "reward": float(hist["reward"]),
                    "security": float(hist["security"]),
                    "utility": float(hist["utility"]),
                }
                for task_id, hist in suffix_history.items()
            },
        }
        json.dump(entry, f)
        f.write("\n")
