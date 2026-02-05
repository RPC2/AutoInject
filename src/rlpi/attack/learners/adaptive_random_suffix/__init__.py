"""
Adaptive Random Suffix Attack Learner.

A baseline approach for adversarial suffix generation using random token-level
search with self-transfer.
"""

from rlpi.attack.learners.adaptive_random_suffix.learner import (
    AdaptiveRandomSuffixLearner,
)

__all__ = ["AdaptiveRandomSuffixLearner"]
