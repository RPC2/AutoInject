from typing import List, TypeVar

from rlpi.attack.learners.base import AdaptiveAttackLearner

T = TypeVar("T")


class NoopAttackLearner(AdaptiveAttackLearner[T]):
    """A learner that does nothing - doesn't modify tasks or update scores.

    This is useful as a baseline or for testing purposes when you want to run
    the benchmark without any adaptive learning.
    """

    def __init__(self, *args, **kwargs):
        # Initialize with empty token list but keep the interface compatible
        super().__init__(
            initial_tokens=[], learning_rate=0.0, exploration_rate=0.0
        )

    def modify_tasks(self, tasks: List[T], user_task: str) -> None:
        """No-op implementation that returns tasks unmodified."""
        pass

    def update_scores(self, success_rate: float, utility_score: float) -> None:
        """No-op implementation that doesn't update any scores."""
        pass

    def save_model(self, path: str) -> None:
        """No-op implementation as there's no model to save."""
        pass
