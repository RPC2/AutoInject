from typing import Generic, List, Protocol, TypeVar


class TaskModifier(Protocol):
    """Protocol defining the interface for modifying tasks.

    Any benchmark system should implement this protocol to work with the attack learner.
    """

    def modify_goal(self, token: str) -> None:
        """Modify the task's goal with the given token.

        Args:
            token: The token to append to the goal
        """
        ...


T = TypeVar("T", bound=TaskModifier)


class AdaptiveAttackLearner(Generic[T]):
    """A learner that adaptively modifies tasks to maximize attack success rate.

    The learner is generic and can work with any task type that implements the TaskModifier
    protocol. This makes it flexible enough to work with different benchmark systems.

    Args:
        learning_rate: Rate at which to update token scores
    """

    def __init__(
        self,
        learning_rate: float = 0.1,
    ):
        self.learning_rate = learning_rate

    def modify_tasks(self, tasks: List[T]) -> List[T]:
        """Modify tasks with current best attack tokens.

        Args:
            tasks: List of tasks to modify

        Returns:
            Modified tasks
        """
        raise NotImplementedError(
            "This method should be implemented by the subclass"
        )

    def update_scores(self, success_rate: float) -> None:
        """Update token effectiveness scores based on benchmark results.

        Args:
            success_rate: The success rate of the last attack attempt
        """
        raise NotImplementedError(
            "This method should be implemented by the subclass"
        )
