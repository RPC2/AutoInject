from typing import Dict, Generic, List, Optional, Protocol, TypeVar

import numpy as np


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
    """A learner that adaptively selects attack tokens to maximize success rate.

    This class implements an epsilon-greedy strategy to learn which tokens are most effective
    at achieving successful attacks. It maintains scores for each token and updates them
    based on the success rate of attacks.

    The learner is generic and can work with any task type that implements the TaskModifier
    protocol. This makes it flexible enough to work with different benchmark systems.

    Args:
        initial_tokens: List of tokens to start with
        learning_rate: Rate at which to update token scores
        exploration_rate: Probability of exploring random tokens
    """

    def __init__(
        self,
        initial_tokens: List[str],
        learning_rate: float = 0.1,
        exploration_rate: float = 0.1,
    ):
        self.attack_tokens = initial_tokens
        self.token_scores = {token: 0.0 for token in initial_tokens}
        self.learning_rate = learning_rate
        self.exploration_rate = exploration_rate
        self._last_used_token: Optional[str] = None

    def modify_tasks(self, tasks: List[T]) -> List[T]:
        """Modify tasks with current best attack tokens.

        Args:
            tasks: List of tasks to modify

        Returns:
            Modified tasks
        """
        selected_token = self._select_token()
        for task in tasks:
            task.modify_goal(selected_token)
        return tasks

    def update_scores(self, success_rate: float) -> None:
        """Update token effectiveness scores based on benchmark results.

        Args:
            success_rate: The success rate of the last attack attempt
        """
        if self._last_used_token is not None:
            self.token_scores[self._last_used_token] = (
                1 - self.learning_rate
            ) * self.token_scores[
                self._last_used_token
            ] + self.learning_rate * success_rate

    def _select_token(self) -> str:
        """Select token using epsilon-greedy strategy.

        Returns:
            Selected token
        """
        if not self.attack_tokens:
            raise ValueError("No attack tokens available")

        if np.random.random() < self.exploration_rate:
            self._last_used_token = np.random.choice(self.attack_tokens)
        else:
            if not self.token_scores:
                raise ValueError("No token scores available")
            self._last_used_token = max(
                self.token_scores.items(), key=lambda x: x[1]
            )[0]

        assert self._last_used_token is not None
        return self._last_used_token

    def get_best_token(self) -> str:
        """Get the token with the highest success rate.

        Returns:
            Best performing token
        """
        return max(self.token_scores.items(), key=lambda x: x[1])[0]

    def get_token_scores(self) -> Dict[str, float]:
        """Get current scores for all tokens.

        Returns:
            Dictionary mapping tokens to their scores
        """
        return self.token_scores.copy()

    def add_token(self, token: str) -> None:
        """Add a new token to the attack set.

        Args:
            token: New token to add
        """
        if token not in self.attack_tokens:
            self.attack_tokens.append(token)
            self.token_scores[token] = 0.0

    def remove_token(self, token: str) -> None:
        """Remove a token from the attack set.

        Args:
            token: Token to remove
        """
        if token in self.attack_tokens:
            self.attack_tokens.remove(token)
            del self.token_scores[token]
            if self._last_used_token == token:
                self._last_used_token = None
