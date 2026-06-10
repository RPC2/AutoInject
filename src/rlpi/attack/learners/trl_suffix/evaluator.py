"""
Single-suffix evaluation infrastructure for reward function.

This evaluator runs individual generated suffixes against AgentDojo tasks
to provide real-time feedback during GRPO training.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from agentdojo.agent_pipeline import AgentPipeline
from agentdojo.agent_pipeline.agent_pipeline import LocalLLM
from agentdojo.attacks.base_attacks import BaseAttack
from agentdojo.base_tasks import BaseInjectionTask, BaseUserTask
from agentdojo.logging import Logger, TraceLogger
from agentdojo.task_suite.task_suite import TaskSuite
from rlpi.attack.learners.trl_suffix.utils import format_trl_suffix_prompt
from rlpi.attack.templates.task_modifier import InjectionTaskModifier

logger = logging.getLogger("SuffixEvaluator")


def _convert_to_numeric(value: Any, name: str) -> float:
    """Convert value to float, handling various types.

    Args:
        value: Value to convert (bool, int, float, or other)
        name: Name for logging (e.g., "utility", "security")

    Returns:
        Float value (0.0 or 1.0 for booleans, 0.0 for non-numeric)
    """
    if isinstance(value, (int, float)):
        return float(value)
    elif isinstance(value, bool):
        return 1.0 if value else 0.0
    elif isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            logger.warning(
                f"{name} is string '{value}' that cannot be converted to float, using 0.0"
            )
            return 0.0
    else:
        if value is not None:
            logger.warning(
                f"{name} is non-numeric type {type(value)}: {value}, using 0.0"
            )
        return 0.0


class SuffixEvaluator:
    """Evaluates a single generated suffix against AgentDojo tasks."""

    def __init__(
        self,
        pipeline: AgentPipeline,
        suite: TaskSuite,
        attacker: BaseAttack,
        user_name: str,
        victim_model_name: str,
        timeout: int = 60,  # Timeout for single evaluation
        logdir: Optional[Path] = None,  # Directory for saving evaluation logs
    ):
        """Initialize the suffix evaluator.

        Args:
            pipeline: The agent pipeline (victim model)
            suite: The task suite
            attacker: The attack strategy (usually ModifiedGoalAttack)
            user_name: User name for prompt formatting
            victim_model_name: Model name for prompt formatting
            timeout: Timeout in seconds for single evaluation (not currently enforced)
            logdir: Optional directory for saving TraceLogger outputs
        """
        self.pipeline = pipeline
        self.suite = suite
        self.attacker = attacker
        self.user_name = user_name
        self.victim_model_name = victim_model_name
        self.timeout = timeout
        self.logdir = logdir

        # Track evaluation statistics
        self.eval_stats = {
            "total_evaluate_suffix_calls": 0,  # Total calls to evaluate_suffix
            "real_evaluations": 0,  # Number of actual pipeline evaluations
        }

        # Track queries used (each evaluate_suffix call = 1 query)
        self.queries_used = 0

    def get_preloaded_model_and_tokenizer(
        self,
    ) -> Tuple[Optional[Any], Optional[Any]]:
        """Extract preloaded model and tokenizer from pipeline if it's a LocalLLM.

        Returns:
            Tuple of (model, tokenizer) or (None, None) if not a LocalLLM
        """
        # Pipeline is a list of elements, find the LocalLLM
        if hasattr(self.pipeline, "elements"):
            for element in self.pipeline.elements:
                if isinstance(element, LocalLLM):
                    return element.model, element.tokenizer
        return None, None

    def evaluate_suffix(
        self,
        suffix: str,
        user_task: BaseUserTask,
        injection_task: BaseInjectionTask,
        attack_type: str = "grpo_eval",
        iteration: Optional[str] = None,
    ) -> Tuple[float, float]:
        """
        Evaluate a single suffix on one task.

        Args:
            suffix: The generated suffix to evaluate
            user_task: The user task object (provides context)
            injection_task: The injection task object (attack goal)
            attack_type: Type of attack for logging (e.g., "grpo_eval", "final_eval")
            iteration: Optional iteration label for tracking training progress (e.g., "session1_iter0")

        Returns:
            (utility_score, security_score) tuple of floats in range [0.0, 1.0]
        """
        # Track function call
        self.eval_stats["total_evaluate_suffix_calls"] += 1
        self.eval_stats["real_evaluations"] += 1

        logger.info(
            f"[EVALUATOR] Evaluation #{self.eval_stats['real_evaluations']}: "
            f"Running AgentDojo pipeline on (user_task={user_task.ID}, "
            f"injection_task={injection_task.ID})"
        )

        # Format the goal with the suffix
        formatted_goal = format_trl_suffix_prompt(
            malicious_goal=injection_task.task.GOAL,
            suffix=suffix,
            user_name=self.user_name,
            model_name=self.victim_model_name,
        )

        # Create task modifier and set formatted goal
        modifier = InjectionTaskModifier(injection_task)
        modifier.set_formatted_goal(formatted_goal)

        # Get injections from attacker
        task_injections = self.attacker.attack(user_task, modifier)

        # Run the task via suite API
        logger.info(
            f"[EVALUATOR] Calling suite.run_task_with_pipeline with "
            f"user_task={user_task.ID}, injection_task={injection_task.ID}"
        )

        logger.info(f"[EVALUATOR] Generated suffix: {suffix}")

        # Use TraceLogger if logdir is available
        # TODO: Check if the attacks are actually logged
        if self.logdir:
            delegate_logger = Logger.get()
            delegate_logger.logdir = self.logdir

            # Build attack_type string with iteration info
            if iteration is not None:
                attack_type_str = f"{attack_type}_{iteration}"
            else:
                attack_type_str = attack_type

            with TraceLogger(
                delegate=delegate_logger,
                suite_name=self.suite.name,
                user_task_id=user_task.ID,
                injection_task_id=injection_task.ID,
                injections=task_injections,
                attack_type=attack_type_str,
                pipeline_name=self.pipeline.name,
            ) as trace_logger:
                try:
                    # Track query usage: each pipeline call = 1 query
                    self.queries_used += 1
                    utility, security = self.suite.run_task_with_pipeline(
                        self.pipeline,
                        user_task,
                        modifier.task,
                        task_injections,
                    )
                except Exception as e:
                    # Catch all errors (JSON parsing, YAML parsing, tool call errors, etc.)
                    # and return failure scores to allow training to continue
                    import traceback

                    error_type = type(e).__name__
                    logger.error(
                        f"[EVALUATOR] Error during evaluation ({error_type}): {e}"
                    )
                    logger.error(f"[EVALUATOR] Problematic suffix: {suffix}")
                    logger.error(
                        f"[EVALUATOR] User Task: {user_task.ID}, Injection Task: {injection_task.ID}"
                    )
                    logger.debug(
                        f"[EVALUATOR] Full traceback:\n{traceback.format_exc()}"
                    )
                    # Return failure scores (utility=0, security=0) to allow training to continue
                    utility, security = 0.0, 0.0

                logger.info(
                    f"[EVALUATOR] Pipeline returned utility={utility}, security={security} for "
                    f"(user_task={user_task.ID}, injection_task={injection_task.ID})"
                )

                # Convert to numeric (handles bool, int, float, str, None)
                numeric_utility = _convert_to_numeric(utility, "utility")
                numeric_security = _convert_to_numeric(security, "security")

                # Set context args for trace logger
                trace_logger.set_contextarg("utility", utility)
                trace_logger.set_contextarg("security", security)
                trace_logger.set_contextarg("numeric_utility", numeric_utility)
                trace_logger.set_contextarg(
                    "numeric_security", numeric_security
                )
                trace_logger.set_contextarg("suffix", suffix)
                trace_logger.set_contextarg(
                    "injection_goal", injection_task.task.GOAL
                )
        else:
            raise ValueError("No logdir provided for evaluation")

        logger.debug(
            f"Evaluated suffix on injection={injection_task.ID}, user={user_task.ID}: "
            f"utility={numeric_utility:.3f}, security={numeric_security:.3f}"
        )

        return numeric_utility, numeric_security

    def get_eval_stats(self) -> Dict[str, int]:
        """Get comprehensive evaluation statistics.

        Returns:
            Dictionary with detailed evaluation statistics including:
            - total_evaluate_suffix_calls: Total calls to evaluate_suffix
            - real_evaluations: Number of actual pipeline evaluations
        """
        return self.eval_stats.copy()

    def log_eval_stats(self) -> None:
        """Log comprehensive evaluation statistics."""
        stats = self.get_eval_stats()
        logger.info(
            f"[EVALUATOR STATS] "
            f"Total calls: {stats['total_evaluate_suffix_calls']}, "
            f"Real evaluations: {stats['real_evaluations']}, "
            f"Queries used: {self.queries_used}"
        )

    def get_queries_used(self) -> int:
        """Get total queries used by this evaluator.

        Returns:
            Number of queries made via evaluate_suffix() calls
        """
        return self.queries_used

    def reset_queries_used(self) -> int:
        """Reset query counter and return previous value.

        Returns:
            Previous value of queries_used before reset
        """
        prev = self.queries_used
        self.queries_used = 0
        return prev
