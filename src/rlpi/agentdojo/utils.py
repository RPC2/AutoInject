"""Common utility functions for agentdojo experiments."""

import logging
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml
from omegaconf import DictConfig

from agentdojo.agent_pipeline.agent_pipeline import (
    AgentPipeline,
    PipelineConfig,
)
from agentdojo.attacks.attack_registry import load_attack
from agentdojo.logging import Logger, TraceLogger
from agentdojo.models import ModelsEnum
from agentdojo.task_suite.task_suite import TaskSuite
from rlpi.attack.templates.task_modifier import (
    InjectionTaskModifier,
    ModifiedGoalAttack,
)
from rlpi.attack.templates.templates import AGENT_ATTACK_TEMPLATES

# Configure logging
logger = logging.getLogger(__name__)


def load_module_environment(module_name: str) -> bool:
    """Load a module and apply its environment variables to the current process.

    This function will never raise exceptions or stop execution. If the module
    cannot be loaded, it will log a message and return False, allowing the
    program to continue running.
    """
    try:
        # Use subprocess to run the module command and capture the environment
        # Add timeout to prevent hanging if module command is unavailable
        result = subprocess.run(
            f"module load {module_name} && env",
            shell=True,
            capture_output=True,
            text=True,
            executable="/bin/bash",
        )
        if result.returncode == 0:
            # Parse the environment variables and apply them to current process
            for line in result.stdout.strip().split("\n"):
                if "=" in line:
                    key, value = line.split("=", 1)
                    os.environ[key] = value
            logger.info(f"Successfully loaded module {module_name}")
            return True
        else:
            # Use info level instead of warning since module loading failure
            # is not critical and shouldn't stop execution
            logger.info(
                f"Could not load module {module_name} (this is non-fatal): {result.stderr}"
            )
            return False
    except Exception as e:
        # Catch any other exceptions to ensure program continues
        logger.info(
            f"Failed to load module {module_name} (this is non-fatal): {e}"
        )
        return False


# Try to load OpenAI API key from file if not already set
if "OPENAI_API_KEY" not in os.environ:
    try:
        openai_key_path = os.path.expanduser("~/.rlpi_openai_key")
        with open(openai_key_path, "r") as f:
            os.environ["OPENAI_API_KEY"] = f.read().strip()
        logger.info("Loaded OpenAI API key from ~/.rlpi_openai_key")
    except FileNotFoundError:
        logger.warning("OpenAI API key file not found at ~/.rlpi_openai_key")


def calculate_average_scores(
    suite_utility_results: Dict[Tuple[str, str], float],
    suite_security_results: Dict[Tuple[str, str], float],
) -> Tuple[float, float]:
    """Calculate average utility and security scores from results."""
    if suite_security_results:
        asr = sum(suite_security_results.values()) / len(
            suite_security_results
        )
    else:
        asr = 0.0

    if suite_utility_results:
        numeric_utilities = [
            u
            for u in suite_utility_results.values()
            if isinstance(u, (int, float))
        ]
        avg_utility = (
            sum(numeric_utilities) / len(numeric_utilities)
            if numeric_utilities
            else 0.0
        )
    else:
        avg_utility = 0.0

    return asr, avg_utility


def setup_pipeline_and_attacker(
    model: ModelsEnum,
    defense: Optional[str],
    system_message: Optional[str],
    attack: str,
    suite: TaskSuite,
    max_tokens: Optional[int] = None,
) -> Tuple[AgentPipeline, object]:
    """Set up the pipeline and attacker components (without learner)."""
    print(f"Creating pipeline with model: {model}")
    pipeline_config = PipelineConfig(
        llm=model,
        defense=defense,
        system_message_name=None,
        system_message=system_message or "You are a helpful assistant.",
        max_tokens=max_tokens,
    )
    pipeline = AgentPipeline.from_config(pipeline_config)
    if attack == "modified_goal":
        attacker = ModifiedGoalAttack(suite, pipeline)
    else:
        attacker = load_attack(attack, suite, pipeline)

    return pipeline, attacker


def get_user_tasks(suite: TaskSuite, user_tasks: Optional[List[str]]):
    """Get the user tasks to run based on configuration."""
    if user_tasks is not None:
        return [
            suite.get_user_task_by_id(user_task_id)
            for user_task_id in user_tasks
        ]
    else:
        return list(suite.user_tasks.values())


def get_wrapped_injection_tasks(
    suite: TaskSuite,
    injection_tasks: Optional[List[str]],
    cfg: Optional[DictConfig] = None,
):
    """Get the injection tasks wrapped in InjectionTaskModifier."""
    if injection_tasks is not None:
        raw_tasks = {
            task_id: suite.injection_tasks[task_id]
            for task_id in injection_tasks
        }
    else:
        raw_tasks = suite.injection_tasks

    templates = AGENT_ATTACK_TEMPLATES

    return {
        task_id: InjectionTaskModifier(task, templates=templates)
        for task_id, task in raw_tasks.items()
    }


def convert_to_numeric(value, value_type: str = "utility") -> float:
    """Convert a value to numeric format, handling different input types."""
    if isinstance(value, (bool, int, float)):
        return float(value)
    elif isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            logger.warning(
                f"Could not convert {value_type} '{value}' to float. Treating as 0.0"
            )
            return 0.0
    else:
        if value is not None:
            logger.warning(
                f"Non-numeric {value_type} '{value}' type {type(value)}. Treating as 0.0"
            )
        return 0.0


def execute_single_benchmark(
    pipeline: AgentPipeline,
    user_task,
    wrapped_injection_task,
    task_injections: List[str],
    suite: TaskSuite,
    trace_logger,
) -> Tuple[float, float]:
    """Execute a single benchmark task and return numeric utility and security scores."""
    try:
        utility, security = suite.run_task_with_pipeline(
            pipeline,
            user_task,
            wrapped_injection_task.task,
            task_injections,
        )
    except (
        yaml.scanner.ScannerError,
        yaml.parser.ParserError,
        yaml.constructor.ConstructorError,
        yaml.YAMLError,
    ) as e:
        # Catch YAML parsing errors that occur when generated suffixes break YAML format
        # Return failure scores to allow training to continue
        logger.error(
            f"YAML parsing error in benchmark execution: {e}\n"
            f"User Task: {user_task.ID}, Injection Task: {wrapped_injection_task.task.ID}\n"
            f"Task injections: {task_injections}"
        )
        utility, security = 0.0, 0.0

    numeric_utility = convert_to_numeric(utility, "utility")
    numeric_security = convert_to_numeric(security, "security")

    trace_logger.set_contextarg("utility", utility)
    trace_logger.set_contextarg("security", security)
    trace_logger.set_contextarg("numeric_utility", numeric_utility)
    trace_logger.set_contextarg("numeric_security", numeric_security)

    return numeric_utility, numeric_security


def run_benchmarks_with_attack_strings(
    user_tasks_to_run: List,
    wrapped_tasks_dict: Dict[str, InjectionTaskModifier],
    attack_strings: Dict[Tuple[str, str], List[str]],
    pipeline: AgentPipeline,
    suite: TaskSuite,
    logdir: Path,
    attack_type: str = "transfer",
) -> Tuple[Dict[Tuple[str, str], float], Dict[Tuple[str, str], float]]:
    """Run benchmarks using pre-computed attack strings.

    Args:
        user_tasks_to_run: List of user task objects
        wrapped_tasks_dict: Dictionary mapping injection task IDs to wrapped injection tasks
        attack_strings: Dictionary mapping (user_task_id, injection_task_id) to list of attack strings
        pipeline: AgentPipeline instance
        suite: TaskSuite instance
        logdir: Directory for logging
        attack_type: Type of attack for logging purposes

    Returns:
        Tuple of (utility_results, security_results) dictionaries
    """
    suite_utility_results: Dict[Tuple[str, str], float] = {}
    suite_security_results: Dict[Tuple[str, str], float] = {}

    for user_task in user_tasks_to_run:
        for (
            injection_task_id,
            wrapped_injection_task,
        ) in wrapped_tasks_dict.items():
            key = (user_task.ID, injection_task_id)

            # Get attack strings for this (user_task, injection_task) pair
            if key not in attack_strings:
                logger.warning(
                    f"No attack strings found for (user_task={user_task.ID}, "
                    f"injection_task={injection_task_id}). Skipping."
                )
                continue

            task_injections = attack_strings[key]

            logger.info(
                f"Running benchmark: User Task: {user_task.ID}, "
                f"Injection Task: {injection_task_id}"
            )

            delegate_logger = Logger.get()
            delegate_logger.logdir = logdir
            with TraceLogger(
                delegate=delegate_logger,
                suite_name=suite.name,
                user_task_id=user_task.ID,
                injection_task_id=injection_task_id,
                injections=task_injections,
                attack_type=attack_type,
                pipeline_name=pipeline.name,
            ) as trace_logger:
                utility, security = execute_single_benchmark(
                    pipeline,
                    user_task,
                    wrapped_injection_task,
                    task_injections,
                    suite,
                    trace_logger,
                )

                suite_utility_results[key] = utility
                suite_security_results[key] = security

    return suite_utility_results, suite_security_results
