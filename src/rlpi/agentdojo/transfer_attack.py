"""Transfer attack strings from one model to another."""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple

import hydra
from omegaconf import DictConfig

from agentdojo.attacks.base_attacks import get_model_name_from_pipeline
from agentdojo.logging import Logger, TraceLogger
from agentdojo.models import ModelsEnum
from agentdojo.task_suite.load_suites import get_suite
from rlpi.agentdojo.utils import (
    calculate_average_scores,
    execute_single_benchmark,
    load_module_environment,
    setup_pipeline_and_attacker,
)
from rlpi.attack.learners.trl_suffix.utils import format_trl_suffix_prompt
from rlpi.attack.utils import set_seed

# Configure logging
logger = logging.getLogger(__name__)


def load_attack_strings_from_config(
    cfg: DictConfig,
    test_tasks: List[Dict[str, str]],
) -> Dict[Tuple[str, str], List[str]]:
    """Load attack strings from config for all test tasks.

    Args:
        cfg: Configuration dictionary containing model_attack_strings
        test_tasks: List of dicts with keys: suite, user_task, injection_task

    Returns:
        Dictionary mapping (user_task_id, injection_task_id) to list of attack strings

    Raises:
        ValueError: If source model is not found in model_attack_strings
    """
    source_model = cfg.source_model
    model_attack_strings = cfg.get("model_attack_strings", {})

    # Check if source model is in the available list
    if source_model not in model_attack_strings:
        available_models = list(model_attack_strings.keys())
        raise ValueError(
            f"Source model '{source_model}' not found in model_attack_strings. "
            f"Available models: {available_models}"
        )

    # Get attack string for source model (single string used for all tasks)
    source_attack_string = model_attack_strings[source_model]

    if not source_attack_string:
        raise ValueError(
            f"Attack string for source model '{source_model}' is empty or not set."
        )

    # Build mapping from (user_task, injection_task) to attack string
    # Use the same attack string for all tasks
    attack_strings = {}

    for task_config in test_tasks:
        user_task_id = task_config["user_task"]
        injection_task_id = task_config["injection_task"]
        key = (user_task_id, injection_task_id)

        # Use the same attack string for all tasks
        # Ensure it's a list
        if isinstance(source_attack_string, str):
            attack_strings[key] = [source_attack_string]
        else:
            attack_strings[key] = source_attack_string

    return attack_strings


def run_transfer_attack(
    cfg: DictConfig,
    logdir: Path,
) -> Dict[str, float]:
    """Run transfer attack experiment.

    Args:
        cfg: Configuration dictionary
        logdir: Directory for logging outputs

    Returns:
        Dictionary with summary statistics

    Raises:
        ValueError: If source model is not found in model_attack_strings
    """
    # Load test tasks
    test_tasks = list(cfg.test_tasks)

    # Load attack strings from config
    logger.info(
        f"Loading attack strings from config for source model: {cfg.source_model}"
    )
    attack_strings = load_attack_strings_from_config(
        cfg=cfg,
        test_tasks=test_tasks,
    )

    if not attack_strings:
        logger.error("No attack strings loaded. Exiting.")
        return {}

    logger.info(f"Loaded {len(attack_strings)} attack string(s)")

    # Group test tasks by suite
    tasks_by_suite: Dict[str, List[Dict[str, str]]] = {}
    for task in test_tasks:
        suite = task["suite"]
        if suite not in tasks_by_suite:
            tasks_by_suite[suite] = []
        tasks_by_suite[suite].append(task)

    # Run tests for each suite
    all_results = {}

    for suite_name, suite_tasks in tasks_by_suite.items():
        logger.info(f"Processing suite: {suite_name}")

        # Get suite instance
        task_suite = get_suite("v1.2", suite_name)

        # Filter attack strings to only those for this suite
        suite_attack_strings = {
            key: strings
            for key, strings in attack_strings.items()
            if any(
                key[0] == task["user_task"]
                and key[1] == task["injection_task"]
                for task in suite_tasks
            )
        }

        # Get only the specific task combinations that have attack strings
        # Create filtered lists that only include the exact (user_task, injection_task) pairs
        user_tasks_to_run: List = []
        wrapped_tasks_dict = {}

        for (
            user_task_id,
            injection_task_id,
        ), _ in suite_attack_strings.items():
            # Get user task if not already added
            user_task = next(
                (
                    task
                    for task in user_tasks_to_run
                    if task.ID == user_task_id
                ),
                None,
            )
            if user_task is None:
                user_task = task_suite.get_user_task_by_id(user_task_id)
                user_tasks_to_run.append(user_task)

            # Get wrapped injection task if not already added
            if injection_task_id not in wrapped_tasks_dict:
                injection_task = task_suite.get_injection_task_by_id(
                    injection_task_id
                )
                from rlpi.attack.templates.task_modifier import (
                    InjectionTaskModifier,
                )
                from rlpi.attack.templates.templates import (
                    AGENT_ATTACK_TEMPLATES,
                )

                wrapped_tasks_dict[injection_task_id] = InjectionTaskModifier(
                    injection_task, templates=AGENT_ATTACK_TEMPLATES
                )

        # Set up pipeline for target model
        target_model = ModelsEnum(cfg.target_model)
        # Use "modified_goal" attack for transfer attacks since we're using pre-formatted attack strings
        # This attack directly uses the injection task's GOAL without any template formatting
        attack_type = cfg.get("attack", "modified_goal")
        pipeline, attacker = setup_pipeline_and_attacker(
            model=target_model,
            defense=cfg.get("defense"),
            system_message=cfg.get("system_message"),
            attack=attack_type,
            suite=task_suite,
        )

        # Run benchmarks - only for exact (user_task, injection_task) pairs that have attack strings
        suite_utility_results: Dict[Tuple[str, str], float] = {}
        suite_security_results: Dict[Tuple[str, str], float] = {}

        # Create lookup dictionaries
        user_tasks_by_id = {task.ID: task for task in user_tasks_to_run}

        # Only iterate over the exact pairs that have attack strings
        for (
            user_task_id,
            injection_task_id,
        ), task_attack_strings in suite_attack_strings.items():
            if user_task_id not in user_tasks_by_id:
                logger.warning(
                    f"User task {user_task_id} not found. Skipping."
                )
                continue
            if injection_task_id not in wrapped_tasks_dict:
                logger.warning(
                    f"Injection task {injection_task_id} not found. Skipping."
                )
                continue

            user_task = user_tasks_by_id[user_task_id]
            wrapped_injection_task = wrapped_tasks_dict[injection_task_id]

            # Format the attack string with important instructions template
            # The attack strings from config are suffixes, similar to TRL suffix format
            if task_attack_strings:
                if isinstance(task_attack_strings, list):
                    if len(task_attack_strings) > 0:
                        attack_suffix = task_attack_strings[0]
                    else:
                        logger.warning(
                            f"Empty attack string list for (user_task={user_task_id}, "
                            f"injection_task={injection_task_id}). Skipping."
                        )
                        continue
                else:
                    attack_suffix = task_attack_strings

                # Get the original malicious goal from the injection task
                malicious_goal = wrapped_injection_task._original_goal

                # Get user name from suite environment (default to "Emma Johnson" like in important_instructions_attacks)
                # Get model name from pipeline
                user_name = "Emma Johnson"  # Default user name from important_instructions_attacks
                model_name = get_model_name_from_pipeline(pipeline)

                # Format using the important instructions template (same as TRL suffix learner)
                formatted_goal = format_trl_suffix_prompt(
                    malicious_goal=malicious_goal,
                    suffix=attack_suffix,
                    user_name=user_name,
                    model_name=model_name,
                )

                wrapped_injection_task.set_formatted_goal(formatted_goal)
            else:
                logger.warning(
                    f"No attack strings for (user_task={user_task_id}, "
                    f"injection_task={injection_task_id}). Skipping."
                )
                continue

            # Use attacker.attack() to get injection dict (same pattern as adaptive_agentdojo.py)
            task_injections = attacker.attack(
                user_task, wrapped_injection_task
            )

            # Get number of repeats from config (default 10)
            num_repeats = cfg.get("num_repeats", 10)

            logger.info(
                f"Running benchmark: User Task: {user_task_id}, "
                f"Injection Task: {injection_task_id}, "
                f"Repeats: {num_repeats}"
            )

            key = (user_task_id, injection_task_id)

            # Collect results from all repetitions
            utility_scores = []
            security_scores = []

            for repeat_idx in range(num_repeats):
                logger.info(
                    f"  Repeat {repeat_idx + 1}/{num_repeats} for "
                    f"User Task: {user_task_id}, Injection Task: {injection_task_id}"
                )

                delegate_logger = Logger.get()
                delegate_logger.logdir = logdir
                with TraceLogger(
                    delegate=delegate_logger,
                    suite_name=task_suite.name,
                    user_task_id=user_task_id,
                    injection_task_id=injection_task_id,
                    injections=task_injections,
                    attack_type=f"transfer_{cfg.source_model}_to_{cfg.target_model}",
                    pipeline_name=pipeline.name,
                ) as trace_logger:
                    utility, security = execute_single_benchmark(
                        pipeline,
                        user_task,
                        wrapped_injection_task,
                        task_injections,
                        task_suite,
                        trace_logger,
                    )

                    utility_scores.append(utility)
                    security_scores.append(security)

            # Average results across repetitions
            avg_utility = (
                sum(utility_scores) / len(utility_scores)
                if utility_scores
                else 0.0
            )
            avg_security = (
                sum(security_scores) / len(security_scores)
                if security_scores
                else 0.0
            )

            logger.info(
                f"  Averaged results over {num_repeats} repeats: "
                f"Utility: {avg_utility:.4f}, Security: {avg_security:.4f}"
            )

            suite_utility_results[key] = avg_utility
            suite_security_results[key] = avg_security

        # Calculate average scores
        asr, avg_utility = calculate_average_scores(
            suite_utility_results, suite_security_results
        )

        logger.info(
            f"Suite {suite_name} - ASR: {asr:.4f}, Utility: {avg_utility:.4f}"
        )

        all_results[f"{suite_name}_asr"] = asr
        all_results[f"{suite_name}_utility"] = avg_utility

        # Store per-task results
        for key, utility in suite_utility_results.items():
            security = suite_security_results.get(key, 0.0)
            all_results[f"{suite_name}_{key[0]}_{key[1]}_utility"] = utility
            all_results[f"{suite_name}_{key[0]}_{key[1]}_security"] = security

    # Calculate overall average
    suite_asrs = [v for k, v in all_results.items() if k.endswith("_asr")]
    suite_utilities = [
        v
        for k, v in all_results.items()
        if k.endswith("_utility") and not k.endswith("_asr")
    ]

    if suite_asrs:
        all_results["overall_asr"] = sum(suite_asrs) / len(suite_asrs)
    if suite_utilities:
        all_results["overall_utility"] = sum(suite_utilities) / len(
            suite_utilities
        )

    # Save results to JSON
    results_path = logdir / "transfer_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Saved results to {results_path}")

    return all_results


@hydra.main(
    version_base="1.1",
    config_path="config",
    config_name="transfer_config",
)
def main(cfg: DictConfig):
    """Main entry point for transfer attack experiment."""
    load_module_environment("eth_proxy")
    print("Configuration used:")
    print(cfg)

    # Set random seed for reproducibility
    set_seed(cfg.get("seed", 1))

    # Log directory is automatically handled by Hydra
    logdir = Path(os.getcwd())
    print(f"Logging outputs to Hydra directory: {logdir}")

    # Run transfer attack
    results = run_transfer_attack(cfg=cfg, logdir=logdir)

    if results:
        logger.info("Transfer attack completed successfully")
        logger.info(f"Overall ASR: {results.get('overall_asr', 0.0):.4f}")
        logger.info(
            f"Overall Utility: {results.get('overall_utility', 0.0):.4f}"
        )
    else:
        logger.error("Transfer attack failed - no results")

    return results.get("overall_asr", 0.0) if results else 0.0


if __name__ == "__main__":
    main()
