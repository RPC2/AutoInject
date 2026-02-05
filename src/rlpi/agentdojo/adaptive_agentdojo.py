import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import hydra
from omegaconf import DictConfig
from tqdm import tqdm

from agentdojo.logging import Logger, TraceLogger
from agentdojo.models import ModelsEnum
from agentdojo.task_suite.load_suites import get_suite
from agentdojo.task_suite.task_suite import TaskSuite
from rlpi.agentdojo.utils import (
    calculate_average_scores,
    execute_single_benchmark,
    get_user_tasks,
    get_wrapped_injection_tasks,
    setup_pipeline_and_attacker,
)
from rlpi.attack.learner_factory import get_attack_learner
from rlpi.attack.utils import set_seed

# Configure logging
logger = logging.getLogger(__name__)


def _find_latest_checkpoint(logdir: Path) -> Optional[Path]:
    """Find the checkpoint file in the log directory.

    Args:
        logdir: Log directory to search

    Returns:
        Path to checkpoint state file, or None if not found
    """
    # Look for single checkpoint state file
    checkpoint_file = logdir / "checkpoint_state.json"
    if checkpoint_file.exists():
        return checkpoint_file

    # Fallback to final_policy_state.json for backwards compatibility
    final_checkpoint = logdir / "final_policy_state.json"
    if final_checkpoint.exists():
        return final_checkpoint

    return None


def _save_checkpoint(learner, iteration, queries_used, logdir):
    """Save checkpoint to a single checkpoint file (overwrites previous).

    Args:
        learner: The learner instance
        iteration: Current iteration number
        queries_used: Total queries used so far
        logdir: Log directory path
    """
    checkpoint_path = f"{logdir}/checkpoint.pt"
    if hasattr(learner, "save_model"):
        # Try to pass queries_used if the method supports it
        import inspect

        sig = inspect.signature(learner.save_model)
        if "queries_used" in sig.parameters:
            learner.save_model(checkpoint_path, queries_used=queries_used)
        else:
            learner.save_model(checkpoint_path)
        logger.info(
            f"Saved checkpoint to {checkpoint_path} (iteration={iteration + 1}, queries_used={queries_used})"
        )
    else:
        logger.warning(
            "Learner does not support save_model, skipping checkpoint"
        )


def run_adaptive_attack(
    cfg: DictConfig,
    suite: TaskSuite,
    model: ModelsEnum,
    logdir: Path,
    user_tasks: Optional[List[str]] = None,
    injection_tasks: Optional[List[str]] = None,
):
    """Run adaptive attack learning with a single pipeline instance."""
    query_budget = cfg.query_budget  # New parameter instead of num_iterations
    attack = cfg.attack
    defense = cfg.defense
    system_message = cfg.system_message
    learner_type = cfg.learner.type

    # Get tasks first so we can pass them to setup
    user_tasks_to_run = get_user_tasks(suite, user_tasks)
    wrapped_tasks_dict = get_wrapped_injection_tasks(
        suite, injection_tasks, cfg
    )

    pipeline, attacker, learner = _setup_pipeline_and_components(
        cfg=cfg,
        model=model,
        defense=defense,
        system_message=system_message,
        attack=attack,
        suite=suite,
        injection_tasks=list(wrapped_tasks_dict.values()),
    )

    # TRL suffix, adaptive random suffix, and LLM inference learners only support single (user_task, injection_task) pair
    # trl_suffix_joint supports multiple pairs
    is_joint_training = learner_type == "trl_suffix_joint"

    if learner_type in [
        "trl_suffix",
        "adaptive_random_suffix",
        "llm_inference",
    ]:
        if len(user_tasks_to_run) != 1:
            raise ValueError(
                f"{learner_type} learner only supports exactly 1 user task, "
                f"but got {len(user_tasks_to_run)} user tasks. "
                f"Please specify a single user task in your configuration."
            )
        if len(wrapped_tasks_dict) != 1:
            raise ValueError(
                f"{learner_type} learner only supports exactly 1 injection task, "
                f"but got {len(wrapped_tasks_dict)} injection tasks. "
                f"Please specify a single injection task in your configuration."
            )

    # Check for existing checkpoint to resume from
    latest_checkpoint = _find_latest_checkpoint(logdir)
    queries_used = 0
    iteration = 0
    early_stop_triggered = False

    if latest_checkpoint is not None and hasattr(learner, "load_model"):
        logger.info(
            f"Found checkpoint: {latest_checkpoint}, attempting to resume..."
        )

        # Load checkpoint (will load both model and state)
        try:
            learner.load_model(str(latest_checkpoint))
        except Exception as e:
            raise RuntimeError(
                f"Failed to load checkpoint from {latest_checkpoint}: {e}. "
                "Checkpoint may be corrupted. Cannot proceed without valid checkpoint."
            ) from e

        # Extract queries_used from checkpoint state
        import json

        try:
            with open(latest_checkpoint, "r") as f:
                checkpoint_state = json.load(f)
                learner_state = checkpoint_state.get("learner_state", {})
                queries_used = learner_state.get("queries_used")

            # Get iteration from learner's experience_count
            # Note: experience_count is the number of experiences completed
            # We use it as the starting iteration number, so the next iteration will be experience_count + 1
            iteration = getattr(learner, "experience_count", 0)
            logger.info(
                f"Starting iteration counter from {iteration} (based on experience_count)"
            )

            if queries_used is None:
                raise ValueError(
                    f"Checkpoint {latest_checkpoint} is missing queries_used. "
                    "Cannot determine remaining query budget. Checkpoint may be incomplete."
                )

            if queries_used < 0:
                raise ValueError(
                    f"Checkpoint {latest_checkpoint} has invalid queries_used={queries_used}. "
                    "Checkpoint may be corrupted."
                )

            remaining_queries = query_budget - queries_used
            logger.info(
                f"Resuming from checkpoint: queries_used={queries_used}, "
                f"remaining={remaining_queries}, experience_count={iteration}"
            )

            if remaining_queries <= 0:
                logger.warning(
                    f"Checkpoint indicates all queries used ({queries_used} >= {query_budget}). "
                    "Nothing to resume."
                )
                return

        except json.JSONDecodeError as e:
            raise RuntimeError(
                f"Checkpoint file {latest_checkpoint} is corrupted (invalid JSON): {e}. "
                "Cannot proceed without valid checkpoint."
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"Failed to read checkpoint state from {latest_checkpoint}: {e}. "
                "Cannot proceed without valid checkpoint."
            ) from e
    else:
        logger.info("No checkpoint found, starting fresh training")

    # For joint training, set up task pairs in the learner
    if is_joint_training:
        if hasattr(learner, "task_pairs"):
            # Build all task pairs
            task_pairs = []
            for user_task in user_tasks_to_run:
                for (
                    injection_task_id,
                    wrapped_injection_task,
                ) in wrapped_tasks_dict.items():
                    # Store the wrapped task (InjectionTaskModifier) directly
                    # It has _original_goal attribute and delegates other attributes to the wrapped task
                    task_pairs.append((user_task, wrapped_injection_task))
            learner.task_pairs = task_pairs
            logger.info(
                f"Set up {len(task_pairs)} task pairs for joint training"
            )

    queries_used = 0
    iteration = 0
    early_stop_triggered = False

    with tqdm(total=query_budget, desc="Running adaptive attack") as pbar:
        while queries_used < query_budget and not early_stop_triggered:
            # For each user task, modify the injection tasks with that context
            for user_task in user_tasks_to_run:
                # Check if we've reached the query budget
                if queries_used >= query_budget:
                    break

                # Set current user_task object before modify_tasks
                if hasattr(learner, "set_current_user_task_obj"):
                    learner.set_current_user_task_obj(user_task)

                # Modify tasks using learner with user task context
                learner.modify_tasks(
                    tasks=list(wrapped_tasks_dict.values()),
                    user_task=user_task.PROMPT,
                )

                # Run benchmarks for this user task
                task_utility_results, task_security_results = _run_benchmarks(
                    cfg=cfg,
                    user_tasks_to_run=[user_task],
                    wrapped_tasks_dict=wrapped_tasks_dict,
                    attacker=attacker,
                    pipeline=pipeline,
                    suite=suite,
                    logdir=logdir,
                    learner_type=learner_type,
                    iteration=iteration + 1,
                )

                # Update learner with scores for this user task
                asr, avg_utility = calculate_average_scores(
                    task_utility_results, task_security_results
                )

                logger.info(f"ASR: {asr}, Utility: {avg_utility}")

                # Increment query counter for main iteration benchmark (already executed)
                queries_used += 1
                pbar.update(1)

                # Update learner with scores (may trigger training)
                if is_joint_training:
                    # Update scores per pair (same for both universal and conditional modes)
                    for (
                        ut_id,
                        it_id,
                    ), utility in task_utility_results.items():
                        security = task_security_results.get(
                            (ut_id, it_id), 0.0
                        )
                        learner.update_scores(
                            success_rate=security,
                            utility_score=utility,
                            user_task_id=ut_id,
                            injection_task_id=it_id,
                            user_task_str=user_task.PROMPT,
                        )
                    # In universal mode, aggregate scores after collecting all pairs
                    is_universal = getattr(
                        learner, "universal_suffix_mode", False
                    )
                    if is_universal and hasattr(
                        learner, "aggregate_universal_scores"
                    ):
                        learner.aggregate_universal_scores()
                else:
                    # For single-pair training, use aggregated scores
                    learner.update_scores(
                        success_rate=asr,
                        utility_score=avg_utility,
                    )

                # Accumulate queries used during training
                if hasattr(learner, "get_training_queries_used"):
                    training_queries = learner.get_training_queries_used()
                    if training_queries > 0:
                        queries_used += training_queries
                        pbar.update(training_queries)
                        # Reset counter for next training session
                        if hasattr(learner, "evaluator") and hasattr(
                            learner.evaluator, "reset_queries_used"
                        ):
                            learner.evaluator.reset_queries_used()

                # Save checkpoint periodically (overwrites previous checkpoint)
                # Save every 10 iterations to balance disk I/O and recovery capability
                if (iteration + 1) % 10 == 0:
                    _save_checkpoint(learner, iteration, queries_used, logdir)

                if hasattr(learner, "early_stopped") and learner.early_stopped:
                    logger.info(
                        f"Early stopping condition met during training. "
                        f"Stopping experiment at iteration {iteration + 1}."
                    )
                    early_stop_triggered = True
                    break

            iteration += 1

    # Save final checkpoint (same file as periodic checkpoints)
    _save_checkpoint(learner, iteration, queries_used, logdir)
    logger.info(f"Saved final checkpoint (queries_used={queries_used})")


def _setup_pipeline_and_components(
    cfg: DictConfig,
    model,
    defense,
    system_message,
    attack,
    suite,
    injection_tasks: List,
):
    """Set up the pipeline, attacker, and learner components."""
    # Use common utility to set up pipeline and attacker
    max_tokens = cfg.get("max_tokens", None)
    pipeline, attacker = setup_pipeline_and_attacker(
        model=model,
        defense=defense,
        system_message=system_message,
        attack=attack,
        suite=suite,
        max_tokens=max_tokens,
    )

    # Extract learner parameters from config, assuming they are under 'learner' key
    learner_params = dict(cfg.learner)
    learner_type = learner_params.pop("type")

    # Set log directory to current working directory if not specified
    if "log_dir" not in learner_params:
        learner_params["log_dir"] = str(Path(os.getcwd()))

    # Get the learner instance
    learner = get_attack_learner(
        model=cfg.model,
        learner_type=learner_type,
        suite=suite,
        **learner_params,
    )

    # If learner supports evaluation infrastructure, inject it
    if hasattr(learner, "set_evaluation_infrastructure"):
        learner.set_evaluation_infrastructure(
            pipeline=pipeline,
            suite=suite,
            attacker=attacker,
            injection_tasks=injection_tasks,
        )
        logger.info("Injected evaluation infrastructure into learner")

    return pipeline, attacker, learner


def _run_benchmarks(
    cfg: DictConfig,
    user_tasks_to_run,
    wrapped_tasks_dict,
    attacker,
    pipeline,
    suite,
    logdir,
    learner_type,
    iteration,
) -> Tuple[Dict[Tuple[str, str], float], Dict[Tuple[str, str], float]]:
    """Run benchmarks for all user tasks and injection tasks combinations."""
    suite_utility_results: Dict[Tuple[str, str], float] = {}
    suite_security_results: Dict[Tuple[str, str], float] = {}

    for user_task in user_tasks_to_run:
        for (
            injection_task_id,
            wrapped_injection_task,
        ) in wrapped_tasks_dict.items():
            task_injections = attacker.attack(
                user_task,
                wrapped_injection_task,
            )

            logger.info(
                f"\nIter {iteration} - Running benchmark: User Task: {user_task.ID}, "
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
                attack_type=f"{cfg.attack}_{learner_type}_iter{iteration}",
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

                suite_utility_results[(user_task.ID, injection_task_id)] = (
                    utility
                )
                suite_security_results[(user_task.ID, injection_task_id)] = (
                    security
                )

    return suite_utility_results, suite_security_results


@hydra.main(
    version_base="1.1",
    config_path="config",
    config_name="config",
)
def main(cfg: DictConfig):
    """Main entry point for the adaptive attack experiment."""
    print("Configuration used:")
    print(cfg)  # Print the loaded config

    # Set random seed for reproducibility
    set_seed(cfg.seed)

    task_suite = get_suite("v1.2", cfg.suite)

    # Convert to list, preserving empty lists (don't convert to None)
    # None means "use all tasks", empty list means "use no tasks"
    user_tasks_list = (
        list(cfg.user_tasks) if cfg.user_tasks is not None else None
    )
    injection_tasks_list = (
        list(cfg.injection_tasks) if cfg.injection_tasks is not None else None
    )

    # Log directory is automatically handled by Hydra in its output directory
    logdir = Path(os.getcwd())
    print(f"Logging outputs to Hydra directory: {logdir}")

    # Run the adaptive attack
    run_adaptive_attack(
        cfg=cfg,
        suite=task_suite,
        model=ModelsEnum(cfg.model),
        logdir=logdir,
        user_tasks=user_tasks_list,
        injection_tasks=injection_tasks_list,
    )


if __name__ == "__main__":
    main()
