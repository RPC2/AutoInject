#!/bin/bash
# Benchmark Banking Suite Script
# This script runs all user tasks in the default banking suite without any attack

# ==============================================================================
# Configuration Variables (edit these to customize your run)
# ==============================================================================

MODELS=(
    # "google/gemini-2.0-flash-001"
    # "google/gemini-2.5-flash"
    # "gpt-4.1-nano"
    "gpt-4o-mini-2024-07-18"
)

BENCHMARK_VERSION="v1.2"

LOGDIR="<log_dir>"

FORCE_RERUN=false

SUITE="banking"

# ==============================================================================
# API Keys and Environment
# ==============================================================================
export OPENAI_API_KEY=$(cat ~/.rlpi_openai_key)

# ==============================================================================
# Run Command
# ==============================================================================

cd ./agentdojo

# Loop through each model and run the benchmark sequentially
for MODEL in "${MODELS[@]}"; do
    echo "=============================================================================="
    echo "Running benchmark for model: ${MODEL}"
    echo "=============================================================================="

    # Build the command
    # Note: We intentionally do NOT provide --attack, which defaults to None.
    # When attack is None, benchmark_suite_without_injections is called,
    # which ONLY runs user tasks and does NOT run any injection tasks.
    CMD="python -m agentdojo.scripts.benchmark"
    CMD="${CMD} --suite ${SUITE}"
    CMD="${CMD} --model ${MODEL}"
    CMD="${CMD} --benchmark-version ${BENCHMARK_VERSION}"
    CMD="${CMD} --logdir ${LOGDIR}"
    # We do NOT add --attack, so it defaults to None (no attack, no injection tasks)
    # We do NOT add --user-task, so all user tasks are run by default

    # Add force-rerun flag if enabled
    if [ "${FORCE_RERUN}" = true ]; then
        CMD="${CMD} --force-rerun"
    fi

    # Print the command being executed
    echo "Running benchmark for ${SUITE} suite (all user tasks, no attack)"
    echo "Command: ${CMD}"
    echo ""

    # Execute the command
    eval ${CMD}

    # Check if the command was successful
    if [ $? -ne 0 ]; then
        echo "ERROR: Benchmark failed for model ${MODEL}"
        echo "Continuing with next model..."
    fi

    echo ""
    echo "Completed benchmark for model: ${MODEL}"
    echo ""
done

echo "=============================================================================="
echo "All benchmarks completed!"
echo "=============================================================================="

