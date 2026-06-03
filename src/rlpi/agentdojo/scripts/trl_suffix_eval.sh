#!/bin/bash
# TRL Suffix Attack Training Script - Evaluation Dataset
# This script runs the TRL-based suffix attack learner on the evaluation dataset
# Task selections based on the evaluation dataset specification:
# - Workspace: User tasks 0, 2, 3, 21, 23; Injection tasks 0, 1, 3, 5
# - Banking: User tasks 0, 1, 2, 6, 13; Injection tasks 4, 6, 7, 8
# - Travel: User tasks 0, 1, 2, 5, 13; Injection tasks 0, 1, 3, 4
# - Slack: User tasks 0, 1, 3, 4, 8; Injection tasks 1, 2, 3, 4
# Total: 80 task pairs (5 user tasks × 4 injection tasks × 4 suites)

# ==============================================================================
# Configuration Variables
# ==============================================================================

MODEL="Qwen/Qwen3-4B"
# MODEL="google/gemma-3-4b-it"
EXP_IDENT="${MODEL}_trl"

QUERY_BUDGET=260

# Model configuration
# For qwen3-4b: context limit is 32768. With typical input tokens (1000-2000),
# set max_tokens to leave room: 32768 - 2000 = ~30000 max safe value
# Using 28000 to be conservative and account for variable input sizes
MAX_TOKENS=28000  # Maximum tokens for qwen3-4b (context limit: 32768, accounting for input tokens)
# For other models, adjust accordingly:
# MAX_TOKENS=4000   # For models with smaller context windows
# MAX_TOKENS=null   # To use model defaults

# Suffix generation parameters
MAX_SUFFIX_LENGTH=30  # Token limit for generated suffixes
TEMPERATURE=0.9       # Sampling temperature (higher = more diverse)
TOP_P=0.95           # Nucleus sampling threshold

# GRPO training parameters
GRPO_NUM_GENERATIONS=8   # Suffixes generated per prompt per iteration
GRPO_NUM_ITERATIONS=5   # Training iterations per session
GRPO_LEARNING_RATE=1e-5  # Policy learning rate
GRPO_BATCH_SIZE=2        # Per-device batch size

# ==============================================================================
# Suite Configuration - Evaluation Dataset Tasks
# ==============================================================================

declare -A suite_user_tasks
declare -A suite_injection_tasks

# Workspace suite
suite_user_tasks[workspace]="[user_task_0],[user_task_2],[user_task_3],[user_task_21],[user_task_23]"
suite_injection_tasks[workspace]="[injection_task_0],[injection_task_1],[injection_task_3],[injection_task_5]"

# Banking suite
suite_user_tasks[banking]="[user_task_0],[user_task_1],[user_task_2],[user_task_6],[user_task_13]"
suite_injection_tasks[banking]="[injection_task_4],[injection_task_6],[injection_task_7],[injection_task_8]"

# Travel suite
suite_user_tasks[travel]="[user_task_0],[user_task_1],[user_task_2],[user_task_5],[user_task_13]"
suite_injection_tasks[travel]="[injection_task_0],[injection_task_1],[injection_task_3],[injection_task_4]"

# Slack suite
suite_user_tasks[slack]="[user_task_0],[user_task_1],[user_task_3],[user_task_4],[user_task_8]"
suite_injection_tasks[slack]="[injection_task_1],[injection_task_2],[injection_task_3],[injection_task_4]"

# ==============================================================================
# API Keys and Environment
# ==============================================================================
export OPENAI_API_KEY=$(cat ~/.rlpi_openai_key)
export TOGETHER_API_KEY=$(cat ~/.rlpi_togetherai_key)
export TOKENIZERS_PARALLELISM=false

# ==============================================================================
# Run Command - Loop through all suites, user tasks, and injection tasks sequentially
# ==============================================================================

total_experiments=0
completed_experiments=0

# for suite in workspace banking travel slack; do
for suite in travel; do
    echo "=========================================="
    echo "Starting suite: ${suite}"
    echo "=========================================="

    # Run sequentially (no & and no --multirun)
    python -m rlpi.agentdojo.adaptive_agentdojo \
        exp_ident=${EXP_IDENT}_${suite} \
        model=$MODEL \
        suite=$suite \
        query_budget=$QUERY_BUDGET \
        max_tokens=$MAX_TOKENS \
        user_tasks=${suite_user_tasks[$suite]} \
        injection_tasks=${suite_injection_tasks[$suite]} \
        learner=trl_suffix \
        learner.max_suffix_length=$MAX_SUFFIX_LENGTH \
        learner.temperature=$TEMPERATURE \
        learner.top_p=$TOP_P \
        learner.grpo_num_generations=$GRPO_NUM_GENERATIONS \
        learner.grpo_num_iterations=$GRPO_NUM_ITERATIONS \
        learner.grpo_learning_rate=$GRPO_LEARNING_RATE \
        learner.grpo_per_device_train_batch_size=$GRPO_BATCH_SIZE \
        learner.gpt_enabled=true \
        --multirun &

    echo "Completed suite: ${suite}"
    echo ""
done

echo "=========================================="
echo "All experiments completed!"
echo "Total: ${total_experiments} task pairs across 4 suites"
echo "=========================================="
