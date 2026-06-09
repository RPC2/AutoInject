#!/bin/bash
# TRL Suffix Joint Training Script
# This script runs joint training on multiple (user_task, injection_task) pairs
# and saves the model for transfer learning

# ==============================================================================
# Configuration Variables (edit these to customize your run)
# ==============================================================================

# MODEL="gpt-4.1-nano"
# MODEL="gpt-4o-mini-2024-07-18"
# MODEL="gpt-5-nano"
# MODEL="gpt-5"
MODEL="google/gemini-2.5-flash"
# MODEL="google/gemini-2.0-flash-001"
EXP_IDENT="${MODEL}_trl_joint_training"

SUITE="banking"
QUERY_BUDGET=200

# Define multiple task pairs for joint training
# Format: [user_task_X,user_task_Y,...] (single bracket, comma-separated)
# Example: Train on (user_task_1, injection_task_0) and (user_task_2, injection_task_1)
USER_TASKS="[user_task_0,user_task_1,user_task_2,user_task_3]"  # Multiple user tasks
INJECTION_TASKS="[injection_task_4]"  # Multiple injection tasks

# Suffix generation parameters
MAX_SUFFIX_LENGTH=30  # Token limit for generated suffixes
TEMPERATURE=0.9       # Sampling temperature (higher = more diverse)
TOP_P=0.95           # Nucleus sampling threshold

# GRPO training parameters
GRPO_NUM_GENERATIONS=8   # Suffixes generated per prompt per iteration
GRPO_NUM_ITERATIONS=5   # Training iterations per session
GRPO_LEARNING_RATE=1e-5  # Policy learning rate
GRPO_BATCH_SIZE=2        # Per-device batch size

# Universal suffix mode (set to true to train ONE universal suffix for all pairs)
# If false, trains conditional policy that generates different suffixes per pair
UNIVERSAL_SUFFIX_MODE=false # Set to true for universal suffix training
UNIVERSAL_REWARD_AGGREGATION="mean"  # How to aggregate: "mean", "min", "max", "median"

# ==============================================================================
# API Keys and Environment
# ==============================================================================
export OPENAI_API_KEY=$(cat ~/.rlpi_openai_key)
export TOKENIZERS_PARALLELISM=false

# ==============================================================================
# Run Command
# ==============================================================================
python -m rlpi.agentdojo.adaptive_agentdojo \
    exp_ident=$EXP_IDENT \
    model=$MODEL \
    suite=$SUITE \
    query_budget=$QUERY_BUDGET \
    user_tasks=$USER_TASKS \
    injection_tasks=$INJECTION_TASKS \
    learner=trl_suffix_joint \
    learner.max_suffix_length=$MAX_SUFFIX_LENGTH \
    learner.temperature=$TEMPERATURE \
    learner.top_p=$TOP_P \
    learner.grpo_num_generations=$GRPO_NUM_GENERATIONS \
    learner.grpo_num_iterations=$GRPO_NUM_ITERATIONS \
    learner.grpo_learning_rate=$GRPO_LEARNING_RATE \
    learner.grpo_per_device_train_batch_size=$GRPO_BATCH_SIZE \
    learner.universal_suffix_mode=$UNIVERSAL_SUFFIX_MODE \
    learner.universal_reward_aggregation=$UNIVERSAL_REWARD_AGGREGATION \
    learner.disable_early_stopping=true \
    --multirun &
