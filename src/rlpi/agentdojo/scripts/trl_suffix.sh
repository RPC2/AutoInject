#!/bin/bash
# TRL Suffix Attack Training Script
# This script runs the TRL-based suffix attack learner with Qwen2-1.5B and LoRA

# ==============================================================================
# Configuration Variables (edit these to customize your run)
# ==============================================================================

# MODEL="gpt-5-nano"
MODEL="google/gemma-3-4b-it"
EXP_IDENT="${MODEL}_trl"

SUITE="banking"
QUERY_BUDGET=260

# USER_TASKS="[user_task_0],[user_task_1],[user_task_2],[user_task_3],[user_task_4],[user_task_5],[user_task_6],[user_task_7],[user_task_8],[user_task_9],[user_task_10],[user_task_11],[user_task_12],[user_task_13],[user_task_14],[user_task_15]"
# INJECTION_TASKS="[injection_task_0],[injection_task_1],[injection_task_2],[injection_task_3],[injection_task_4],[injection_task_5],[injection_task_6],[injection_task_7],[injection_task_8]"
USER_TASKS="[user_task_0]"
INJECTION_TASKS="[injection_task_0]"

# Suffix generation parameters
MAX_SUFFIX_LENGTH=30  # Token limit for generated suffixes
TEMPERATURE=0.9       # Sampling temperature (higher = more diverse)
TOP_P=0.95           # Nucleus sampling threshold

# GRPO training parameters
GRPO_NUM_GENERATIONS=8   # Suffixes generated per prompt per iteration
GRPO_NUM_ITERATIONS=5   # Training iterations per session (reduced from 10 to fit query budget)
GRPO_LEARNING_RATE=5e-6  # Policy learning rate (reasonable for GRPO)
GRPO_BATCH_SIZE=1        # Per-device batch size
GRPO_WARMUP_STEPS=20     # Learning rate warmup steps
GRPO_GRADIENT_ACCUMULATION_STEPS=4  # Gradient accumulation steps (increased for stability)

# Stability parameters (critical for preventing gradient explosion)
GRPO_MAX_GRAD_NORM=0.1   # Gradient clipping - lower = more aggressive clipping
GRPO_BETA=0.1           # KL penalty - higher = stronger constraint on policy updates
GRPO_ADAM_EPSILON=5e-6   # Adam epsilon - higher = more stable optimizer
GRPO_ADAM_BETA2=0.98      # Adam beta2 - higher = more conservative optimizer

# ==============================================================================
# API Keys and Environment
# ==============================================================================
export OPENAI_API_KEY=$(cat ~/.rlpi_openai_key)
export TOGETHER_API_KEY=$(cat ~/.rlpi_togetherai_key)
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
    learner=trl_suffix \
    learner.max_suffix_length=$MAX_SUFFIX_LENGTH \
    learner.temperature=$TEMPERATURE \
    learner.top_p=$TOP_P \
    learner.grpo_num_generations=$GRPO_NUM_GENERATIONS \
    learner.grpo_num_iterations=$GRPO_NUM_ITERATIONS \
    learner.grpo_learning_rate=$GRPO_LEARNING_RATE \
    learner.gpt_enabled=true
    # +hydra.launcher.additional_parameters.gpus=titan_rtx:1 \
    # --multirun &

