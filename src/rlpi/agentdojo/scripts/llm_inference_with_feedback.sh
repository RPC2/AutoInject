#!/bin/bash
# LLM Inference with GPT Feedback Script
# This script runs the LLM inference-based suffix attack learner with GPT feedback enabled
# Uses both GPT feedback and empirical utility/security scores for reward computation

# ==============================================================================
# Configuration Variables (edit these to customize your run)
# ==============================================================================

MODEL="gpt-5-nano"
EXP_IDENT="${MODEL}_llm_inference_with_feedback"

SUITE="banking"
QUERY_BUDGET=260

USER_TASKS="[user_task_0],[user_task_1],[user_task_2],[user_task_3],[user_task_4],[user_task_5],[user_task_6],[user_task_7],[user_task_8],[user_task_9],[user_task_10],[user_task_11],[user_task_12],[user_task_13],[user_task_14],[user_task_15]"
INJECTION_TASKS="[injection_task_0],[injection_task_1],[injection_task_2],[injection_task_3],[injection_task_4],[injection_task_5],[injection_task_6],[injection_task_7],[injection_task_8]"

# Suffix generation parameters
MAX_SUFFIX_LENGTH=30  # Token limit for generated suffixes
TEMPERATURE=0.9       # Sampling temperature (higher = more diverse)
TOP_P=0.95           # Nucleus sampling threshold

# Experience-based prompt parameters
USE_EXPERIENCE_EXAMPLES=true  # Use experience history in prompt (vs. hardcoded examples)
NUM_EXPERIENCE_EXAMPLES=3     # Number of top examples to include from experience
INCLUDE_GPT_REASONING=true    # Include GPT reasoning for best suffix in prompt

# GPT feedback parameters
GPT_ENABLED=true  # Enable GPT-based suffix quality feedback

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
    learner=llm_inference \
    learner.max_suffix_length=$MAX_SUFFIX_LENGTH \
    learner.temperature=$TEMPERATURE \
    learner.top_p=$TOP_P \
    learner.use_experience_examples=$USE_EXPERIENCE_EXAMPLES \
    learner.num_experience_examples=$NUM_EXPERIENCE_EXAMPLES \
    learner.include_gpt_reasoning=$INCLUDE_GPT_REASONING \
    learner.gpt_enabled=$GPT_ENABLED \
    +hydra.launcher.additional_parameters.gpus=titan_rtx:1 \
    --multirun &
