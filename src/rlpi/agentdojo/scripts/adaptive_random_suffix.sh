#!/bin/bash

# Set environment variables
export OPENAI_API_KEY=$(cat ~/.rlpi_openai_key)
export OPENROUTER_API_KEY=$(cat ~/.rlpi_openrouter_key)
export TOKENIZERS_PARALLELISM=false
model_name=gpt-4o-mini-2024-07-18

declare -A suite_user_tasks
declare -A suite_injection_tasks

# Banking suite
suite_user_tasks[banking]="[user_task_0],[user_task_1],[user_task_2],[user_task_6],[user_task_13]"
suite_injection_tasks[banking]="[injection_task_4],[injection_task_6],[injection_task_7],[injection_task_8]"

# Run adaptive random suffix baseline across all suites
# Hydra's --multirun will handle all combinations of user_tasks and injection_tasks
# for suite in banking slack travel workspace; do
for suite in banking; do
    echo "=========================================="
    echo "Starting suite: ${suite}"
    echo "=========================================="

    python -m rlpi.agentdojo.adaptive_agentdojo \
        exp_ident=${model_name}_adaptive_random_suffix_subsuite \
        model=${model_name} \
        suite=${suite} \
        query_budget=260 \
        user_tasks=${suite_user_tasks[$suite]} \
        injection_tasks=${suite_injection_tasks[$suite]} \
        learner=adaptive_random_suffix \
        learner.use_self_transfer=true \
        learner.max_suffix_length=30 \
        learner.mutation_rate=0.3 \
        --multirun &

    echo "Completed suite: ${suite}"
    echo ""
done

echo "=========================================="
echo "All experiments completed!"
echo "=========================================="
