#!/bin/bash
export OPENAI_API_KEY=$(cat ~/.rlpi_openai_key)
model_name=gpt-4o-mini-2024-07-18

python -m rlpi.agentdojo.adaptive_agentdojo \
    exp_ident=${model_name}_baselines_all_tasks \
    model=${model_name} \
    suite=slack \
    query_budget=10 \
    injection_tasks=[injection_task_0] \
    user_tasks=[user_task_0] \
    attack=direct \
    learner=noop
    # hydra.launcher.additional_parameters={}
    # --multirun &
