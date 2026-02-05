#!/bin/bash

model_name="facebook/Meta-SecAlign-70B"
suite=slack
user_tasks="[user_task_0]"
injection_tasks="[injection_task_1]"

export NCCL_NET=Socket
export NCCL_SHM_DISABLE=0
export NCCL_P2P_LEVEL=NVL

echo "[run_meta_secalign_local] Running adaptive_agentdojo..."
python -m rlpi.agentdojo.adaptive_agentdojo \
  exp_ident=Meta-SecAlign_local_attack \
  model=${model_name} \
  suite=${suite} \
  learner=trl_suffix \
  query_budget=100 \
  user_tasks=${user_tasks} \
  injection_tasks=${injection_tasks} \
  learner.grpo_num_generations=2 \
  learner.grpo_num_iterations=3

  # learner=noop \