#!/bin/bash
# Evolutionary-search attack baseline (LLM-mutation MAP-Elites, following
# Nasr et al. 2025). Uses the same mutator model (Qwen2-1.5B) and query budget
# (260) as AutoInject for a fair comparison.
#
# Requires an OpenAI API key in ~/.rlpi_openai_key (and, for OpenRouter models
# such as Claude, an OpenRouter key in ~/.rlpi_openrouter_key).

export OPENAI_API_KEY=$(cat ~/.rlpi_openai_key)
export TOKENIZERS_PARALLELISM=false

MODEL=${1:-gpt-4.1-nano}
SUITE=${2:-banking}
QUERY_BUDGET=${3:-260}

echo "Evolutionary search baseline | model=${MODEL} suite=${SUITE} budget=${QUERY_BUDGET}"

python -m rlpi.agentdojo.adaptive_agentdojo \
    exp_ident=${MODEL}_evolutionary_search \
    model=${MODEL} \
    suite=${SUITE} \
    query_budget=${QUERY_BUDGET} \
    learner=evolutionary_search
    # To sweep many (user_task, injection_task) pairs on a SLURM cluster,
    # uncomment the two lines below and set a GPU resource for your scheduler:
    # +hydra.launcher.additional_parameters.gpus=<your_gpu>:1 \
    # --multirun &
