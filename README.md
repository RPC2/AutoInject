# AutoInject

Official code for the paper **"Learning to Inject: Automated Prompt Injection via Reinforcement Learning"** (Xin Chen, Jie Zhang, Florian Tramèr).

AutoInject is a **black-box reinforcement-learning framework** that learns adversarial
suffixes for prompt injection against LLM agents. Its core idea is a *learned,
comparison-based reward*: each candidate suffix is scored against the best suffix seen
so far, turning the binary attack-success signal into a dense reward that GRPO can
optimize. It can also incorporate a utility objective when task-completion feedback is
available. Attacks are evaluated on [AgentDojo](https://github.com/ethz-spylab/agentdojo).

> **Abstract.** Prompt injection is a critical vulnerability in LLM agents, yet the
> strongest methods still rely on human red-teamers and hand-crafted prompts. Adapting
> automated jailbreak optimizers does not close this gap: jailbreaks shape models toward
> generic compliance, while prompt injection requires emitting specific tool calls with
> correct parameters. The success signal is binary, and randomly sampled suffixes almost
> never trigger it—so standard optimizers have no gradient to follow. We present
> AutoInject, a black-box reinforcement learning (RL) framework that learns adversarial
> suffixes for prompt injection. A learned comparison-based reward scores each candidate
> against the best suffix seen so far, turning the binary signal into a dense reward
> suitable for RL optimization. The framework supports both online query-based attacks
> and offline-trained transferable suffixes that need no utility access at deployment,
> and incorporates a utility objective when task-completion feedback is available. On
> AgentDojo, AutoInject outperforms template attacks, GCG, TAP, and adaptive attack
> across production models, with statistically significant improvements under McNemar's
> test (p < 0.05). Suffixes learned by AutoInject also break Meta-SecAlign-70B, a model
> fine-tuned specifically to resist prompt injection, where template attacks fail
> outright.

---

## Repository layout

```
AutoInject/
├── agentdojo/                      # Vendored AgentDojo benchmark (task suites + agent pipeline)
├── src/rlpi/
│   ├── agentdojo/
│   │   ├── adaptive_agentdojo.py   # Main entry point: online attacks / training
│   │   ├── config/                 # Hydra configs (learners, suites)
│   │   └── scripts/                # Example run scripts for every experiment
│   └── attack/
│       ├── learner_factory.py      # Dispatches learner type -> implementation
│       ├── templates/              # Attack prompt templates / task modifiers
│       └── learners/
│           ├── trl_suffix/         # AutoInject (GRPO suffix learner) — the paper's method
│           ├── adaptive_random_suffix/  # Random adaptive-attack baseline
│           ├── llm_inference/      # LLM-inference baseline
│           ├── evolutionary_search/# Evolutionary-search baseline
│           └── noop.py             # Template attacks (no learning)
└── setup.py
```

## Installation

```bash
# 1. Install PyTorch matching your CUDA setup (see https://pytorch.org).
pip install torch

# 2. Install AutoInject and the vendored AgentDojo benchmark.
pip install -e .
cd agentdojo && pip install -e . && cd ..
```

Python 3.10+ is required. A CUDA-capable GPU is needed to train the GRPO suffix policy
(`trl_suffix`); the template and API-only baselines run on CPU.

## API keys

Models are accessed through provider APIs. Save your keys to these files (each script
reads them automatically):

| File | Used for |
|------|----------|
| `~/.rlpi_openai_key`     | OpenAI models (GPT-4.1-nano, GPT-5-nano, GPT-4o-mini feedback) |
| `~/.rlpi_openrouter_key` | OpenRouter models (e.g. Claude, some Gemini routes) |

We recommend issuing a test query to your chosen provider first to confirm the key works.

## Quickstart

Run AutoInject on a single (user task, injection task) pair in the banking suite:

```bash
export OPENAI_API_KEY=$(cat ~/.rlpi_openai_key)
export TOKENIZERS_PARALLELISM=false

python -m rlpi.agentdojo.adaptive_agentdojo \
    exp_ident=quickstart \
    model=gpt-4.1-nano \
    suite=banking \
    query_budget=260 \
    user_tasks=[user_task_0] \
    injection_tasks=[injection_task_0] \
    learner=trl_suffix
```

Configuration is managed by [Hydra](https://hydra.cc); any field can be overridden on the
command line (e.g. `suite=travel`, `learner=noop`, `query_budget=100`). Results are
written under `outputs/` (see [Output structure](#output-structure)).

## Reproducing the paper

Example scripts live in `src/rlpi/agentdojo/scripts/`. Edit the variables at the top of
each script (model, suite, task lists) to match the configuration you want.

### Attacks and baselines

These runs produce the core attack-vs-baseline comparisons reported in the paper's main
results (attack success rate and utility across production models and task suites). Run
each method across the four suites (`suite=banking|slack|travel|workspace`) and the models
you want to evaluate.

| Method | Script | Learner |
|--------|--------|---------|
| **AutoInject (ours)** | `trl_suffix.sh` | `trl_suffix` |
| Template attacks (Direct, Ignore Previous, Important Instructions, InjecAgent, System Message, Tool Knowledge) | `attack_baselines.sh` | `noop` (set `attack=`) |
| Random adaptive attack | `adaptive_random_suffix.sh` | `adaptive_random_suffix` |
| LLM-inference attack | `llm_inference_with_feedback.sh`, `llm_inference_no_feedback.sh` | `llm_inference` |
| Evolutionary search | `evolutionary_search.sh` | `evolutionary_search` |
| Benign utility (no attack) | `user_task_benchmark.sh` | — |

Template baselines select the strategy via `attack=` (e.g.
`attack=direct|ignore_previous|important_instructions|injecagent|system_message|tool_knowledge`).

### Ablations (§4.4)

Ablations are command-line overrides on the `trl_suffix` learner:

```bash
# Effect of feedback model: disable the learned comparison-based reward
python -m rlpi.agentdojo.adaptive_agentdojo learner=trl_suffix learner.gpt_enabled=false ...

# Effect of LLM policy: swap the attack policy model
python -m rlpi.agentdojo.adaptive_agentdojo learner=trl_suffix learner.attack_model_name=<hf_model> ...

# Effect of GRPO training: compare against the random adaptive baseline
python -m rlpi.agentdojo.adaptive_agentdojo learner=adaptive_random_suffix ...
```

## Supported learners

| `learner=` | Description |
|-----------|-------------|
| `trl_suffix`        | **AutoInject** — GRPO suffix policy with comparison-based dense reward |
| `adaptive_random_suffix` | Random adaptive attack (feedback-guided mutation) |
| `llm_inference`     | LLM-inference attack baseline |
| `evolutionary_search` | Evolutionary search with LLM mutation (MAP-Elites) |
| `noop`              | Template attacks (no learning) |

## Output structure

Hydra writes each run to a timestamped directory:

```
outputs/{learner.type}/{suite}/{date}/{exp_ident}-{time}/   # single runs
multirun/{learner.type}/{suite}/{date}/{exp_ident}-{time}/  # --multirun sweeps
```

## Running on a SLURM cluster

The configs ship with Hydra's `submitit_slurm` launcher. To sweep many
(user task, injection task) pairs in parallel, append `--multirun` and provide a GPU
resource spec for your scheduler:

```bash
python -m rlpi.agentdojo.adaptive_agentdojo \
    ... \
    +hydra.launcher.additional_parameters.gpus=<your_gpu>:1 \
    --multirun
```

Without SLURM, omit `--multirun` to run a single configuration locally.

## Citation

```bibtex
@inproceedings{chen2026autoinject,
  title     = {Learning to Inject: Automated Prompt Injection via Reinforcement Learning},
  author    = {Chen, Xin and Zhang, Jie and Tram{\`e}r, Florian},
  year      = {2026}
  % TODO: add venue and arXiv/OpenReview URL once public.
}
```

## License

Released under the [MIT License](LICENSE). The vendored `agentdojo/` directory retains its
own upstream license.
