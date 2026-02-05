# AutoInject

This is the repository for paper Learning to Inject: Automated prompt Injection via Reinforcement Learning.

**Note**: The current version is not fully tested and robust yet; Some errors might occur when one tries to install this current version.

## Installation and running experiments from the paper
Installing the environment:
```
pip3 install torch
# Note: You might need to refer to the pytorch official installation guide to match your own cuda version and setup.
pip install -e .
cd agentdojo && pip install -e .
```

If you want to test/use any OpenAI models, save your OpenAI API key under `~/.rlpi_openai_key`. We suggest to first try querying an OpenAI model using this key to test that it is working.

If you want to test/use any OpenRouter models (e.g., Claude models), save your OpenRouter API key under `~/.rlpi_openrouter_key`.

Most experiments from the paper can be run from files under `src/rlpi/agentdojo/scripts`.