import logging
import os
from typing import Dict

from agentdojo.models import MODEL_NAMES, MODEL_PROVIDERS, ModelsEnum

# FIXED: Use lazy imports to avoid CUDA library error when not using PPO
from rlpi.attack.learners.noop import NoopAttackLearner
from rlpi.attack.learners.random_learner import RandomAttackLearner
from rlpi.attack.utils import get_user_name_from_environment

logger = logging.getLogger(__name__)

# Cache logprob probe results to avoid repeated network calls
_LOGPROB_PROBE_CACHE: Dict[str, bool] = {}


def _get_provider(model_name: str) -> str | None:
    """Resolve provider whether the key is a raw string or ModelsEnum value."""
    provider = MODEL_PROVIDERS.get(model_name)
    if provider:
        return provider

    # Fallback: search enum values
    for enum_key, prov in MODEL_PROVIDERS.items():
        if isinstance(enum_key, ModelsEnum) and enum_key.value == model_name:
            return prov
    return None


def _probe_local_logprob_support(model_name: str) -> bool:
    """Probe HF/vLLM-style local models for logprob support."""
    if model_name in _LOGPROB_PROBE_CACHE:
        return _LOGPROB_PROBE_CACHE[model_name]

    # Assume SecAlign supports logprobs to avoid expensive double-load of shards
    if model_name == "facebook/Meta-SecAlign-70B":
        _LOGPROB_PROBE_CACHE[model_name] = True
        return True

    try:
        import torch  # type: ignore
        from transformers import (  # type: ignore
            AutoModelForCausalLM,
            AutoTokenizer,
        )
    except Exception as exc:  # pragma: no cover - optional dependency
        logger.info(
            f"transformers not available for local logprob probe: {exc}"
        )
        _LOGPROB_PROBE_CACHE[model_name] = False
        return False

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto" if torch.cuda.is_available() else None,
            torch_dtype=torch.float16 if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )
        if tokenizer.pad_token is None and hasattr(tokenizer, "eos_token"):
            tokenizer.pad_token = tokenizer.eos_token

        prompt = "Return the word test."
        inputs = tokenizer(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            _ = model.generate(
                **inputs,
                max_new_tokens=1,
                do_sample=False,
                output_scores=True,
                return_dict_in_generate=True,
            )

        _LOGPROB_PROBE_CACHE[model_name] = True
        return True
    except Exception as exc:  # noqa: BLE001
        logger.info(f"Local logprob probe failed for {model_name}: {exc}")
        _LOGPROB_PROBE_CACHE[model_name] = False
        return False


def _probe_logprob_support(model_name: str) -> bool:
    """Best-effort probe to see if a model supports logprobs (OpenAI or local HF)."""
    if model_name in _LOGPROB_PROBE_CACHE:
        return _LOGPROB_PROBE_CACHE[model_name]

    provider = _get_provider(model_name)
    if provider == "local":
        return _probe_local_logprob_support(model_name)

    try:
        from openai import OpenAI  # type: ignore
    except Exception:
        logger.info("openai package not installed; falling back for logprobs.")
        _LOGPROB_PROBE_CACHE[model_name] = False
        return False

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.info("OPENAI_API_KEY not set; skipping logprob probe.")
        _LOGPROB_PROBE_CACHE[model_name] = False
        return False

    client = OpenAI(api_key=api_key)
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": "Return the word test."}],
            max_tokens=1,
            temperature=0.0,
            logprobs=True,
            top_logprobs=2,
        )
        has_logprobs = bool(
            response.choices and response.choices[0].logprobs is not None
        )
        _LOGPROB_PROBE_CACHE[model_name] = has_logprobs
        return has_logprobs
    except Exception as exc:  # noqa: BLE001
        logger.info(f"Logprob probe failed for {model_name}: {exc}")
        _LOGPROB_PROBE_CACHE[model_name] = False
        return False


def _resolve_feedback_model(victim_model: str, fallback_model: str) -> str:
    """Choose a feedback model that supports logprobs, with fallback."""
    # Special case: SecAlign models are slow for feedback, use GPT-4o-mini instead
    if "secalign" in victim_model.lower():
        logger.info(
            f"Detected SecAlign model {victim_model}; "
            f"using {fallback_model} as feedback model for speed."
        )
        return fallback_model

    # Special case: Qwen3-4B models use GPT-4o-mini as feedback model
    if "qwen3-4b" in victim_model.lower():
        logger.info(
            f"Detected Qwen3-4B model {victim_model}; "
            f"using {fallback_model} as feedback model."
        )
        return fallback_model

    # Special case: Gemma-3 models use GPT-4o-mini as feedback model
    if "gemma-3" in victim_model.lower():
        logger.info(
            f"Detected Gemma-3 model {victim_model}; "
            f"using {fallback_model} as feedback model."
        )
        return fallback_model

    provider = _get_provider(victim_model)

    if provider == "local":
        if _probe_local_logprob_support(victim_model):
            return victim_model
        logger.info(
            f"Model {victim_model} provider={provider} lacks logprob support; "
            f"using fallback {fallback_model} as the feedback model."
        )
        return fallback_model

    if provider != "openai":
        logger.info(
            f"Model {victim_model} provider={provider} lacks logprob support; "
            f"using fallback {fallback_model} as the feedback model."
        )
        return fallback_model

    if _probe_logprob_support(victim_model):
        return victim_model

    logger.info(
        f"Model {victim_model} did not return logprobs; "
        f"falling back to {fallback_model} as the feedback model."
    )
    return fallback_model


def get_attack_learner(
    model: str,
    learner_type: str,
    suite=None,
    **kwargs,
):
    """Factory function to get the appropriate attack learner.

    Args:
        model: Victim model string (e.g., "gpt-4o-2024-08-06")
        learner_type: Type of learner ("ppo", "trl_suffix", "llm_inference", "adaptive_random_suffix", "noop", "random")
        suite: TaskSuite instance for extracting user name
        **kwargs: Additional learner-specific parameters from config

    Returns:
        The appropriate attack learner instance
    """
    # Extract victim model display name
    victim_model_name = MODEL_NAMES.get(model, "the AI assistant")

    # Choose feedback model with fallback if logprobs are unsupported
    # Allow direct override of feedback_model (useful for forcing GPT models)
    if "feedback_model" in kwargs:
        selected_feedback_model = kwargs.pop("feedback_model")
        logger.info(
            f"Using explicitly set feedback_model: {selected_feedback_model}"
        )
    else:
        fallback_feedback_model = kwargs.pop(
            "feedback_fallback_model", "gpt-4o-mini-2024-07-18"
        )
        selected_feedback_model = _resolve_feedback_model(
            victim_model=model, fallback_model=fallback_feedback_model
        )

    # Extract user name from suite if provided
    user_name = "User"
    if suite is not None:
        try:
            environment = suite.load_and_inject_default_environment({})
            user_name = get_user_name_from_environment(suite.name, environment)
        except Exception:
            pass  # Use default

    if learner_type == "ppo":
        from rlpi.attack.learners.ppo.learner import PPOAttackLearner

        return PPOAttackLearner(
            default_agent_name=victim_model_name, user_name=user_name, **kwargs
        )

    elif learner_type == "trl_suffix":
        from rlpi.attack.learners.trl_suffix import TRLSuffixLearner

        return TRLSuffixLearner(
            victim_model_name=victim_model_name,
            user_name=user_name,
            feedback_model=selected_feedback_model,  # Fallback if victim lacks logprobs
            **kwargs,  # Includes attack_model_name="Qwen/..." from config (attack policy model)
        )

    elif learner_type == "trl_suffix_joint":
        from rlpi.attack.learners.trl_suffix.joint_learner import (
            TRLSuffixJointLearner,
        )

        return TRLSuffixJointLearner(
            victim_model_name=victim_model_name,
            user_name=user_name,
            feedback_model=selected_feedback_model,  # Fallback if victim lacks logprobs
            **kwargs,  # Includes attack_model_name="Qwen/..." from config (attack policy model)
        )

    elif learner_type == "adaptive_random_suffix":
        from rlpi.attack.learners.adaptive_random_suffix import (
            AdaptiveRandomSuffixLearner,
        )

        return AdaptiveRandomSuffixLearner(
            victim_model_name=victim_model_name,
            user_name=user_name,
            feedback_model=selected_feedback_model,  # Fallback if victim lacks logprobs
            **kwargs,  # Includes attack_model_name="Qwen/..." from config (attack policy model)
        )

    elif learner_type == "llm_inference":
        from rlpi.attack.learners.llm_inference import LLMInferenceLearner

        return LLMInferenceLearner(
            victim_model_name=victim_model_name,
            user_name=user_name,
            feedback_model=selected_feedback_model,  # Fallback if victim lacks logprobs
            **kwargs,  # Includes attack_model_name="Qwen/..." from config (attack policy model)
        )

    elif learner_type == "noop":
        return NoopAttackLearner(**kwargs)
    elif learner_type == "random":
        return RandomAttackLearner(**kwargs)
    else:
        raise ValueError(f"Unknown learner type: {learner_type}")
