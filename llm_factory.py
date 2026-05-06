import importlib
import logging
from typing import Dict

from llm_registry import LLMRegistry
from utils.llm import BaseLLM

"""
Factory + Singleton Cache Pattern.

Responsibilities:
1. Create LLM instances.
2. Ensure only ONE instance per provider.
3. Decouple provider implementation from application code.

Agents never create LLMs.
Graph never creates LLMs.
Only this file controls lifecycle.
"""

logger = logging.getLogger(__name__)

_LLM_CACHE: Dict[str, BaseLLM] = {}


# _PROVIDERS = {"anthropic": AnthropicClient, "groq": GroqClient, "openai": OpenAIClient}
module_map = {
    "mlx": "utils.mlx_utils",
    "mock": "utils.mock_llm",
    "groq": "utils.groq_utils",
    "openai": "utils.openai_utils",
    "ollama": "utils.ollama_utils",
    "anthropic": "utils.anthropic_utils",
}


def create_llm(provider: str, **kwargs) -> BaseLLM:
    """
    Returns a cached LLM instance if already created. Otherwise, creates, caches, and returns it.
    This gives singleton-like behavior per provider.
    """
    # key = (provider, model)
    if provider not in module_map:
        raise ValueError(f"Unknown provider: {provider}")

    if provider not in _LLM_CACHE:
        module = importlib.import_module(module_map[provider])
        logger.info(f"module {module.__name__} imported. Creating LLM instance for {provider}")

        provider_cls = LLMRegistry.get(provider)
        _LLM_CACHE[provider] = provider_cls(**kwargs)
        logger.info(f"Created LLM instance for {provider}")

    return _LLM_CACHE[provider]
