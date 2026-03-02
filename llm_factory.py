import logging
from typing import Dict

from utils.llm import BaseLLM
from .llm_registry import LLMRegistry

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


def create_llm(provider: str, **kwargs) -> BaseLLM:
    """
    Returns a cached LLM instance if already created. Otherwise creates, caches, and returns it.
    This gives singleton-like behavior per provider.
    """
    # key = (provider, model)
    key = provider

    if key not in _LLM_CACHE:
        provider_cls = LLMRegistry.get(key)
        _LLM_CACHE[key] = provider_cls(**kwargs)
        logger.info(f"Created LLM instance for {provider}")

    return _LLM_CACHE[key]
