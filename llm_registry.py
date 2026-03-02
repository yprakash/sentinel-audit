import logging

logger = logging.getLogger(__name__)


class LLMRegistry:
    _registry = {}

    @classmethod
    def register(cls, name: str, provider_cls):
        cls._registry[name] = provider_cls
        logger.debug(f"Registered LLM provider '{name}' for {provider_cls.__name__}")

    @classmethod
    def get(cls, name: str):
        if name not in cls._registry:
            raise ValueError(f"Provider '{name}' not registered.")
        return cls._registry[name]
