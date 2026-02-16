import functools
import logging
import time
from typing import Callable, Any, Awaitable

from prometheus_client import Histogram, Counter

from metrics import start_metrics_server

logger = logging.getLogger(__name__)
_missing_usage_logged = set()

LLM_REQUESTS_TOTAL = Counter(
    "llm_requests_total",
    "Total LLM requests",
    ["provider", "model", "status"],
)

LLM_LATENCY = Histogram(
    "llm_request_latency_seconds",
    "LLM request latency in seconds",
    ["provider", "model"],
    buckets=(0.1, 0.3, 0.5, 1, 2, 5, 10, 20, 30, 60),
)

LLM_INPUT_TOKENS = Counter(
    "llm_input_tokens_total",
    "Total input tokens",
    ["provider", "model"],
)

LLM_OUTPUT_TOKENS = Counter(
    "llm_output_tokens_total",
    "Total output tokens",
    ["provider", "model"],
)


def extract_usage(response):
    # Some providers return token usage under different shapes:
    # response.usage
    # response.metadata.usage
    # response.token_usage
    # dictionary: response["usage"]
    if hasattr(response, "usage"):
        return response.usage
    if isinstance(response, dict) and "usage" in response:
        return response["usage"]
    return None


def init(port: int, app_name: str, shutdown_event, interval: int):
    start_metrics_server(port, app_name, shutdown_event, interval)


def llm_metrics(provider: str) -> Callable:
    """
    Decorator for capturing LLM metrics.
    Usage:
        @llm_metrics("openai")
        async def generate(...):
            ...
    """

    def decorator(func: Callable[..., Awaitable[Any]]) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            model = kwargs.get("model", "unknown")
            start = time.perf_counter()
            status = "success"
            response = None

            try:
                response = await func(*args, **kwargs)
                return response
            except Exception:
                status = "error"
                raise

            finally:
                duration = time.perf_counter() - start
                LLM_REQUESTS_TOTAL.labels(
                    provider=provider,
                    model=model,
                    status=status,
                ).inc()

                LLM_LATENCY.labels(
                    provider=provider,
                    model=model,
                ).observe(duration)

                if response and hasattr(response, "usage"):
                    usage = response.usage
                    input_tokens = getattr(usage, "prompt_tokens", 0)
                    output_tokens = getattr(usage, "completion_tokens", 0)

                    LLM_INPUT_TOKENS.labels(
                        provider=provider,
                        model=model,
                    ).inc(input_tokens)

                    LLM_OUTPUT_TOKENS.labels(
                        provider=provider,
                        model=model,
                    ).inc(output_tokens)

                elif response:
                    key = (provider, model)
                    if key not in _missing_usage_logged:
                        logger.warning(
                            "LLM provider '%s' model '%s' response has no 'usage' attribute. "
                            "Available attributes: %s",
                            provider,
                            model,
                            [attr for attr in dir(response) if not attr.startswith("_")]
                        )
                        _missing_usage_logged.add(key)

        return wrapper

    return decorator
