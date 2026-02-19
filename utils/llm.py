import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Optional

from prometheus_client import Histogram, Counter

from metrics import start_metrics_server

logger = logging.getLogger(__name__)

# Track which provider/model combinations have already logged missing usage
_missing_usage_logged = set()

# ---- Prometheus Metrics ----

LLM_REQUESTS_TOTAL = Counter(
    "llm_requests_total",
    "Total LLM requests",
    ["provider", "model", "agent_role", "status"],
)

LLM_LATENCY = Histogram(
    "llm_request_duration_seconds",
    "LLM request latency in seconds",
    ["provider", "model", "agent_role"],
    buckets=(0.1, 0.3, 0.5, 1, 2, 5, 10, 20, 30, 60),
)

LLM_INPUT_TOKENS = Counter(
    "llm_tokens_prompt_total",
    "Total input tokens",
    ["provider", "model", "agent_role"],
)

LLM_OUTPUT_TOKENS = Counter(
    "llm_tokens_completion_total",
    "Total output tokens",
    ["provider", "model", "agent_role"],
)

# Metrics for future use
LLM_TOKENS_TOTAL = Counter(
    "llm_tokens_total",
    "Total LLM tokens",
    ["provider", "model", "agent_role"],
)
LLM_TTFT_SECONDS = Histogram(  # Crucial for "perceived speed" in UI
    "llm_ttft_seconds",
    "Time to First Token",
    ["provider", "model", "agent_role"],
    buckets=(0.1, 0.3, 0.5, 1, 2, 5, 10, 20, 30, 60),
)
LLM_ITL_SECONDS = Histogram(  # Measures "smoothness" of streaming
    "llm_itl_seconds",
    "Inter-Token Latency",
    ["provider", "model", "agent_role"],
    buckets=(0.1, 0.3, 0.5, 1, 2, 5, 10, 20, 30, 60),
)

# Economic Metrics (Gauges)
# llm_request_cost_usd: Calculate (tokens * price_per_token) on the fly.
# llm_budget_remaining_usd
# Agent-Specific Context Labels: model_name, agent_role (strategist vs. adversary), contract_name, and status_code.


def extract_usage(response: Any) -> Optional[dict]:
    """
    Normalize token usage extraction across providers.
    Note: This is just for reference. Should be deleted in the future.

    Supports:
    - response.usage
    - response["usage"]
    - Other future shapes can be added here centrally.
    """
    if response is None:
        return None

    # Object-style usage
    if hasattr(response, "usage"):
        usage = response.usage
        return {
            "input": getattr(usage, "prompt_tokens", 0),
            "output": getattr(usage, "completion_tokens", 0),
        }

    # Dict-style usage
    if isinstance(response, dict) and "usage" in response:
        usage = response["usage"]
        return {
            "input": usage.get("prompt_tokens", 0),
            "output": usage.get("completion_tokens", 0),
        }

    return None


async def init(port: int, app_name: str, shutdown_event, interval: int):
    """
    Initialize metrics server once per service.
    Should be called during application startup.
    """
    await start_metrics_server(port, app_name, shutdown_event, interval)


class BaseLLM(ABC):
    """
    Base async LLM abstraction.

    - Enforces provider-specific implementation via `_generate_impl`
    - Centralizes metrics recording
    - Keeps provider files free from observability concerns
    """

    def __init__(
            self,
            provider: str,
            model: str,
            agent_role: str = None,
    ) -> None:
        if not provider:
            raise ValueError("Provider name must be defined.")
        self.provider = provider.strip()
        self.model = model
        self.agent_role = agent_role if agent_role else "unknown"

    @abstractmethod
    def extract_usage(self, response):
        pass

    @abstractmethod
    async def _generate_impl(self, model: str, **kwargs) -> Any:
        """
        Provider-specific async implementation.

        Must:
        - Perform actual API call
        - Return provider response object
        """
        raise NotImplementedError

    async def generate(self, model: str, **kwargs) -> Any:
        """
        Public async entrypoint.

        Responsibilities:
        - Capture success/error status
        - Measure latency and Record Prometheus metrics
        - Delegate actual logic to `_generate_impl`
        """

        start = time.perf_counter()
        status = "success"
        response = None
        if model is None or not model:
            model = self.model

        try:
            response = await self._generate_impl(model=model, **kwargs)
            return response
        except Exception:
            status = "error"
            raise

        finally:
            duration = time.perf_counter() - start
            self._record_metrics(
                model=model,
                agent_role=self.agent_role,
                duration=duration,
                status=status,
                response=response,
            )

    def _record_metrics(
            self,
            model: str,
            agent_role: str,
            duration: float,
            status: str,
            response: Any,
    ) -> None:
        """
        Record all Prometheus metrics for this request.

        This method is synchronous and safe in async contexts
        because Prometheus client operations are thread-safe.
        """

        # Total request counter
        LLM_REQUESTS_TOTAL.labels(
            provider=self.provider,
            model=model,
            agent_role=agent_role,
            status=status,
        ).inc()

        # Latency histogram
        LLM_LATENCY.labels(
            provider=self.provider,
            model=model,
            agent_role=agent_role,
        ).observe(duration)

        # Token usage extraction
        usage = self.extract_usage(response)

        if usage:
            LLM_INPUT_TOKENS.labels(
                provider=self.provider,
                model=model,
                agent_role=agent_role,
            ).inc(usage["input"])

            LLM_OUTPUT_TOKENS.labels(
                provider=self.provider,
                model=model,
                agent_role=agent_role,
            ).inc(usage["output"])

        elif response:
            # Log only once per provider/model to avoid log flooding
            key = (self.provider, model)

            if key not in _missing_usage_logged:
                logger.warning(
                    "LLM provider '%s' model '%s' has no detectable token usage field.",
                    self.provider,
                    model,
                )
                _missing_usage_logged.add(key)
