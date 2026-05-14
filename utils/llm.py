import asyncio
import json
import logging
import os
import time
import traceback
from abc import ABC, abstractmethod
from asyncio import Semaphore
from typing import Any, Dict

from prometheus_client import Histogram, Counter

from utils.metrics import start_metrics_server
from utils.shutdown_manager import shutdown_manager

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

LLM_THINKING_TOKENS = Counter(
    "llm_tokens_thinking_total",
    "Total thinking tokens",
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


async def shutdown_llm_client(client=None, active_tasks: set[asyncio.Task] = None):
    """
    Gracefully shutdown:
    1. Wait for all in-flight LLM calls
    2. Close client like AsyncOpenAI/AsyncGroq
    """
    start = time.perf_counter()
    client_name = client.__class__.__name__ if client else "UNDEFINED_CLIENT"
    if active_tasks:
        logger.info(f"Waiting for completion of {len(active_tasks)} {client_name} active tasks")
        await asyncio.gather(*active_tasks, return_exceptions=True)

    if client:
        logger.info(f"Closing {client_name} client...")
        try:
            await client.aclose()
        except:
            logger.info(f"{client_name}.aclose() failed...")
            try:
                await client.close()
            except:
                logger.info(f"{client_name}.close() failed...")

    duration = time.perf_counter() - start
    logger.info(f"Graceful shut down of {client_name} Completed in {duration} seconds")


def init(port: int, app_name: str, shutdown_event, interval: int):
    """
    Initialize metrics server once per service.
    Should be called during application startup.
    """
    start_metrics_server(port, app_name, shutdown_event, interval)


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
            model: str = None,
            agent_role: str = None,
            max_concurrent: int = 5,
    ) -> None:
        if not provider:
            raise ValueError("Provider name must be defined.")
        self.provider = provider.strip()
        if model is None:
            key = self.provider.upper() + "_MODEL_NAME"
            model = os.environ.get(key, None)

        self.model = model
        self.agent_role = agent_role if agent_role else "unknown"
        self.semaphore = Semaphore(max_concurrent)
        self.active_tasks: set[asyncio.Task] = set()
        # It is must to register after instantiation for graceful shutdown
        shutdown_manager.register(self)

    @abstractmethod
    def get_ai_message_from_response(self, response):
        raise NotImplementedError

    @abstractmethod
    async def shutdown(self) -> None:
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

    async def get_available_models(self):
        try:
            models = await self._client.models.list()
            return models.data
        except Exception as e:
            print("Connection failed:", type(e).__name__, str(e))

    def write_llm_output(self, output: dict) -> None:
        def _save_json(file_path: str, data: Any) -> None:
            with open(file_path, "w") as f:
                json.dump(data, f, indent=2)
            logger.info("Written %s LLM output to %s", self.provider, file_path)

        file_path = f"mocked_outputs/{self.agent_role}.json"
        _save_json(file_path, output)
        file_path = f"mocked_outputs/{self.agent_role}_{int(time.time())}.json"
        _save_json(file_path, output)

    async def generate(self, model: str = None, **kwargs) -> Any:
        """
        Public async entrypoint.

        Responsibilities:
        - Capture success/error status
        - Measure latency and Record Prometheus metrics
        - Delegate actual logic to `_generate_impl`
        """

        start = time.perf_counter()
        status = "SUCCESS"
        response = None
        if model is None or not model:
            model = self.model

        current_task = asyncio.current_task()
        self.active_tasks.add(current_task)

        try:
            async with self.semaphore:
                response = await self._generate_impl(model=model, **kwargs)
                return response
        except Exception as e:
            status = "ERROR"
            error_msg = traceback.format_exc()
            logger.exception(error_msg)
            raise e

        finally:
            self.active_tasks.discard(current_task)
            duration = time.perf_counter() - start
            self._record_metrics(
                model=model,
                agent_role=self.agent_role,
                duration=duration,
                status=status,
                response=response,
            )

    def extract_usage(self, response: Any) -> Dict[str, int]:
        usage = response.get("usage", None) if response else None
        if not usage:
            logger.warning("%s LLM returned empty usage response: %s", self.provider, response)
            return {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            }

        input_key = "prompt_tokens" if "prompt_tokens" in usage else "input_tokens"
        output_key = "completion_tokens" if "completion_tokens" in usage else "output_tokens"
        prompt_tokens = usage.get(input_key, 0)
        completion_tokens = usage.get(output_key, 0)
        thinking_tokens = usage.get("thinking_tokens", 0)

        total_tokens = usage.get("total_tokens", 0)
        if not total_tokens:
            total_tokens = prompt_tokens + completion_tokens + thinking_tokens

        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "thinking_tokens": thinking_tokens,
            "total_tokens": total_tokens,
        }

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
        log_line = f"LLM output status={status} for provider={self.provider}, model={model}, agent_role={agent_role}, duration={duration:.3f}s"

        if usage:
            LLM_INPUT_TOKENS.labels(
                provider=self.provider,
                model=model,
                agent_role=agent_role,
            ).inc(usage["prompt_tokens"])

            LLM_OUTPUT_TOKENS.labels(
                provider=self.provider,
                model=model,
                agent_role=agent_role,
            ).inc(usage["completion_tokens"])

            LLM_THINKING_TOKENS.labels(
                provider=self.provider,
                model=model,
                agent_role=agent_role,
            ).inc(usage["thinking_tokens"])
            log_line += f", input_tokens={usage["prompt_tokens"]}, total_tokens={usage["total_tokens"]}"

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

        if status == "SUCCESS":
            logger.info(log_line)
        else:
            logger.error(log_line)
