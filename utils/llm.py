import asyncio
import json
import logging
import os
import time
import traceback
from abc import ABC, abstractmethod
from asyncio import Semaphore
from typing import Any, Optional

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


async def shutdown_llm_client(client = None, active_tasks: set[asyncio.Task] = None):
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
        await client.aclose()

    duration = time.perf_counter() - start
    logger.info(f"Graceful shut down of {client_name} Completed in {duration} seconds")


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
            model = os.environ.get(key, "")
        self.model = model
        self.agent_role = agent_role if agent_role else "unknown"
        self.semaphore = Semaphore(max_concurrent)
        self.active_tasks: set[asyncio.Task] = set()
        # It is must to register after instantiation for graceful shutdown
        shutdown_manager.register(self)

    @abstractmethod
    async def shutdown(self) -> None:
        pass

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

    def write_llm_output(self, output):
        file_path = f"mocked_outputs/{self.agent_role}.txt"
        with open(file_path, 'a') as file:
            file.write(output + os.linesep)
        log_line = f"Written LLM output to {file_path}"

        output_dict = json.loads(output)
        if isinstance(output_dict, dict):
            file_path = f"mocked_outputs/{self.agent_role}.json"
            with open(file_path, 'w') as file:
                json.dump(output_dict, file, indent=4)
            log_line += f" as well as {file_path}"

        logger.info(log_line)

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
            input_key = "prompt_tokens" if "prompt_tokens" in usage else "input_tokens"
            output_key = "completion_tokens" if "completion_tokens" in usage else "output_tokens"
            LLM_INPUT_TOKENS.labels(
                provider=self.provider,
                model=model,
                agent_role=agent_role,
            ).inc(usage[input_key])

            LLM_OUTPUT_TOKENS.labels(
                provider=self.provider,
                model=model,
                agent_role=agent_role,
            ).inc(usage[output_key])
            log_line += f", input_tokens={usage[input_key]}, output_tokens={usage[output_key]}"

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
