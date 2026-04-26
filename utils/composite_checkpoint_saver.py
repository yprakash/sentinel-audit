import asyncio
import logging
import time
from typing import List

from langgraph.checkpoint.base import BaseCheckpointSaver
from prometheus_client import Counter, Histogram

from async_utils import run_with_timeout_logging
from utils.checkpointing_utils import get_checkpointers_async

SUCCESS = "success"
FAILURE = "failure"
logger = logging.getLogger(__name__)

CHECKPOINT_OPS_TOTAL = Counter(
    "checkpoint_ops_total",
    "Total checkpoint operations",
    ["backend", "operation", "status"],
)
CHECKPOINT_LATENCY = Histogram(
    "checkpoint_latency_seconds",
    "Checkpoint operation latency",
    ["backend", "operation"],
)


class CompositeCheckpointSaver(BaseCheckpointSaver):
    """
    CompositeCheckpointSaver

    A multi-backend checkpoint saver that separates the serving path from
    the audit/replication path while maintaining strong performance guarantees.

    Design:
    - checkpointers[0] → PRIMARY (read + write, blocking, source of truth)
    - checkpointers[1:] → SECONDARY (write-only, async, best-effort)

    Execution Model:
    - Primary writes are synchronous to guarantee correctness
    - Secondary writes are executed as background tasks
    - Concurrency is bounded via a semaphore to prevent overload
    - Excess tasks are naturally backpressured (queued on semaphore)

    Performance Strategy:
    - Secondary checkpoint writes are overlapped with downstream latency (e.g., LLM calls, tool execution)
    - This effectively hides I/O latency behind agent execution time
    - Ensures near-zero impact on end-to-end request latency

    Observability:
    - All operations are wrapped with timeout-aware logging
    - Latency, slow execution, and failures are captured centrally
    - Backend-agnostic (Redis, Kafka, Postgres, etc.)

    Failure Model:
    - Primary failure → propagated (request fails)
    - Secondary failure → logged and ignored (best-effort audit)

    Use Cases:
    - Fast state recovery (Redis)
    - Asynchronous audit/event logging (Kafka)
    - Future extensibility to relational or blob stores
    """

    def __init__(
            self,
            checkpointers: List[BaseCheckpointSaver],
            *,
            warn_after: float = 2,
            timeout: float = 5,
            max_concurrency: int = 100,
    ):
        if not checkpointers:
            raise ValueError("At least one checkpointer is required")

        super().__init__(serde=checkpointers[0].serde)

        self.checkpointers = checkpointers
        self._tasks: set[asyncio.Task] = set()
        self.warn_after = warn_after
        self.timeout = timeout
        # Use Semaphore for natural backpressure under heavy LOAD (~5K RPS)
        self._semaphore = asyncio.Semaphore(max_concurrency)  # ensures controlled parallelism, no overload

    def _record_metrics(self, backend: str, operation: str, start_time: float, success: bool):
        duration = time.perf_counter() - start_time

        CHECKPOINT_LATENCY.labels(
            backend=backend,
            operation=operation,
        ).observe(duration)

        CHECKPOINT_OPS_TOTAL.labels(
            backend=backend,
            operation=operation,
            status=SUCCESS if success else FAILURE,
        ).inc()

    def _spawn_task(self, coro):
        # spawns async task in the background. negligible cost, returns immediately
        task = asyncio.create_task(coro)  # Execution happens concurrently on event loop
        self._tasks.add(task)

        def _cleanup(t: asyncio.Task):
            self._tasks.discard(t)  # removes the reference from the set, but does not cancel anything
            if t.cancelled():
                logger.debug("Ignoring the Cancelled task %s", t)  # Remove NO need to log
                return  # expected, ignore

            exc = t.exception()
            if exc:
                logger.error("Background task failed", exc_info=exc)

        task.add_done_callback(_cleanup)

    async def _safe_aput(self, cp: BaseCheckpointSaver, config, checkpoint, metadata, new_versions):
        backend = cp.__class__.__name__
        start = time.perf_counter()

        async with self._semaphore:
            try:
                await run_with_timeout_logging(
                    cp.aput(config, checkpoint, metadata, new_versions),
                    warn_after=self.warn_after,
                    timeout=self.timeout,
                    name=f"{backend}.aput",
                    raise_on_timeout=False,
                )
                self._record_metrics(backend, "aput", start, True)
                logger.debug("Checkpoint written to %s", backend)
            except Exception:
                self._record_metrics(backend, "aput", start, False)
                logger.exception("Secondary write failed: %s", backend)

    async def aget_tuple(self, config, index=None):
        for i, cp in enumerate(self.checkpointers):
            backend = cp.__class__.__name__
            if index is not None and index != i:
                continue

            start = time.perf_counter()

            try:
                result = await cp.aget_tuple(config)
                self._record_metrics(backend, "aget_tuple", start, True)
                if result is not None:
                    return result
            except Exception:
                self._record_metrics(backend, "aget_tuple", start, False)
                logger.exception("%s.aget_tuple() FAILED", backend)

        return None

    async def aput(self, config, checkpoint, metadata, new_versions):
        primary = self.checkpointers[0]
        backend = primary.__class__.__name__
        start = time.perf_counter()

        agent_name = config.get("configurable", {}).get("agent_name", "None")
        if "checkpoint_ns" not in config["configurable"]:
            config["configurable"]["checkpoint_ns"] = agent_name

        try:
            result = await run_with_timeout_logging(  # blocking, must succeed
                primary.aput(config, checkpoint, metadata, new_versions),
                warn_after=self.warn_after,  # Warns if primary checkpointer delays
                timeout=self.timeout,  # FAIL FAST
                name=f"{backend}.aput",
                raise_on_timeout=True,  # Request should fail, MUST be True
            )
            self._record_metrics(backend, "aput", start, True)
            if logger.isEnabledFor(logging.DEBUG):
                logger.info("Checkpoint written to PRIMARY %s.aput in %.3f seconds",
                            backend, time.perf_counter() - start)

        except Exception:
            self._record_metrics(backend, "aput", start, False)
            logger.exception("PRIMARY write FAILED: %s.aput", backend)
            raise

        for i in range(1, len(self.checkpointers)):
            cp = self.checkpointers[i]
            self._spawn_task(
                self._safe_aput(cp, config, checkpoint, metadata, new_versions)
            )  # non-blocking secondary writes

        return result

    async def aclose(self):
        # Wait for background tasks up to `timeout` seconds, then cancel remaining tasks.
        logger.info("CompositeCheckpointSaver shutdown Initiated. len(pending_tasks): %d", len(self._tasks))
        if not self._tasks:
            return

        done, pending = await asyncio.wait(list(self._tasks), timeout=self.timeout)
        for t in pending:
            t.cancel()  # Schedules a CancelledError to be thrown into the task. Returns immediately
            # The task may still be: running, awaiting I/O, cleaning up in finally. So task is not finished yet

        await asyncio.gather(*pending, return_exceptions=True)  # Cancellation actually completes
        # Runs cleanup (finally blocks), Exits properly. Prevents “dangling tasks”
        logger.info("CompositeCheckpointSaver shutdown: %d done, %d cancelled", len(done), len(pending))

    async def alist(self, config, **kwargs):
        """
        Asynchronously iterates over checkpoints for a given thread.

        Purpose:
        - Provides a best-effort mechanism to inspect or traverse checkpoint history.
        - Intended for debugging, audit, and offline analysis use cases.

        Behavior:
        - Iterates over the first checkpointer that successfully returns results.
        - Does NOT merge results across multiple backends.
        - Stops after the first successful backend to avoid duplication and inconsistency.

        Guarantees:
        - No guarantee of complete history.
        - No guarantee of strict ordering across checkpoints.
        - Depends entirely on the underlying checkpointer implementation.

        Design Notes:
        - Primary backends like Redis typically store only the latest checkpoint, so iteration may yield a single item.
        - Secondary backends (e.g., Kafka, Postgres) may contain richer history,
          but are not used for reads in this composite design.

        Recommended Usage:
        - Debugging agent state transitions
        - Inspecting recent checkpoints
        - Lightweight audit or replay scenarios
        - Typically exposed via a debug/ops APIs (GET /debug/checkpoints?thread_id=...)

        Not Recommended For:
        - Production-critical logic
        - Strict replay or time-travel requirements
        - Systems requiring complete and ordered history

        For production execution, use `aget_tuple()` which provides O(1) access
        to the latest checkpoint and is part of the serving path.
        """
        for cp in self.checkpointers:
            backend = cp.__class__.__name__
            start = time.perf_counter()

            try:
                async for item in cp.alist(config, **kwargs):
                    yield item
                self._record_metrics(backend, "alist", start, True)
                return
            except Exception:
                self._record_metrics(backend, "alist", start, False)
                logger.exception("%s.alist() FAILED", backend)


async def main():
    import sys
    import uuid
    from datetime import datetime, UTC
    from utils.log_util import set_module_log_level
    from utils.kafka_utils import KafkaClientFactory

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    set_module_log_level('aiokafka')

    cps_to_test = ["AsyncKafkaSaver", "AsyncRedisSaver", "PostgreSQL"]  #, "PostgresWithPGvector"]
    for cp in cps_to_test:
        cp = await get_checkpointers_async([cp])
        assert len(cp) == 1

    thread_id = f"test-thread-{uuid.uuid4().hex[:8]}"
    config = {"configurable": {"thread_id": thread_id, "agent_name": "testing"}}
    logger.info("Testing with %s", config)
    checkpoint_id = str(uuid.uuid4())  # Dummy checkpoint + metadata
    checkpoint = {
        "id": checkpoint_id,
        "ts": datetime.now(UTC).isoformat(),
        "state": {"step": "test", "status": "ok"},
    }
    metadata = {"source": "connectivity-test"}
    new_versions = {}  # minimal

    checkpointers = await get_checkpointers_async(cps_to_test)
    composite_checkpointer = CompositeCheckpointSaver(checkpointers)

    try:
        # --- Test WRITE (aput) ---
        await composite_checkpointer.aput(config, checkpoint, metadata, new_versions)
        print("Checkpoint writes successful", checkpoint_id)

        # --- Test READ (aget_tuple) ---
        for i, ckpt in enumerate(checkpointers):
            try:
                result = await composite_checkpointer.aget_tuple(config, i)
                if result:
                    logger.info("Checkpoint read %d successful: %s", i, result.checkpoint)
            except Exception:
                logger.exception("Cehckpoint read FAILED")

        # --- Test LIST (alist) ---
        logger.info("Listing checkpoints:")
        async for item in composite_checkpointer.alist(config, limit=5):
            logger.info(item.checkpoint)

    except Exception:
        logger.exception("ERROR: CompositeCheckpointSaver test failed: ")
    finally:
        await KafkaClientFactory.close_all()

    print("=== Done ===")


if __name__ == "__main__":
    asyncio.run(main())
