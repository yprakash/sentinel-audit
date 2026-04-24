import asyncio
import logging

from typing import Awaitable, Optional, TypeVar

T = TypeVar("T")
logger = logging.getLogger(__name__)


async def run_with_timeout_logging(
        coro: Awaitable[T],
        *,
        warn_after: float,
        timeout: float,
        name: Optional[str] = None,
        context: Optional[dict] = None,
        raise_on_timeout: bool = False,
) -> Optional[T]:
    """
    Execute an async operation with:
      - A warning log if it exceeds `warn_after` seconds
      - A hard timeout at `timeout` seconds

    Behavior:
      - If execution exceeds `warn_after`, a warning is logged (non-blocking)
      - If execution exceeds `timeout`, the task is canceled
      - If `raise_on_timeout=True`, TimeoutError is propagated
      - Otherwise, returns None on timeout

    Notes:
      - This is a wrapper around `asyncio.wait_for`, so cancellation is enforced
      - The warning is implemented via `loop.call_later` (lightweight, non-blocking)
      - Designed for reusable infra-level usage (DB calls, Kafka writes, API calls, etc.)
    """

    loop = asyncio.get_running_loop()
    task = asyncio.current_task()

    # Derive a readable label for logging
    label = name or getattr(getattr(coro, "__class__", None), "__name__", "async_task")

    def warn_callback():
        if not task.done():
            logger.warning(
                "%s still running after %.2fs | context=%s",
                label, warn_after, context
            )

    warn_handle = loop.call_later(warn_after, warn_callback)

    try:
        result = await asyncio.wait_for(coro, timeout=timeout)
        return result
    except asyncio.TimeoutError:
        if raise_on_timeout:
            raise
        else:
            logger.exception("%s timed out after %.2fs | context=%s", label, timeout, context)
        return None
    except asyncio.CancelledError:
        # Propagate cancellation (important for cooperative cancellation)
        logger.warning("%s was cancelled | context=%s", label, context)
        raise
    except Exception:
        # Real failure inside the coroutine
        raise
    finally:
        warn_handle.cancel()  # Always cancel the scheduled warning callback


# From here, all methods are just to test in local. Can't be used anywhere in application
async def main():
    import sys

    logging.basicConfig(
        level=logging.DEBUG,  # capture DEBUG and above
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        stream=sys.stdout,  # force stdout
    )
    logging.getLogger().setLevel(logging.DEBUG)

    # ---- test coroutines ----
    async def test_slow_task():
        await asyncio.sleep(2)
        return "done"

    async def test_error_task():
        await asyncio.sleep(0.5)
        raise ValueError("boom")

    # Case 1: completes but triggers warning (warn_after < runtime)
    res1 = await run_with_timeout_logging(
        test_slow_task(),
        warn_after=1.0,
        timeout=5.0,
        name="test_slow_task",
        context={"case": 1},
    )
    print("res1:", res1)

    # Case 2: timeout
    res2 = await run_with_timeout_logging(
        test_slow_task(),
        warn_after=1.0,
        timeout=1.5,
        name="timeout_task",
        context={"case": 2},
    )
    print("res2:", res2)

    # Case 3: exception inside coroutine
    try:
        await run_with_timeout_logging(
            test_error_task(),
            warn_after=1.0,
            timeout=3.0,
            name="test_error_task",
            context={"case": 3},
        )
    except Exception as e:
        print("caught:", repr(e))


if __name__ == "__main__":
    asyncio.run(main())
