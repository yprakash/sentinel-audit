from __future__ import annotations

import asyncio
import logging
import signal
import time
from typing import Awaitable, Callable, Optional, Protocol

logger = logging.getLogger(__name__)


class Shutdownable(Protocol):
    async def shutdown(self) -> None:
        ...


class _ShutdownManager:
    def __init__(
            self,
            *,
            per_component_timeout: float = 10.0,  # max time per component shutdown
            global_timeout: Optional[float] = 20.0,  # overall shutdown deadline (None = no limit)
    ):
        self.per_component_timeout = per_component_timeout
        self.global_timeout = global_timeout
        self._shutdown_started = False
        self._lock = asyncio.Lock()
        self._callbacks: list[Callable[[], Awaitable[None]]] = []

    def register(self, component: Shutdownable) -> None:
        if self._shutdown_started:
            raise RuntimeError("Cannot register new components during shutdown")
        self._callbacks.append(component.shutdown)

    def register_callback(self, fn: Callable[[], Awaitable[None]]) -> None:
        if self._shutdown_started:
            raise RuntimeError("Cannot register new callbacks during shutdown")
        self._callbacks.append(fn)

    async def shutdown(self) -> None:
        start_time = time.perf_counter()

        async with self._lock:
            if self._shutdown_started:
                return
            self._shutdown_started = True

        def _name(fn):
            return getattr(fn, "__qualname__", repr(fn))

        async def _run_target(target: Callable[[], Awaitable[None]]) -> None:
            try:
                await asyncio.wait_for(target(), timeout=self.per_component_timeout)
            except asyncio.TimeoutError:
                logger.warning("Timeout: %s", _name(target))
            except asyncio.CancelledError:
                raise  # propagate cancellation
            except Exception:
                logger.exception("Error in %s", _name(target))

        callbacks = list(self._callbacks)  # Use snapshot to avoid race conditions
        logger.info(f"Starting shutdown of {len(callbacks)} components")
        tasks = [_run_target(t) for t in callbacks]

        try:
            if self.global_timeout:
                await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True),
                    timeout=self.global_timeout,
                )
            else:
                await asyncio.gather(*tasks, return_exceptions=True)
        except asyncio.TimeoutError:
            logger.warning("Global shutdown timeout reached")
        finally:
            for t in tasks:
                if not t.done():
                    t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

        logger.info(f"Shutdown completed in %.3f seconds", time.perf_counter() - start_time)


shutdown_event = asyncio.Event()


def setup_signal_handlers(loop: asyncio.AbstractEventLoop | None = None):
    def _handler():
        shutdown_event.set()
        logger.info("Shutdown process initiated")

    loop = loop or asyncio.get_running_loop()
    loop.add_signal_handler(signal.SIGINT, _handler)
    loop.add_signal_handler(signal.SIGTERM, _handler)


shutdown_manager = _ShutdownManager()

__all__ = ["setup_signal_handlers", "shutdown_event", "shutdown_manager"]
