import asyncio
import logging
import os
import signal

from dotenv import load_dotenv, find_dotenv

from utils.log_util import setup_logging

logger = logging.getLogger(__name__)


def register_shutdown(loop: asyncio.AbstractEventLoop):
    # Call it: Inside your async entrypoint, After init()
    # After the loop exists and Before starting long-running tasks
    async def shutdown():
        logger.info("Shutdown initiated...")
        tasks = [t for t in asyncio.all_tasks(loop)
                 if t is not asyncio.current_task()]
        for task in tasks:
            task.cancel()

        await asyncio.gather(*tasks, return_exceptions=True)

        logging.shutdown()  # Flush logging before exit
        loop.stop()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(  # Not supported on Windows event loop policy
                sig,
                lambda s=sig: asyncio.create_task(shutdown())
            )
        except NotImplementedError:
            logger.warning("Signal handlers not supported on this platform")


def init(app_name: str | None = None):
    dotenv_path = find_dotenv()
    if not dotenv_path:
        raise RuntimeError(".env file not found")

    load_dotenv(dotenv_path)
    root_dir = os.path.dirname(dotenv_path)
    setup_logging(root_dir)
    logger.info('Initializing %s service..', app_name or '')


def get_env(key, default=None, throw=True):
    # though configparser can Organize (configuration settings into sections/hierarchical),
    # all other ways are Not cloud-native and Less secure for sensitive data
    val = os.getenv(key, default)
    if val is None:
        if throw and default is None:
            raise RuntimeError(f'env variable "{key}" not found. Please check')
        val = default
    return val
