import json
import logging
import logging.config
import os
import sys

from concurrent_log_handler import ConcurrentRotatingFileHandler

logger = logging.getLogger(__name__)


def setup_logging(
        root_dir,
        default_log_config='py_logging.json',
        default_level=logging.INFO,
        env_key='LOG_CFG'
):
    root_logger = logging.getLogger()
    # Force deterministic configuration
    if root_logger.handlers:
        root_logger.handlers.clear()

    value = os.getenv(env_key, None)
    if value:
        path = value
    else:
        path = os.path.join(root_dir, 'config', default_log_config)

    if os.path.exists(path):
        with open(path, 'rt') as f:
            config = json.load(f)
        logging.config.dictConfig(config)
    else:
        logs_dir = os.path.join(root_dir, 'logs')
        os.makedirs(logs_dir, exist_ok=True)
        log_path = os.path.join(logs_dir, 'pipeline.log')

        file_handler = ConcurrentRotatingFileHandler(
            log_path,
            maxBytes=2 ** 20,  # When log file is about to exceed this size, it is closed, renamed, and creates new file
            backupCount=10
        )
        stdout_handler = logging.StreamHandler(sys.stdout)

        formatter = logging.Formatter(
            "%(levelname)s %(asctime)s.%(msecs)03d %(threadName)s %(name)s:%(lineno)d %(message)s",
            "%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(formatter)
        stdout_handler.setFormatter(formatter)

        root_logger.setLevel(default_level)
        root_logger.addHandler(file_handler)
        root_logger.addHandler(stdout_handler)

    logger.info(
        "Logging initialized from %s at level: %s", path,
        logging.getLevelName(root_logger.getEffectiveLevel())
    )


def set_module_log_level(module, log_level=logging.INFO):
    if module:
        logging.getLogger(module).setLevel(log_level)
        logger.info('Changed log level for %s to %s', module, logging.getLevelName(log_level))
