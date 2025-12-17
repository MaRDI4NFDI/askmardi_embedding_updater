import logging

from prefect import get_run_logger
from prefect.exceptions import MissingContextError

_LOG_FORMAT = (
    "%(asctime)s | %(levelname)s | %(name)s | "
    "[thread=%(threadName)s id=%(thread)d] | %(message)s"
)

def get_logger_safe() -> logging.Logger:
    """Return an available logger for Prefect or local execution."""
    try:
        # Prefect logger – do NOT touch handlers or formatters
        return get_run_logger()

    except MissingContextError:
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter(_LOG_FORMAT))
            logger.addHandler(handler)

        logger.propagate = False
        return logger
