import logging

from prefect import get_run_logger
from prefect.exceptions import MissingContextError


def get_logger_safe():
    """Return an available logger for Prefect or local execution.

    Returns:
        logging.Logger: Prefect run logger when available, otherwise a module logger.
    """
    try:
        return get_run_logger()
    except MissingContextError:
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
        return logger
