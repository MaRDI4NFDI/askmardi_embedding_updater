import logging
import os
from pathlib import Path
from typing import Dict, Optional

import yaml
import logging.config
from prefect.blocks.system import Secret
from prefect.context import get_run_context
from prefect.exceptions import MissingContextError

from helper.logger import get_logger_safe

CONFIG_PATH = Path("config.yaml")

_cache = None  # keep config in memory
is_prefect_environment: bool = True
_cfg_log_once = False


def load_config(config_path: Path = CONFIG_PATH):
    """Load configuration and apply Prefect and environment overrides when available.

    Args:
        config_path: Optional path to the config file.

    Returns:
        dict: Parsed configuration data with Prefect and environment overrides applied.

    Raises:
        FileNotFoundError: If the expected config file does not exist.
    """
    global _cache
    logger = get_logger_safe()
    if _cache is not None:
        return _cache

    if not config_path.exists():
        raise FileNotFoundError(f"Config file missing: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        _cache = yaml.safe_load(f)
        if is_prefect_environment:
            _apply_prefect_lakefs_credentials(_cache, logger)
        else:
            logger.debug("Prefect environment not found; skipping overrides from lakeFS secrets.")
        _apply_env_overrides(_cache, logger)
        _populate_constants(_cache)
        return _cache


def _populate_constants(config: Dict) -> None:
    """Constants are initialized directly in helper.constants at import time."""
    return


def get_local_state_db_path() -> Path:
    """
    Resolve the local path for the state database using config.yaml.

    Returns:
        Path: Local filesystem path for the state DB.

    Raises:
        KeyError: If the required lakefs.state_db_filename is missing.
    """
    config = load_config()
    lakefs_cfg = config.get("lakefs", {})
    prefix = lakefs_cfg.get("state_db_filename_prefix")
    if not prefix:
        raise KeyError("Missing 'state_db_filename_prefix' in lakefs configuration (config.yaml -> lakefs.state_db_filename_prefix)")
    collection = config.get("qdrant", {}).get("collection")
    if not collection:
        raise KeyError("Missing 'collection' in qdrant configuration (config.yaml -> qdrant.collection)")
    db_filename = f"{prefix}__{collection}.db"
    return Path("state") / db_filename


def get_state_db_filename() -> str:
    """
    Resolve the filename for the state database using config.yaml.

    Returns:
        str: State DB filename from config.
    """
    return get_local_state_db_path().name


def check_for_config() -> bool:
    """Validate presence of ``config.yaml`` before running the flow.

    Returns:
        bool: True when the config file is present; False otherwise.
    """
    if CONFIG_PATH.exists():
        return True

    print(
        "config.yaml not found. Please copy config_example.yaml to config.yaml "
        "and fill in the required settings."
    )
    return False


def cfg(section: str, config_path: Path = CONFIG_PATH) -> dict:
    """
    Retrieve a configuration section by name.

    Args:
        section: Section key to fetch from the loaded config.
        config_path: Optional path to override default config location.

    Returns:
        dict: Subsection of configuration values.

    Raises:
        KeyError: If the section is not present in the config file.
    """
    global _cfg_log_once
    logger = get_logger_safe()
    if not _cfg_log_once:
        logger.debug(f"Checking for config file at: {config_path}")
        _cfg_log_once = True

    config = load_config(config_path)
    if section not in config:
        raise KeyError(f"Missing '{section}' section in config.yaml")
    return config[section]

def _apply_prefect_lakefs_credentials(config: Dict, logger) -> None:
    """Override LakeFS credentials with Prefect secrets when running in Prefect.

    Args:
        config: Loaded configuration dictionary to mutate.
        logger: Logger for informational messages.
    """
    try:
        get_run_context()
    except MissingContextError:
        logger.debug("Prefect context unavailable; skipping Prefect secret overrides.")
        return

    logger.debug("Prefect environment found; loading lakeFS credentials from secrets.")

    credentials = _load_credentials_from_prefect("lakefs", logger)
    if credentials:
        existing = config.get("lakefs") or {}
        updated = {**existing, **credentials}
        config["lakefs"] = updated
        logger.debug("lakeFS set from prefect secrets.")


def _load_credentials_from_prefect(name: str, logger) -> Optional[Dict[str, str]]:
    """Attempt to read credentials from Prefect secret blocks.

    Args:
        name: Base name for Prefect secret blocks.
        logger: Logger for status updates.

    Returns:
        dict | None: Credential mapping if retrieved; otherwise None.
    """
    try:
        user = Secret.load(f"{name}-user").get()
        password = Secret.load(f"{name}-password").get()
        return {"user": user, "password": password}
    except Exception:
        logger.debug(f"Could not read {name} credentials from Prefect.")
        return None


def _apply_env_overrides(config: Dict, logger) -> None:
    """Override config values using environment variables when provided."""
    _apply_env_lakefs_credentials(config, logger)
    _apply_env_qdrant_config(config, logger)


def _apply_env_lakefs_credentials(config: Dict, logger) -> None:
    """Override LakeFS credentials using environment variables when provided."""
    env_user = os.environ.get("LAKEFS_USER")
    env_password = os.environ.get("LAKEFS_PASSWORD")

    if not env_user and not env_password:
        logger.debug("No LakeFS environment credentials found; skipping overrides.")
        return

    existing = config.get("lakefs") or {}
    updated = {**existing}

    if env_user:
        updated["user"] = env_user
        logger.debug("LakeFS user loaded from environment variable LAKEFS_USER.")
    if env_password:
        updated["password"] = env_password
        logger.debug("LakeFS password loaded from environment variable LAKEFS_PASSWORD.")

    config["lakefs"] = updated


def _apply_env_qdrant_config(config: Dict, logger) -> None:
    """Override Qdrant configuration using environment variables when provided."""
    env_qdrant_url = os.environ.get("QDRANT_URL")
    if not env_qdrant_url:
        logger.debug("No Qdrant environment URL found; skipping override.")
        return

    existing = config.get("qdrant") or {}
    updated = {**existing, "url": env_qdrant_url}
    config["qdrant"] = updated
    logger.debug("Qdrant URL loaded from environment variable QDRANT_URL.")

def setup_prefect_logging() -> None:
    """
    Apply a logging configuration early so that it is picked up by Prefect's run logger.

    This must be called exactly once and before any Prefect flows, tasks,
    or get_run_logger() calls are executed.

    """
    LOGGING_CONFIG = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "prefect_with_threads": {
                "format": (
                    "%(asctime)s | %(levelname)s | %(name)s | "
                    "[thread=%(threadName)s id=%(thread)d] | %(message)s"
                )
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "prefect_with_threads",
                "stream": "ext://sys.stdout",
            }
        },
        "root": {
            "level": "INFO",
            "handlers": ["console"],
        },
    }

    # Prevent reconfiguration
    if getattr(setup_prefect_logging, "_configured", False):
        return

    logging.config.dictConfig(LOGGING_CONFIG)
    setup_prefect_logging._configured = True
