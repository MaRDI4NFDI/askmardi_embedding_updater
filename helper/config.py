import logging
import os
from pathlib import Path
from typing import Dict, Optional

import yaml
from prefect import get_run_logger
from prefect.blocks.system import Secret
from prefect.context import get_run_context
from prefect.exceptions import MissingContextError

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
    logger = _get_logger()
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
    """Fill constant placeholders with values from the loaded config.

    Args:
        config: Parsed configuration dictionary.
    """
    # Local import to avoid circular dependency at module import time.
    from helper import constants

    if constants.SOFTWARE_PROFILE_QID is None:
        constants.SOFTWARE_PROFILE_QID = config["mardi_kg"].get(
            "mardi_software_profile_qid"
        )

    if constants.MARDI_PROFILE_TYPE_PID is None:
        constants.MARDI_PROFILE_TYPE_PID = config["mardi_kg"].get(
            "mardi_profile_type_pid"
        )


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
    logger = _get_logger()
    if not _cfg_log_once:
        logger.debug(f"Checking for config file at: {config_path}")
        _cfg_log_once = True

    config = load_config(config_path)
    if section not in config:
        raise KeyError(f"Missing '{section}' section in config.yaml")
    return config[section]


def _get_logger():
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
