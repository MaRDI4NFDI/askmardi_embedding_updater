import yaml
from pathlib import Path
from typing import Dict

CONFIG_PATH = Path("config.yaml")

_cache = None  # keep config in memory


def load_config():
    """
    Load and cache configuration from `config.yaml`.

    Returns:
        dict: Parsed configuration data.

    Raises:
        FileNotFoundError: If the expected config file does not exist.
    """
    global _cache
    if _cache is not None:
        return _cache

    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Config file missing: {CONFIG_PATH}")

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        _cache = yaml.safe_load(f)
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


def cfg(section: str) -> dict:
    """
    Retrieve a configuration section by name.

    Args:
        section: Section key to fetch from the loaded config.

    Returns:
        dict: Subsection of configuration values.

    Raises:
        KeyError: If the section is not present in the config file.
    """
    config = load_config()
    if section not in config:
        raise KeyError(f"Missing '{section}' section in config.yaml")
    return config[section]
