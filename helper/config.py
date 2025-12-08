import yaml
from pathlib import Path

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
        return _cache


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
