"""
Utilities for plan-based workflow execution.
"""

import logging
from pathlib import Path

from helper import config as config_helper
from helper.lakefs import get_lakefs_client
from helper.logger import get_logger_safe


def get_plan_from_lakefs(plan_name: str) -> str:
    """
    Download and return the contents of a plan file from LakeFS.

    Args:
        plan_name: Plan identifier, with or without .json (e.g., "plan_localworker_01").

    Returns:
        str | None: The JSON string content of the plan file, or None if not found.
    """
    logger = get_logger_safe()
    lakefs = get_lakefs_client()
    config = config_helper.load_config()
    lakefs_cfg = config.get("lakefs", {})
    repo = lakefs_cfg["state_repo"]
    branch = lakefs_cfg["branch"]
    directory_prefix = lakefs_cfg.get("state_repo_directory", "").strip("/")

    filename = plan_name if plan_name.endswith(".json") else f"{plan_name}.json"
    path_in_repo = "/".join(filter(None, [directory_prefix, "planned", filename]))

    logger.info("Fetching plan file from %s:%s/%s", repo, branch, path_in_repo)
    try:
        obj = lakefs.objects_api.get_object(
            repository=repo,
            ref=branch,
            path=path_in_repo,
            _preload_content=False,
        )
        content = obj.data.decode("utf-8") if hasattr(obj, "data") else obj.read().decode("utf-8")
        return content
    except Exception:
        logger.error("Plan file not found or could not be read: %s", path_in_repo)
        return None
