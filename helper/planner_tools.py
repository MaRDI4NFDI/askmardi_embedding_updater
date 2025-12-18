"""
Utilities for plan-based workflow execution.
"""

import logging

from helper import config as config_helper
from helper.lakefs import get_lakefs_client
from helper.logger import get_logger_safe


def plan_exists_on_lakefs(plan_name: str) -> bool:
    """
    Verify that a given plan file exists in LakeFS under the planned/ prefix.

    Args:
        plan_name: Plan identifier, with or without .json (e.g., "plan_localworker_01").
        logger: Logger for status and error messages.

    Returns:
        True or False.

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

    logger.info("Looking for plan file at %s:%s/%s", repo, branch, path_in_repo)
    try:
        lakefs.objects_api.stat_object(repo, branch, path_in_repo)
    except Exception:
        logger.error("Plan file not found in LakeFS: %s", path_in_repo)
        return False

    return True
