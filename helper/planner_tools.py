"""
Utilities for plan-based workflow execution.
"""

import logging
from pathlib import Path
from typing import List, Any

from helper import config as config_helper
from helper.lakefs import get_lakefs_client
from helper.logger import get_logger_safe
from tasks.init_db_task import get_connection


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


def get_cran_items_having_doc_pdf() -> List[Any]:
    logger = get_logger_safe()

    logger.info("Updating embeddings_index from component_index")
    conn = get_connection()
    cursor = conn.cursor()

    # Get components that have a QIDs in table software_index (=CRAN package exists in MaRDI KG) and
    # a matching entry in the table component_index (=PDF documentation file exists in lakeFS).
    cursor.execute(
        """
        SELECT si.qid, ci.component
        FROM software_index si
                 JOIN component_index ci ON ci.qid = si.qid
        """
    )
    components: List[Any] = cursor.fetchall()
    total_components = len(components)

    cursor.execute(
        "SELECT COUNT(*) FROM embeddings_index"
    )
    already_embedded = cursor.fetchone()[0]
    remaining = max(total_components - already_embedded, 0)

    logger.info(
        f"Found {total_components:,} component records; {remaining:,} pending embeddings"
    )

    if not components:
        conn.close()
        logger.info("No components to process; embeddings_index unchanged.")
        return 0

    conn.close()

    return components