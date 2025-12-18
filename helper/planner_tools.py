"""
Utilities for plan-based workflow execution.
"""

import logging
from pathlib import Path
from typing import List, Any
import json

from helper import config as config_helper
from helper.lakefs import get_lakefs_client
from helper.logger import get_logger_safe
from tasks.init_db_task import get_connection


def get_plan_from_lakefs(plan_name: str) -> str:
    """
    Download and return the contents of a plan file from LakeFS.

    Args:
        plan_name (str): Plan identifier, with or without .json
            (e.g., "plan_localworker_01").

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
    """
    Retrieve all (qid, component) rows where documentation PDFs exist.

    Returns:
        list[tuple[str, str]]: Tuples of QID and component path for CRAN docs
        that are present in LakeFS and referenced in the KG.
    """
    logger = get_logger_safe()

    logger.info("Updating embeddings_index from component_index")
    conn = get_connection()
    cursor = conn.cursor()

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


def convert_worker_plan_to_list(plan_content: str) -> List[Any]:
    """
    Convert a worker plan JSON string into a list of (qid, component) tuples.

    Args:
        plan_content: Raw JSON string of the plan file.

    Returns:
        list[tuple[str, str]]: Entries extracted from the plan.
    """
    logger = get_logger_safe()
    try:
        payload = json.loads(plan_content)
    except json.JSONDecodeError as exc:
        logger.error("Failed to parse plan JSON: %s", exc)
        return []

    entries = payload.get("entries", [])
    results: List[Any] = []
    for entry in entries:
        qid = entry.get("qid")
        component = entry.get("component")
        if qid and component:
            results.append((qid, component))
        else:
            logger.warning("Skipping plan entry missing qid/component: %s", entry)
    return results
