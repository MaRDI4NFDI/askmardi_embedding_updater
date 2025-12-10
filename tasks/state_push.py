import sqlite3
from typing import Dict, Optional

from lakefs_client import ApiException
from prefect import task, get_run_logger

from helper.constants import STATE_DB_PATH
from helper.lakefs import upload_state_db, commit_state_db


@task(name="push_state_db_to_lakefs")
def push_state_db_to_lakefs(
    db_path: str = str(STATE_DB_PATH),
    baseline_counts: Optional[Dict[str, int]] = None,
) -> None:
    """
    Upload the local SQLite DB to LakeFS and create a commit when changes occurred.

    Args:
        db_path: Local path of the state DB file to persist.
        baseline_counts: Optional snapshot of table counts taken before this run.
    """
    logger = get_run_logger()
    logger.debug(f"[push_state_db] Start pushing state DB...")

    try:
        changed = upload_state_db(local_path=db_path)

        # If no changes, skip commit and return success
        if not changed:
            logger.debug("[push_state_db] No changes detected — skipping commit.")
            return

        try:
            current_counts = snapshot_table_counts(db_path)
            message = _format_commit_message(current_counts, baseline_counts)
            commit_state_db(message=message)
            logger.info("[push_state_db] New version of state DB committed successfully.")

        except ApiException as e:
            body = getattr(e, "body", "")
            if "no changes" in str(body).lower():
                logger.info("[push_state_db] Nothing to commit — already up-to-date")
            else:
                raise     # Re-raise real errors

    except Exception as e:
        logger.error(f"[push_state_db] Failed saving state DB: {e}")
        raise


def snapshot_table_counts(db_path: str = str(STATE_DB_PATH)) -> Dict[str, int]:
    """Collect row counts for key state tables.

    Args:
        db_path: Path to the SQLite state database.

    Returns:
        dict: Mapping of table name to row count.
    """
    tables = ["software_index", "component_index", "embeddings_index"]
    counts: Dict[str, int] = {}
    conn = sqlite3.connect(db_path)
    try:
        for table in tables:
            cursor = conn.execute(f"SELECT COUNT(*) FROM {table}")
            counts[table] = cursor.fetchone()[0]
    finally:
        conn.close()
    return counts


def _format_commit_message(current: Dict[str, int], baseline: Optional[Dict[str, int]]) -> str:
    """Create a commit message summarizing current counts and deltas.

    Args:
        current: Current table counts.
        baseline: Baseline counts captured before changes, if available.

    Returns:
        str: Human-friendly commit message.
    """
    parts = []
    for table, count in current.items():
        delta = None
        if baseline is not None:
            delta = count - baseline.get(table, 0)
        delta_str = f" ({delta:+})" if delta is not None else ""
        parts.append(f"{table}:{count}{delta_str}")
    summary = "; ".join(parts)
    return f"Updated state database for askmardi_embedding_updater [{summary}]"

