from datetime import datetime, timezone
from typing import List

from prefect import task, get_run_logger

from helper.lakefs import list_components
from tasks.init_db_task import get_connection


@task(name="update_lakefs_file_index")
def update_lakefs_file_index(qids: List[str], db_path: str) -> None:
    """
    Refresh the component_index table with LakeFS component listings.

    Args:
        qids: Wikibase QIDs whose components should be indexed.
        db_path: Path to the workflow's SQLite state database.
    """
    logger = get_run_logger()
    logger.info(f"Updating LakeFS file index for {len(qids):,} QIDs")

    if not qids:
        logger.info("No QIDs provided; skipping LakeFS file index update.")
        return

    conn = get_connection(db_path)
    cursor = conn.cursor()
    timestamp = datetime.now(timezone.utc).isoformat()

    for qid in qids:
        components = list_components(qid)
        if not components:
            logger.info(f"No components found in LakeFS for {qid}")
            continue

        cursor.executemany(
            """
            INSERT OR REPLACE INTO component_index
                (qid, component, checksum, updated_at)
            VALUES (?, ?, ?, ?)
            """,
            [(qid, component, checksum, timestamp) for component, checksum in components],
        )
        conn.commit()
        logger.info(f"Indexed {len(components):,} components for {qid}")

    conn.close()
    logger.info("LakeFS file index update complete.")
