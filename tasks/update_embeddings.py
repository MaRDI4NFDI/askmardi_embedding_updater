from datetime import datetime, timezone

from prefect import task, get_run_logger

from helper.constants import STATE_DB_PATH
from tasks.init_db_task import get_connection


@task(name="update_embeddings")
def update_embeddings(db_path: str = str(STATE_DB_PATH)) -> int:
    """
    Synchronize embeddings_index rows with the current component_index entries.

    Args:
        db_path: Path to the workflow's SQLite state database.

    Returns:
        int: Number of component records processed into embeddings_index.
    """
    logger = get_run_logger()
    logger.info("Updating embeddings_index from component_index")

    conn = get_connection(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT qid, component, checksum FROM component_index")
    components = cursor.fetchall()
    logger.info(f"Found {len(components):,} component records to sync")

    if not components:
        conn.close()
        return 0

    timestamp = datetime.now(timezone.utc).isoformat()
    cursor.executemany(
        """
        INSERT OR REPLACE INTO embeddings_index
            (qid, component, checksum, updated_at)
        VALUES (?, ?, ?, ?)
        """,
        [(qid, component, checksum, timestamp) for qid, component, checksum in components],
    )
    conn.commit()
    conn.close()

    logger.info("Embeddings index update complete")
    return len(components)
