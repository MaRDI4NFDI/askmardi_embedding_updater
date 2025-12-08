from datetime import datetime, timezone

from prefect import task, get_run_logger

from helper.constants import STATE_DB_PATH
from tasks.init_db_task import get_connection


@task(name="log_qids_with_components")
def get_software_items_with_pdf_component(db_path: str = str(STATE_DB_PATH)) -> int:
    """
    Log the overlap between software_index and component_index.

    Prints the first 5 QIDs that exist in component_index among those listed
    in software_index, and returns the total overlap count.

    Args:
        db_path: Path to the workflow's SQLite state database.
    """
    logger = get_run_logger()
    conn = get_connection(db_path)
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT si.qid
        FROM software_index si
        JOIN component_index ci ON ci.qid = si.qid
        GROUP BY si.qid
        """
    )
    matching = [row[0] for row in cursor.fetchall()]
    conn.close()

    total = len(matching)
    sample = matching[:5]
    logger.info(f"QIDs with components ({total} total). Sample: {sample}")
    return total

def _do_embedding_update():
    logger = get_run_logger()
    conn = get_connection()
    cursor = conn.cursor()

    get_software_items_with_pdf_component()

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



@task(name="update_embeddings")
def update_embeddings(db_path: str = str(STATE_DB_PATH)) -> None:
    """
    Synchronize embeddings_index rows with the current component_index entries.

    Args:
        db_path: Path to the workflow's SQLite state database.

    Returns:
        None
    """
    logger = get_run_logger()

    # First get an overview
    get_software_items_with_pdf_component()

    logger.info("Updating embeddings_index from component_index")

    logger.info("Embeddings index update complete")
