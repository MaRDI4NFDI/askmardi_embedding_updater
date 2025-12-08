from datetime import datetime, timezone

from prefect import task, get_run_logger

from helper.constants import STATE_DB_PATH
from tasks.init_db_task import get_connection


@task(name="log_qids_with_components")
def get_software_items_with_pdf_component(
    db_path: str = str(STATE_DB_PATH),
) -> int:
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

    # First get an overview
    get_software_items_with_pdf_component.fn(db_path=db_path)

    logger.info("Updating embeddings_index from component_index")

    conn = get_connection(db_path)
    cursor = conn.cursor()

    # Get components that have a QIDs in software_index and
    # a matching entry in the component_index (=files in lakeFS).
    cursor.execute(
        """
        SELECT si.qid, ci.component, ci.checksum
        FROM software_index si
        JOIN component_index ci ON ci.qid = si.qid
        """
    )
    components = cursor.fetchall()
    logger.info(f"Found {len(components):,} component records to sync")

    if not components:
        conn.close()
        logger.info("No components to process; embeddings_index unchanged.")
        return 0


#    timestamp = datetime.now(timezone.utc).isoformat()
#    cursor.executemany(
#        """
#        INSERT OR REPLACE INTO embeddings_index
#            (qid, component, checksum, updated_at)
#        VALUES (?, ?, ?, ?)
#        """,
#        [(qid, component, checksum, timestamp) for qid, component, checksum in components],
#    )
#    conn.commit()
#    conn.close()

    logger.info("Embeddings index update complete")

    return len(components)
