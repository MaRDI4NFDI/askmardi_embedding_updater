from prefect import task, get_run_logger
import sqlite3
from pathlib import Path

from helper.config import get_local_state_db_path


@task(name="init_database")
def init_db_task() -> None:
    """
    Ensure the SQLite database exists and required tables are present.
    """
    logger = get_run_logger()
    resolved_path = str(get_local_state_db_path())
    logger.info(f"Initializing SQLite database at: {resolved_path}")
    _init_db()
    logger.info("Database initialized (or already existed)")



SCHEMA_QUERIES = [
    """
    CREATE TABLE IF NOT EXISTS software_index (
        qid TEXT PRIMARY KEY,
        updated_at TEXT
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS component_index (
        qid TEXT,
        component TEXT,
        updated_at TEXT,
        PRIMARY KEY (qid, component)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS embeddings_index (
        qid TEXT,
        component TEXT,
        updated_at TEXT,
        status TEXT DEFAULT 'ok',
        PRIMARY KEY (qid, component)
    );
    """
]


def _init_db() -> None:
    """
    Create the SQLite database directory and apply schema migrations.
    """
    resolved_path = str(get_local_state_db_path())
    Path(resolved_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(resolved_path)
    try:
        for query in SCHEMA_QUERIES:
            conn.execute(query)
        conn.commit()
    finally:
        conn.close()


def get_connection() -> sqlite3.Connection:
    """
    Open a SQLite connection to the workflow state database.

    Returns:
        sqlite3.Connection: Active connection to the DB.
    """
    resolved_path = str(get_local_state_db_path())
    return sqlite3.connect(resolved_path)
