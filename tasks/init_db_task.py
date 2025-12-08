from prefect import task, get_run_logger
import sqlite3
from pathlib import Path


@task(name="init_database")
def init_db_task(db_path: str) -> None:
    """
    Ensure the SQLite database exists and required tables are present.

    Args:
        db_path: Filesystem path to the SQLite DB file.
    """
    logger = get_run_logger()
    logger.info(f"Initializing SQLite database at: {db_path}")
    _init_db(db_path)
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
        checksum TEXT,
        updated_at TEXT,
        PRIMARY KEY (qid, component)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS embeddings_index (
        qid TEXT,
        component TEXT,
        checksum TEXT,
        updated_at TEXT,
        PRIMARY KEY (qid, component)
    );
    """
]


def _init_db(db_path: str) -> None:
    """
    Create the SQLite database directory and apply schema migrations.

    Args:
        db_path: Filesystem path to the SQLite DB file.
    """
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        for query in SCHEMA_QUERIES:
            conn.execute(query)
        conn.commit()
    finally:
        conn.close()


def get_connection(db_path: str) -> sqlite3.Connection:
    """
    Open a SQLite connection to the workflow state database.

    Args:
        db_path: Filesystem path to the SQLite DB file.

    Returns:
        sqlite3.Connection: Active connection to the DB.
    """
    return sqlite3.connect(db_path)
