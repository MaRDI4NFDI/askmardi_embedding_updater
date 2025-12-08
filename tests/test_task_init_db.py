import sqlite3

from tasks.init_db_task import _init_db


def test_init_db_creates_expected_tables(tmp_path):
    db_path = tmp_path / "state.db"

    _init_db(str(db_path))

    conn = sqlite3.connect(str(db_path))
    cursor = conn.execute(
        """
        SELECT name FROM sqlite_master
        WHERE type='table' AND name IN (
            'software_index', 'component_index', 'embeddings_index'
        )
        """
    )
    tables = {row[0] for row in cursor.fetchall()}
    conn.close()

    assert tables == {"software_index", "component_index", "embeddings_index"}
