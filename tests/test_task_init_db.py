import sqlite3

from tasks.init_db_task import _init_db


def test_init_db_creates_expected_tables(tmp_path):
    db_path = tmp_path / "state.db"

    # Patch the path resolver to target the temp DB.
    import tasks.init_db_task as init_module
    init_module.get_local_state_db_path = lambda: db_path
    _init_db()

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
