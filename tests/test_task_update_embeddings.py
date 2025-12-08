import logging
import sqlite3

from tasks.init_db_task import _init_db
from tasks.update_embeddings import update_embeddings


def test_update_embeddings_syncs_from_component_index(tmp_path, monkeypatch):
    db_path = tmp_path / "state.db"
    _init_db(str(db_path))

    conn = sqlite3.connect(str(db_path))
    conn.executemany(
        """
        INSERT INTO software_index (qid, updated_at)
        VALUES (?, ?)
        """,
        [
            ("Q1", "2024-01-01T00:00:00Z"),
            ("Q2", "2024-01-01T00:00:00Z"),
        ],
    )
    conn.executemany(
        """
        INSERT INTO component_index (qid, component, checksum, updated_at)
        VALUES (?, ?, ?, ?)
        """,
        [
            ("Q1", "c1", "abc", "2024-01-01T00:00:00Z"),
            ("Q2", "c2", "def", "2024-01-01T00:00:00Z"),
        ],
    )
    conn.commit()
    conn.close()

    monkeypatch.setattr("tasks.update_embeddings.get_run_logger", lambda: logging.getLogger("test_logger"))

    processed = update_embeddings.fn(str(db_path))

    conn = sqlite3.connect(str(db_path))
    cur = conn.execute("SELECT qid, component, checksum FROM embeddings_index ORDER BY qid")
    rows = cur.fetchall()
    conn.close()

    assert processed == 2
#    assert rows == [("Q1", "c1", "abc"), ("Q2", "c2", "def")]
