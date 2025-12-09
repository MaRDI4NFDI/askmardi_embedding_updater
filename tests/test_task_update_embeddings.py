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
        INSERT INTO component_index (qid, component, updated_at)
        VALUES (?, ?, ?)
        """,
        [
            ("Q1", "c1", "2024-01-01T00:00:00Z"),
            ("Q2", "c2", "2024-01-01T00:00:00Z"),
        ],
    )
    conn.commit()
    conn.close()

    monkeypatch.setattr("tasks.update_embeddings.get_run_logger", lambda: logging.getLogger("test_logger"))
    monkeypatch.setattr(
        "tasks.update_embeddings.perform_pdf_indexing",
        lambda components, db_path, **_: len(components),
    )
    monkeypatch.setattr(
        "tasks.update_embeddings.QdrantManager",
        lambda **_: type("Q", (), {"is_available": lambda self: True})(),
    )

    processed = update_embeddings.fn(str(db_path))

    conn = sqlite3.connect(str(db_path))
    cur = conn.execute("SELECT COUNT(*) FROM embeddings_index")
    count = cur.fetchone()[0]
    conn.close()

    assert processed == 2
    assert count == 0  # perform_pdf_indexing stubbed; no DB writes here
