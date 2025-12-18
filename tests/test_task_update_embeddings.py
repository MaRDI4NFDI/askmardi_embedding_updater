import logging
import sqlite3
from pathlib import Path

import pytest

from helper.planner_tools import get_cran_items_having_doc_pdf
from tasks.init_db_task import _init_db
from tasks.update_embeddings import update_embeddings


@pytest.mark.integration
def test_update_embeddings_syncs_from_component_index(tmp_path, monkeypatch):
    """Integration-style smoke test for update_embeddings pulling from component_index."""

    config_path = Path(__file__).resolve().parent.parent / "config.yaml"

    if not config_path.exists():
        pytest.skip("config.yaml not found; skipping LakeFS integration test")

    db_path = tmp_path / "state.db"
    monkeypatch.setattr("tasks.init_db_task.get_local_state_db_path", lambda: db_path)
    monkeypatch.setattr("helper.config.get_local_state_db_path", lambda: db_path)
    _init_db()

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
        "tasks.update_embeddings.embed_and_upload_all_PDFs",
        lambda components, **_: len(components),
    )
    monkeypatch.setattr(
        "tasks.update_embeddings.EmbedderTools",
        lambda *a, **k: type("E", (), {"embedding_dimension": 3})(),
    )
    monkeypatch.setattr(
        "tasks.update_embeddings.QdrantManager",
        lambda **_: type("Q", (), {"is_available": lambda self: True})(),
    )

    cran_items = get_cran_items_having_doc_pdf()
    processed = update_embeddings.fn(cran_items_having_doc_pdf=cran_items)

    conn = sqlite3.connect(str(db_path))
    cur = conn.execute("SELECT COUNT(*) FROM embeddings_index")
    count = cur.fetchone()[0]
    conn.close()

    assert processed == 2
    assert count == 0  # embed_and_upload_all_PDFs stubbed; no DB writes here
