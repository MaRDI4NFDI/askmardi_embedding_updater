import logging
import sqlite3
from typing import List, Tuple

from tasks.init_db_task import _init_db
from tasks.update_lakefs_file_index import update_lakefs_file_index


def test_update_lakefs_file_index_persists_components(tmp_path, monkeypatch):
    db_path = tmp_path / "state.db"
    _init_db(str(db_path))

    components_for_qid: List[Tuple[str, None]] = [("comp1.txt", None), ("comp2.txt", None)]

    monkeypatch.setattr("tasks.update_lakefs_file_index.list_components", lambda qid: components_for_qid)
    monkeypatch.setattr("tasks.update_lakefs_file_index.get_run_logger", lambda: logging.getLogger("test_logger"))

    update_lakefs_file_index.fn(["Q123"], str(db_path))

    conn = sqlite3.connect(str(db_path))
    cur = conn.execute("SELECT qid, component FROM component_index ORDER BY component")
    rows = cur.fetchall()
    conn.close()

    assert rows == [("Q123", "comp1.txt"), ("Q123", "comp2.txt")]
