import logging
import sqlite3

from tasks.init_db_task import _init_db
from tasks.update_lakefs_file_index import update_file_index_from_lakefs


def test_update_lakefs_file_index_scans_s3_gateway(tmp_path, monkeypatch):
    db_path = tmp_path / "state.db"
    _init_db(str(db_path))

    pages = [
        {
            "Contents": [
                {"Key": "main/raw_html/Q123/doc1.html"},
                {"Key": "main/raw_pdf/Q987/doc2.PDF"},
                {"Key": "main/other/no_qid.txt"},
            ]
        }
    ]

    class FakePaginator:
        def paginate(self, Bucket, Prefix):
            assert Bucket == "repo"
            assert Prefix == "main/"
            return pages

    class FakeClient:
        def get_paginator(self, name):
            assert name == "list_objects_v2"
            return FakePaginator()

    def fake_get_s3_client():
        return FakeClient()

    monkeypatch.setattr(
        "tasks.update_lakefs_file_index.get_lakefs_s3_client", fake_get_s3_client
    )
    monkeypatch.setattr(
        "tasks.update_lakefs_file_index.cfg",
        lambda section: {
            "data_repo": "repo",
            "branch": "main",
            "url": "http://example.test",
            "user": "user",
            "password": "pass",
        },
    )
    monkeypatch.setattr("tasks.update_lakefs_file_index.get_run_logger", lambda: logging.getLogger("test_logger"))

    update_file_index_from_lakefs.fn(str(db_path))

    conn = sqlite3.connect(str(db_path))
    cur = conn.execute(
        "SELECT qid, component FROM component_index ORDER BY component"
    )
    rows = cur.fetchall()
    conn.close()

    assert rows == [
        ("Q123", "main/raw_html/Q123/doc1.html"),
        ("Q987", "main/raw_pdf/Q987/doc2.PDF"),
    ]
