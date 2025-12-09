import io
import sqlite3
from datetime import datetime, timezone

import pytest
from langchain_core.documents import Document

from tasks.init_db_task import _init_db
from tasks.update_embeddings import perform_pdf_indexing, update_embeddings


@pytest.fixture()
def temp_db(tmp_path):
    db_path = tmp_path / "state.db"
    _init_db(str(db_path))
    return db_path


class FakeS3Client:
    def __init__(self, data: bytes):
        self.data = data
        self.calls = []

    def get_object(self, Bucket, Key):
        self.calls.append((Bucket, Key))
        return {"Body": io.BytesIO(self.data)}


class FakeEmbedder:
    def __init__(self):
        self.embedding_dimension = 3

    def load_pdf_file(self, path):
        return [Document(page_content="test content", metadata={})]

    def split_and_filter(self, docs, min_length=250):
        return [Document(page_content="chunked text", metadata=docs[0].metadata.copy())]

    def embed_text(self, text):
        return [1.0, 2.0, 3.0]

    def format_documents(self, docs, limit=5):
        return ""


class FakeQdrantManager:
    def __init__(self, **kwargs):
        self.uploaded = []

    def ensure_collection(self, vector_size: int):
        return None

    def upload_documents(self, documents, embed_fn, id_prefix=None):
        self.uploaded.append((documents, id_prefix))


def test_perform_pdf_indexing_inserts_embeddings(monkeypatch, temp_db):
    conn = sqlite3.connect(str(temp_db))
    conn.executemany(
        """
        INSERT INTO software_index (qid, updated_at)
        VALUES (?, ?)
        """,
        [("Q1", datetime.now(timezone.utc).isoformat())],
    )
    conn.executemany(
        """
        INSERT INTO component_index (qid, component, updated_at)
        VALUES (?, ?, ?)
        """,
        [("Q1", "path/to/file.pdf", datetime.now(timezone.utc).isoformat())],
    )
    conn.commit()
    conn.close()

    fake_s3 = FakeS3Client(b"pdf-bytes")
    monkeypatch.setattr("tasks.update_embeddings.get_lakefs_s3_client", lambda: fake_s3)
    monkeypatch.setattr("tasks.update_embeddings.EmbedderTools", lambda *a, **k: FakeEmbedder())
    fake_qdrant = FakeQdrantManager()
    monkeypatch.setattr("tasks.update_embeddings.QdrantManager", lambda **k: fake_qdrant)
    monkeypatch.setattr("tasks.update_embeddings.cfg", lambda section: {"data_repo": "bucket"} if section == "lakefs" else {})
    monkeypatch.setattr("tasks.update_embeddings.get_run_logger", lambda: type("L", (), {"info": lambda *a, **k: None, "debug": lambda *a, **k: None, "warning": lambda *a, **k: None})())

    processed = perform_pdf_indexing(
        components=[("Q1", "path/to/file.pdf")], db_path=str(temp_db)
    )

    conn = sqlite3.connect(str(temp_db))
    rows = conn.execute("SELECT qid, component FROM embeddings_index").fetchall()
    conn.close()

    assert processed == 1
    assert rows == [("Q1", "path/to/file.pdf")]
    assert fake_qdrant.uploaded, "Vectors should be uploaded to Qdrant"


def test_update_embeddings_returns_processed_count(monkeypatch, temp_db):
    conn = sqlite3.connect(str(temp_db))
    conn.executemany(
        """
        INSERT INTO software_index (qid, updated_at)
        VALUES (?, ?)
        """,
        [("Q1", "2024-01-01T00:00:00Z"), ("Q2", "2024-01-01T00:00:00Z")],
    )
    conn.executemany(
        """
        INSERT INTO component_index (qid, component, updated_at)
        VALUES (?, ?, ?)
        """,
        [
            ("Q1", "file1.pdf", "2024-01-01T00:00:00Z"),
            ("Q2", "file2.pdf", "2024-01-01T00:00:00Z"),
        ],
    )
    conn.commit()
    conn.close()

    fake_s3 = FakeS3Client(b"pdf-bytes")
    monkeypatch.setattr("tasks.update_embeddings.get_lakefs_s3_client", lambda: fake_s3)
    monkeypatch.setattr("tasks.update_embeddings.EmbedderTools", lambda *a, **k: FakeEmbedder())
    fake_qdrant = FakeQdrantManager()
    monkeypatch.setattr("tasks.update_embeddings.QdrantManager", lambda **k: fake_qdrant)
    monkeypatch.setattr(
        "tasks.update_embeddings.cfg",
        lambda section: {"data_repo": "bucket"} if section == "lakefs" else {},
    )
    monkeypatch.setattr(
        "tasks.update_embeddings.get_run_logger",
        lambda: type(
            "L", (), {"info": lambda *a, **k: None, "debug": lambda *a, **k: None, "warning": lambda *a, **k: None}
        )(),
    )

    processed = update_embeddings.fn(str(temp_db))
    assert processed == 2
