import io
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pytest
from langchain_core.documents import Document

from tasks.init_db_task import _init_db
from tasks.update_embeddings import embed_and_upload_all_PDFs, update_embeddings


@pytest.fixture()
def temp_db(tmp_path, monkeypatch):
    """Create a temporary state DB and patch helpers to point to it."""
    db_path = tmp_path / "state.db"
    monkeypatch.setattr("tasks.init_db_task.get_local_state_db_path", lambda: db_path)
    monkeypatch.setattr("helper.config.get_local_state_db_path", lambda: db_path)
    monkeypatch.setattr("tasks.update_embeddings.get_connection", lambda: sqlite3.connect(str(db_path)))
    _init_db()
    return db_path


class FakeS3Client:
    """Stub S3 client returning predefined bytes."""

    def __init__(self, data: bytes):
        """Store payload bytes and reset call log."""
        self.data = data
        self.calls = []

    def get_object(self, Bucket, Key):
        """Record the call and return a fake body stream."""
        self.calls.append((Bucket, Key))
        return {"Body": io.BytesIO(self.data)}


class FakeEmbedder:
    """Simple embedder stub with deterministic outputs."""

    def __init__(self):
        """Initialize embedding dimension and chunk defaults."""
        self.embedding_dimension = 3
        self.chunk_params = {"min_chunk_size": 250}

    def load_pdf_file(self, path):
        """Return a single-page document for testing."""
        return [Document(page_content="test content", metadata={})]

    def split_and_filter(self, documents=None, min_length=250, **kwargs):
        """Return one chunk that meets the minimum length requirement."""
        return [Document(page_content="x" * 300, metadata=documents[0].metadata.copy())]

    def embed_text(self, text):
        """Return a fixed embedding vector for text."""
        return [1.0, 2.0, 3.0]

    def embed_document(self, doc):
        """Return a fixed embedding vector for a document."""
        return [1.0, 2.0, 3.0]


class FakeQdrantManager:
    """Stubbed Qdrant manager that records uploads."""

    def __init__(self, **kwargs):
        """Initialize with empty upload log."""
        self.uploaded = []

    def is_available(self):
        """Always report the service as reachable."""
        return True

    def ensure_collection(self, vector_size: int):
        """No-op collection creation for tests."""
        return None

    def collection_size(self):
        """Return a fake collection size to satisfy logging."""
        return 0

    def embed_and_upload_documents(self, documents, embed_fn, id_prefix=None):
        """Delegate to upload_documents to mimic upload behavior."""
        return self.upload_documents(documents, embed_fn, id_prefix=id_prefix)

    def upload_documents(self, documents, embed_fn, id_prefix=None):
        """Simulate uploading documents by invoking embed_fn and logging count."""
        for idx, doc in enumerate(documents):
            _ = embed_fn(doc)  # ensure callable works
            assert hasattr(doc, "metadata")
            assert hasattr(doc, "page_content")
        self.uploaded.append(len(documents))
        return None


def test_perform_pdf_indexing_inserts_embeddings(monkeypatch, temp_db):
    """End-to-end-ish test of embedding and DB update with stubs."""
    conn = sqlite3.connect(str(temp_db))
    conn.execute(
        """
        INSERT INTO software_index (qid, updated_at)
        VALUES (?, ?)
        """,
        ("Q1", datetime.now(timezone.utc).isoformat()),
    )
    conn.execute(
        """
        INSERT INTO component_index (qid, component, updated_at)
        VALUES (?, ?, ?)
        """,
        ("Q1", "path/to/file.pdf", datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()
    conn.close()

    monkeypatch.setattr(
        "tasks.update_embeddings.download_file",
        lambda key, dest_path: Path(dest_path).write_bytes(b"pdf-bytes"),
    )
    monkeypatch.setattr("tasks.update_embeddings.EmbedderTools", lambda *a, **k: FakeEmbedder())
    fake_qdrant = FakeQdrantManager()
    monkeypatch.setattr("tasks.update_embeddings.QdrantManager", lambda **k: fake_qdrant)
    monkeypatch.setattr(
        "tasks.update_embeddings.cfg",
        lambda section: (
            {"data_repo": "bucket"}
            if section == "lakefs"
            else {"collection": "test_collection"}
            if section == "qdrant"
            else {"model_name": "stub"}
            if section == "embedding"
            else {}
        ),
    )
    monkeypatch.setattr("tasks.update_embeddings.get_run_logger", lambda: type("L", (), {"info": lambda *a, **k: None, "debug": lambda *a, **k: None, "warning": lambda *a, **k: None})())
    monkeypatch.setattr("tasks.update_embeddings.get_connection", lambda: sqlite3.connect(str(temp_db)))
    # Avoid loading real lakefs client
    monkeypatch.setattr("tasks.update_embeddings.download_file", lambda key, dest_path: Path(dest_path).write_bytes(b"pdf-bytes"))

    processed = embed_and_upload_all_PDFs(
        components=[("Q1", "path/to/file.pdf")]
    )

    conn = sqlite3.connect(str(temp_db))
    rows = conn.execute("SELECT qid, component, status FROM embeddings_index").fetchall()
    conn.close()

    assert rows == [("Q1", "path/to/file.pdf", "ok")]
    assert processed == 1
    assert fake_qdrant.uploaded, "Vectors should be uploaded to Qdrant"


def test_update_embeddings_returns_processed_count(monkeypatch, temp_db):
    """Ensure update_embeddings reports processed count when embedding succeeds."""
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

    monkeypatch.setattr(
        "tasks.update_embeddings.download_file",
        lambda key, dest_path: Path(dest_path).write_bytes(b"pdf-bytes"),
    )
    def fake_embed_and_upload_all_PDFs(
        components,
        qdrant_manager=None,
        max_number_of_pdfs=None,
        document_type=None,
        embedder=None,
        timeout_seconds=None,
        max_pages=None,
    ):
        """Return the number of components to mimic successful embedding."""
        return len(components)

    monkeypatch.setattr(
        "tasks.update_embeddings.embed_and_upload_all_PDFs",
        fake_embed_and_upload_all_PDFs,
    )
    monkeypatch.setattr("tasks.update_embeddings.EmbedderTools", lambda *a, **k: FakeEmbedder())
    fake_qdrant = FakeQdrantManager()
    monkeypatch.setattr("tasks.update_embeddings.QdrantManager", lambda **k: fake_qdrant)
    monkeypatch.setattr(
        "tasks.update_embeddings.cfg",
        lambda section: (
            {"data_repo": "bucket"}
            if section == "lakefs"
            else {"collection": "test_collection"}
            if section == "qdrant"
            else {"model_name": "stub"}
            if section == "embedding"
            else {}
        ),
    )
    monkeypatch.setattr(
        "tasks.update_embeddings.get_run_logger",
        lambda: type(
            "L", (), {"info": lambda *a, **k: None, "debug": lambda *a, **k: None, "warning": lambda *a, **k: None}
        )(),
    )
    monkeypatch.setattr("tasks.update_embeddings.EmbedderTools", lambda *a, **k: FakeEmbedder())
    monkeypatch.setattr("tasks.update_embeddings.get_connection", lambda: sqlite3.connect(str(temp_db)))

    from helper.planner_tools import get_cran_items_having_doc_pdf

    cran_items = get_cran_items_having_doc_pdf()

    processed = update_embeddings.fn(cran_items_having_doc_pdf=cran_items)
    assert processed == 2
