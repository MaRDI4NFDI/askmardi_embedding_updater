from langchain_core.documents import Document

from helper import qdrant_manager


class DummyCollectionInfo:
    """Minimal collection info stub used for testing."""
    def __init__(self, points_count=0, payload_schema=None):
        """Lightweight stand-in for collection metadata.

        Args:
            points_count: Number of stored points in the collection.
            payload_schema: Optional schema dict; defaults to having page_content indexed.
        """
        self.points_count = points_count
        # Ensure payload_schema exists so _ensure_text_index short-circuits in tests.
        self.payload_schema = payload_schema or {"page_content": {}}


class DummyCollections:
    """Container holding fake collection descriptors."""

    def __init__(self, names):
        """Initialize with a list of collection names."""
        self.collections = [type("C", (), {"name": n}) for n in names]


class DummyClient:
    """Minimal QdrantClient stub for unit tests."""

    def __init__(self):
        """Set up internal tracking flags and placeholders."""
        self.created = False
        self.deleted = False
        self.uploaded = []
        self.queries = []
        self.collections = DummyCollections([])
        self.collection_info = DummyCollectionInfo()

    def get_collections(self):
        """Return available collections."""
        return self.collections

    def delete_collection(self, name):
        """Mark that a collection deletion was requested."""
        self.deleted = True

    def create_collection(self, **kwargs):
        """Mark creation and record the created collection."""
        self.created = True
        self.collections = DummyCollections([kwargs["collection_name"]])

    def upload_points(self, collection_name, points):
        """Record uploaded points for assertion."""
        self.uploaded.append((collection_name, points))

    def query_points(self, **kwargs):
        """Record query parameters and return a dummy result."""
        self.queries.append(kwargs)
        point = type(
            "Point",
            (),
            {"payload": {"page_content": "text", "meta": "m"}},
        )
        return type("Result", (), {"points": [point]})

    def get_collection(self, collection_name):
        """Return the stored collection info stub."""
        return self.collection_info


def test_qdrant_manager_ensure_and_upload(monkeypatch):
    """Ensure collection creation and upload call paths work."""
    client = DummyClient()
    monkeypatch.setattr(
        qdrant_manager,
        "QdrantClient",
        lambda **_: client,
    )
    # Lightweight stand-in for qdrant models
    class FakeModels:
        """Lightweight stand-in for qdrant_client.models."""

        class Distance:
            """Fake distance enum."""
            COSINE = "COSINE"

        class VectorParams:
            """Stub vector parameter container."""
            def __init__(self, size, distance, on_disk):
                """Store vector configuration parameters."""
                self.size = size
                self.distance = distance
                self.on_disk = on_disk

        class HnswConfigDiff:
            """Stub HNSW configuration container."""
            def __init__(self, on_disk):
                """Store HNSW configuration."""
                self.on_disk = on_disk

        class PointStruct:
            """Stub point struct mirroring qdrant model API."""
            def __init__(self, id, vector, payload):
                """Capture point fields for upload."""
                self.id = id
                self.vector = vector
                self.payload = payload

    monkeypatch.setattr(qdrant_manager, "models", FakeModels)

    mgr = qdrant_manager.QdrantManager(collection_name="test")
    mgr.ensure_collection(vector_size=10)
    assert client.created is True

    doc = Document(page_content="hello", metadata={})
    mgr.embed_and_upload_documents([doc], embed_fn=lambda d: [1.0], id_prefix="Q1")
    assert client.uploaded
    collection, points = client.uploaded[0]
    assert collection == "test"
    assert points[0].payload["page_content"] == "hello"
    assert points[0].id == "cd323cb5-8784-58c5-85a9-4419450f4724"


def test_qdrant_manager_query(monkeypatch):
    """Verify query returns documents mapped from the dummy client."""
    client = DummyClient()
    monkeypatch.setattr(
        qdrant_manager,
        "QdrantClient",
        lambda **_: client,
    )
    monkeypatch.setattr(qdrant_manager, "models", type("M", (), {"Distance": type("D", (), {"COSINE": "COSINE"})}))

    mgr = qdrant_manager.QdrantManager(collection_name="test")
    docs = mgr.query("hi", embed_fn=lambda text: [0.5])
    assert docs[0].page_content == "text"
