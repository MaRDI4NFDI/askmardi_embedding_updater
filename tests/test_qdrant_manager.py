from langchain_core.documents import Document

from helper import qdrant_manager


class DummyCollectionInfo:
    def __init__(self, points_count=0):
        self.points_count = points_count


class DummyCollections:
    def __init__(self, names):
        self.collections = [type("C", (), {"name": n}) for n in names]


class DummyClient:
    def __init__(self):
        self.created = False
        self.deleted = False
        self.uploaded = []
        self.queries = []
        self.collections = DummyCollections([])
        self.collection_info = DummyCollectionInfo()

    def get_collections(self):
        return self.collections

    def delete_collection(self, name):
        self.deleted = True

    def create_collection(self, **kwargs):
        self.created = True
        self.collections = DummyCollections([kwargs["collection_name"]])

    def upload_points(self, collection_name, points):
        self.uploaded.append((collection_name, points))

    def query_points(self, **kwargs):
        self.queries.append(kwargs)
        point = type(
            "Point",
            (),
            {"payload": {"page_content": "text", "meta": "m"}},
        )
        return type("Result", (), {"points": [point]})

    def get_collection(self, collection_name):
        return self.collection_info


def test_qdrant_manager_ensure_and_upload(monkeypatch):
    client = DummyClient()
    monkeypatch.setattr(
        qdrant_manager,
        "QdrantClient",
        lambda **_: client,
    )
    # Lightweight stand-in for qdrant models
    class FakeModels:
        class Distance:
            COSINE = "COSINE"

        class VectorParams:
            def __init__(self, size, distance, on_disk):
                self.size = size
                self.distance = distance
                self.on_disk = on_disk

        class HnswConfigDiff:
            def __init__(self, on_disk):
                self.on_disk = on_disk

        class PointStruct:
            def __init__(self, id, vector, payload):
                self.id = id
                self.vector = vector
                self.payload = payload

    monkeypatch.setattr(qdrant_manager, "models", FakeModels)

    mgr = qdrant_manager.QdrantManager(collection_name="test")
    mgr.ensure_collection(vector_size=10)
    assert client.created is True

    doc = Document(page_content="hello", metadata={})
    mgr.upload_documents([doc], embed_fn=lambda d: [1.0], id_prefix="Q1")
    assert client.uploaded
    collection, points = client.uploaded[0]
    assert collection == "test"
    assert points[0].payload["page_content"] == "hello"
    assert points[0].id == "cd323cb5-8784-58c5-85a9-4419450f4724"


def test_qdrant_manager_query(monkeypatch):
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
