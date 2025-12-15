import pytest
from langchain_core.documents import Document

from helper import embedder_tools


class SlowChunker:
    def split_documents(self, docs):
        import time

        time.sleep(1)
        return docs


def test_split_and_filter_times_out(monkeypatch):
    """
    split_and_filter should raise TimeoutError when chunking exceeds the timeout.
    """
    # Mock embeddings to a lightweight stub
    monkeypatch.setattr(
        embedder_tools,
        "HuggingFaceEmbeddings",
        lambda model_name=None: type("FakeEmbeddings", (), {"embed_query": lambda _, text: [len(text)]})(),
    )
    # Inject a slow chunker
    monkeypatch.setattr(
        embedder_tools,
        "SemanticChunker",
        lambda embeddings=None, **kwargs: SlowChunker(),
    )

    tool = embedder_tools.EmbedderTools(model_name="unused")
    docs = [Document(page_content="test", metadata={})]

    with pytest.raises(TimeoutError):
        tool.split_and_filter(docs, timeout_seconds=0.1)
