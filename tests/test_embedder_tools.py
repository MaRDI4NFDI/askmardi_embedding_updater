from pathlib import Path

from langchain_core.documents import Document

from helper_embedder import embedder_tools


def test_embedder_tools_embed_text_returns_vector(monkeypatch):
    """Embedding returns list of floats."""
    monkeypatch.setattr(
        embedder_tools,
        "HuggingFaceEmbeddings",
        lambda model_name=None: type(
            "FakeEmbeddings", (), {"embed_query": lambda _, text: [float(len(text))]}
        )(),
    )
    monkeypatch.setattr(
        embedder_tools,
        "SemanticChunker",
        lambda embeddings=None: type(
            "FakeChunker", (), {"split_documents": lambda _, docs: docs}
        )(),
    )
    tool = embedder_tools.EmbedderTools(model_name="unused")

    vec = tool.embed_text("hello")
    assert vec == [5.0]


def test_embedder_tools_split_and_filter(monkeypatch):
    """Chunks shorter than min_length are filtered out."""
    monkeypatch.setattr(
        embedder_tools,
        "HuggingFaceEmbeddings",
        lambda model_name=None: type(
            "FakeEmbeddings", (), {"embed_query": lambda _, text: [float(len(text))]}
        )(),
    )
    monkeypatch.setattr(
        embedder_tools,
        "SemanticChunker",
        lambda embeddings=None: type(
            "FakeChunker", (), {"split_documents": lambda _, docs: docs}
        )(),
    )
    tool = embedder_tools.EmbedderTools(model_name="unused")

    # Mock chunker to return known documents
    short = Document(page_content="short", metadata={})
    long = Document(page_content="l" * 300, metadata={})

    chunks = tool.split_and_filter([short, long], min_length=250)
    assert chunks == [long]
