from functools import partial
from typing import Any, List

import pytest

from helper.config import cfg
from helper_embedder.embedder_tools import EmbedderTools
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from openai import OpenAI
from qdrant_client import QdrantClient


def _llm_health_check(llm_client: OpenAI) -> None:
    """Perform a minimal chat completion to verify LLM availability."""
    completion = llm_client.chat.completions.create(
        model="llama3.2:latest",
        messages=[{"role": "user", "content": "Test"}],
        temperature=0,
    )
    if not completion or not getattr(completion, "choices", None):
        raise AssertionError("LLM health check returned no choices")
    content = completion.choices[0].message.content
    if not content:
        raise AssertionError("LLM health check returned empty content")
    # print(f"[_llm_health_check] Completion: {completion}")


def _custom_retrieve(
    query: str,
    client: QdrantClient,
    embeddings: Any,
    collection: str,
) -> List[Document]:
    """Fetch top candidate documents from Qdrant for a query."""
    query_embedding = embeddings.embed_query(query)
    res = client.query_points(
        collection_name=collection,
        with_vectors=True,
        query=query_embedding,
        limit=5,
    )
    docs = []
    for point in res.points:
        content = point.payload.get("page_content", "")
        metadata = point.payload
        docs.append(Document(page_content=content, metadata=metadata))
    return docs


def _format_retrieved_data(docs: List[Document]) -> str:
    """Join the top results into a context block."""
    return "\n\n".join([doc.page_content for doc in docs[:3]])


@pytest.mark.integration
def test_live_rag_chain_with_qdrant():
    """
    Run a live RAG query against Qdrant if everything is reachable.
    Automatically skipped when services or credentials are missing.
    """
    qdrant_cfg = cfg("qdrant")
    url = qdrant_cfg.get("url", "http://localhost:6333")
    collection = qdrant_cfg.get("collection", "rag-collection")
    print(f"[Live RAG] Qdrant config: url={url}, collection={collection}")

    try:
        client = QdrantClient(url=url, api_key=qdrant_cfg.get("api_key"))
        _ = client.get_collections()
    except Exception:
        pytest.skip("Qdrant not reachable; skipping live RAG test")

    llm_cfg = cfg("llm")
    llm_host = llm_cfg.get("host")
    llm_model = llm_cfg.get("model_name")
    llm_api_key = llm_cfg.get("api_key")

    print(
        "[Live RAG] LLM config: "
        f"base_url={llm_host}, model={llm_model}, api_key_present={bool(llm_api_key)}"
    )

    if not llm_host or not llm_api_key or not llm_model:
        pytest.skip("LLM configuration incomplete; skipping live RAG test")

    # Set up embedder
    embedder = EmbedderTools(
        model_name=cfg("embedding").get(
            "model_name",
            "sentence-transformers/all-MiniLM-L6-v2",
        )
    )
    embeddings = embedder.embeddings

    retriever_ranked = RunnableLambda(
        partial(
            _custom_retrieve,
            client=client,
            embeddings=embeddings,
            collection=collection,
        )
    )

    prompt = PromptTemplate(
        template="""You are an assistant for question-answering tasks. Use only the provided context below to answer the question.
    Do not use any prior knowledge or make assumptions.

    If the answer is not in the context:
    - Respond with exactly: "I don't know."
    - Do not write anything else.

    If the answer is in the context:
    - Keep the answer concise.
    - Use at most ten sentences.

    Question: {question}
    Context:
    {context}
    Answer:
    """,
        input_variables=["context", "question"],
    )

    # Main LLM for chain execution
    llm = ChatOpenAI(
        base_url=llm_host,
        api_key=llm_api_key,
        model_name=llm_model,
        temperature=0,
    )

    # Validate LLM availability independently
    client_health = OpenAI(base_url=llm_host, api_key=llm_api_key)
    _llm_health_check(client_health)

    # RAG Chain
    rag_chain = (
        {
            "context": retriever_ranked | _format_retrieved_data,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    result = rag_chain.invoke("What could be the mathematical concept of this?")
    print(f"[Live RAG] Result: {result}")
    assert isinstance(result, str)
    assert len(result) > 0


@pytest.mark.integration
def test_llm_health_check_runs():
    llm_cfg = cfg("llm")
    llm_host = llm_cfg.get("host")
    llm_api_key = llm_cfg.get("api_key")

    if not llm_host or not llm_api_key:
        pytest.skip("LLM configuration incomplete; skipping health check test")

    client = OpenAI(
        base_url=llm_host,
        api_key=llm_api_key,
    )
    _llm_health_check(client)
