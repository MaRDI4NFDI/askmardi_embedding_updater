from functools import partial
from typing import Any, List

import pytest

from helper.config import cfg
from helper_embedder.embedder_tools import EmbedderTools
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from openai import OpenAI
from qdrant_client import QdrantClient


def _custom_retrieve(
    query: str,
    client: QdrantClient,
    embeddings: Any,
    collection: str,
) -> List[Document]:
    """Fetch top candidate documents from Qdrant for a query.

    Args:
        query: Natural language question to embed and search.
        client: Qdrant client instance to query against.
        embeddings: Embedding model that provides `embed_query`.
        collection: Qdrant collection name.

    Returns:
        List[Document]: Ranked documents returned by Qdrant.
    """
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
    """Join the top results into a context block.

    Args:
        docs: Retrieved documents sorted by similarity.

    Returns:
        str: Concatenated page content for downstream prompting.
    """
    return "\n\n".join([doc.page_content for doc in docs[:3]])


def _run_llm(prompt_text: str, llm_client: OpenAI, llm_model: str) -> str:
    """Query the configured OpenAI-compatible model for a response.

    Args:
        prompt_text: Fully rendered prompt text.
        llm_client: OpenAI client instance.
        llm_model: Model identifier to use for chat completions.

    Returns:
        str: Content returned by the chat completion.
    """
    completion = llm_client.chat.completions.create(
        model=llm_model,
        messages=[{"role": "user", "content": prompt_text}],
        temperature=0,
    )
    return completion.choices[0].message.content or ""


@pytest.mark.integration
def test_live_rag_chain_with_qdrant():
    """
    Run a live RAG query against Qdrant if the service and API keys are available.
    Skips automatically when Qdrant or the generative model credentials are absent.
    """
    qdrant_cfg = cfg("qdrant")
    url = qdrant_cfg.get("url", "http://localhost:6333")
    collection = qdrant_cfg.get("collection", "rag-collection")
    print(f"[Live RAG] Qdrant config: url={url}, collection={collection}")

    try:
        client = QdrantClient(
            url=url,
            api_key=qdrant_cfg.get("api_key"),
        )
        _ = client.get_collections()
    except Exception as e:
        print( e )
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
Do not use any prior knowledge or make assumptions. If the answer is not in the context, respond with "I don't know".
Keep the answer concise and use ten sentences.
Question: {question}
Context:
{context}
Answer:
""",
        input_variables=["context", "question"],
    )

    llm_client = OpenAI(
        base_url=llm_host,
        api_key=llm_api_key,
    )

    rag_chain = (
        {
            "context": retriever_ranked | _format_retrieved_data,
            "question": RunnablePassthrough(),
        }
        | prompt
        | RunnableLambda(partial(_run_llm, llm_client=llm_client, llm_model=llm_model))
    )

    result = rag_chain.invoke("What does this collection document?")
    assert isinstance(result, str)
    assert len(result) > 0
