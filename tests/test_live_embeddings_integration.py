import os
import pytest

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from qdrant_client import QdrantClient

from helper.config import cfg
from helper_embedder.embedder_tools import EmbedderTools


@pytest.mark.integration
def test_live_rag_chain_with_qdrant():
    """
    Run a live RAG query against Qdrant if the service and API keys are available.
    Skips automatically when Qdrant or the generative model credentials are absent.
    """
    qdrant_cfg = cfg("qdrant")
    host = qdrant_cfg.get("host")
    port = qdrant_cfg.get("port", 6333)
    collection = qdrant_cfg.get("collection", "rag-collection")
    url = host if host and str(host).startswith("http") else None

    try:
        client = QdrantClient(url=url, host=None if url else host, port=port, api_key=qdrant_cfg.get("api_key"))
        _ = client.get_collections()
    except Exception:
        pytest.skip("Qdrant not reachable; skipping live RAG test")

    embedder = EmbedderTools(model_name=cfg("embedding").get("model_name", "sentence-transformers/all-MiniLM-L6-v2"))
    embeddings = embedder.embeddings

    def custom_retrieve(query: str):
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

    retriever_ranked = RunnableLambda(custom_retrieve)

    def format_retrieved_data(docs):
        return "\n\n".join([doc.page_content for doc in docs[:3]])

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

    from langchain_community.chat_models import ChatOllama

    model = ChatOllama(model="llama3:8b", temperature=0)

    rag_chain = (
        {
            "context": retriever_ranked | format_retrieved_data,
            "question": RunnablePassthrough(),
        }
        | prompt
        | model
        | StrOutputParser()
    )

    result = rag_chain.invoke("What does this collection document?")
    assert isinstance(result, str)
    assert len(result) > 0
