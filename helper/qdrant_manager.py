import uuid
from typing import Callable, List, Optional

from langchain_core.documents import Document
from qdrant_client import QdrantClient, models


class QdrantManager:
    """Manage Qdrant collections and vector search operations."""

    def __init__(
        self,
        url: str = "http://localhost:6333",
        api_key: Optional[str] = None,
        collection_name: str = "rag-collection",
        distance: str = "COSINE",
    ) -> None:
        """
        Initialize the Qdrant client connection.

        Args:
            url: Full URL to Qdrant, including scheme and port.
            api_key: API key for authenticated Qdrant deployments.
            collection_name: Name of the target collection to manage.
            distance: Distance metric name (COSINE, EUCLID, DOT).
        """
        self.collection_name = collection_name
        distance_name = distance.upper()
        self.distance = getattr(models.Distance, distance_name, models.Distance.COSINE)

        client_kwargs = {"url": url}
        if api_key:
            client_kwargs["api_key"] = api_key

        self.client = QdrantClient(**client_kwargs)

    def is_available(self) -> bool:
        """
        Check whether the configured Qdrant service is reachable.

        Returns:
            bool: True if a basic API call succeeds, False otherwise.
        """
        try:
            self.client.get_collections()
            return True
        except Exception:
            return False

    def recreate_collection(self, vector_size: int) -> None:
        """
        Drop and recreate the managed collection for a new embedding size.

        Args:
            vector_size: Dimensionality of the embeddings to store.
        """
        if self._collection_exists():
            self.client.delete_collection(self.collection_name)

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=self.distance,
                on_disk=True,
            ),
            hnsw_config=models.HnswConfigDiff(on_disk=True),
        )

    def ensure_collection(self, vector_size: int) -> None:
        """
        Create the collection if it does not exist already.

        Args:
            vector_size: Dimensionality of the embeddings to store.
        """
        if self._collection_exists():
            return

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=self.distance,
                on_disk=True,
            ),
            hnsw_config=models.HnswConfigDiff(on_disk=True),
        )

    def embed_and_upload_documents(
        self,
        documents: List[Document],
        embed_fn: Callable[[object], List[float]],
        id_prefix: Optional[str] = None,
    ) -> None:
        """
        Embed and upload a batch of documents to the qdrant database / collection.

        Args:
            documents: Documents to embed and upload.
            embed_fn: Callable that returns an embedding vector for a Document.
            id_prefix: Optional prefix to keep point IDs unique across batches.
        """
        points = []
        for idx, doc in enumerate(documents):
            vector = embed_fn(doc)

            payload = {
                **doc.metadata,
                "page_content": doc.page_content,
            }

            # Use deterministic UUIDs when prefixed; otherwise fall back to integer IDs
            point_id = (
                str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{id_prefix}-{idx}"))
                if id_prefix
                else idx
            )

            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload,
                )
            )

        self.client.upload_points(
            collection_name=self.collection_name,
            points=points,
        )

    def query(
        self,
        query: str,
        embed_fn: Callable[[object], List[float]],
        limit: int = 30,
    ) -> List[Document]:
        """
        Retrieve the top similar documents for a query string.

        Args:
            query: Natural language query text.
            embed_fn: Callable that returns an embedding vector for the query.
            limit: Maximum number of documents to return.

        Returns:
            List[Document]: Ranked documents returned from Qdrant.
        """
        query_embedding = embed_fn(query)

        result = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            limit=limit,
            with_vectors=False,
        )

        documents: List[Document] = []
        for point in result.points:
            page_content = point.payload.get("page_content", "")

            metadata = {
                k: v for k, v in point.payload.items()
                if k != "page_content"
            }

            documents.append(
                Document(
                    page_content=page_content,
                    metadata=metadata,
                )
            )

        return documents

    def collection_size(self) -> int:
        """
        Fetch the current point count for the managed collection.

        Returns:
            int: Number of points stored in the collection.
        """
        collection_info = self.client.get_collection(
            collection_name=self.collection_name
        )
        return collection_info.points_count

    def _collection_exists(self) -> bool:
        """
        Check if the managed collection already exists.

        Returns:
            bool: True if the collection exists, False otherwise.
        """
        collections = self.client.get_collections()
        collection_names = {collection.name for collection in collections.collections}
        return self.collection_name in collection_names
