import uuid
from typing import Callable, List, Optional

from langchain_core.documents import Document
from prefect import get_run_logger
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import ResponseHandlingException


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

        # Ensure full-text index (once)
        self._ensure_text_index()

    def ensure_collection(self, vector_size: int) -> None:
        """
        Ensure that the managed Qdrant collection exists and is fully initialized.

        If the collection does not yet exist, it is created with the specified
        vector configuration. Regardless of whether the collection already exists,
        this method also ensures that required payload indexes (e.g. a full-text
        index for lexical search) are present.

        Args:
            vector_size: Dimensionality of the embedding vectors stored in the
                collection.

        Returns:
            None
        """
        if not self._collection_exists():
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=self.distance,
                    on_disk=True,
                ),
                hnsw_config=models.HnswConfigDiff(on_disk=True),
            )

        # Always ensure text index
        self._ensure_text_index()

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

    def _ensure_text_index(self) -> None:
        """
        Ensure that a full-text payload index exists for page_content.

        This operation is expensive and may time out while Qdrant builds
        the index in the background. Timeouts are treated as non-fatal.
        """
        logger = get_run_logger()

        info = self.client.get_collection(self.collection_name)
        payload_schema = info.payload_schema or {}

        payload_schema_avail = "page_content" in payload_schema
        logger.debug(f"Full-text index on page_content: {payload_schema_avail} ")

        if payload_schema_avail:
            return

        try:
            logger.warning(f"Triggering creation of full-text index on page_content ...")
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="page_content",
                field_schema="text",
            )
            logger.info("Triggered creation of full-text index on page_content")
        except ResponseHandlingException as exc:
            logger.warning(
                "Timed out while creating text index for page_content. "
                "Index creation may still be running in background: %s",
                exc,
            )
