import logging
import os
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from prefect import get_run_logger
from prefect.exceptions import MissingContextError


def _get_logger():
    try:
        return get_run_logger()
    except MissingContextError:
        return logging.getLogger(__name__)

class EmbedderTools:
    """Utilities for loading documents and generating semantic embeddings."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        """
        Initialize the embedding model and semantic chunker.

        Args:
            model_name: Hugging Face embedding model identifier.
        """
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name)
        self.chunker = SemanticChunker(embeddings=self.embeddings)
        self.embedding_dimension = len(self.embed_text("dimension probe"))

    def load_pdfs(self, folder_path: str) -> List[Document]:
        """
        Load all PDF files from a directory into LangChain documents.

        Args:
            folder_path: Directory containing PDF files.

        Returns:
            List[Document]: Documents with page content and metadata.
        """
        all_docs: List[Document] = []
        for filename in os.listdir(folder_path):
            if filename.endswith(".pdf"):
                loader = PyPDFLoader(os.path.join(folder_path, filename))
                all_docs.extend(loader.load())
        return all_docs

    def load_pdf_file(self, file_path: str) -> List[Document]:
        """
        Load a single PDF file into LangChain documents.

        Args:
            file_path: Absolute or relative path to the PDF file.

        Returns:
            List[Document]: Documents with page content and metadata.
        """
        loader = PyPDFLoader(file_path)
        return loader.load()

    def split_and_filter(
        self, documents: List[Document], min_length: int = 250, timeout_seconds: int = 300
    ) -> List[Document]:
        """
        Create semantic chunks with a timeout and discard short segments.

        Args:
            documents: Source documents to split.
            min_length: Minimum allowed character length for a chunk.
            timeout_seconds: Maximum time allowed for semantic chunking.

        Returns:
            List[Document]: Filtered semantic chunks.

        Raises:
            TimeoutError: If semantic chunking exceeds the allotted time.
        """
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self.chunker.split_documents, documents)
            try:
                chunks = future.result(timeout=timeout_seconds)
            except FuturesTimeoutError as exc:
                future.cancel()
                raise TimeoutError(
                    f"Semantic chunking timed out after {timeout_seconds}s"
                ) from exc
        return [chunk for chunk in chunks if len(chunk.page_content) > min_length]

    def embed_text(self, text: str) -> List[float]:
        """
        Generate an embedding vector for the provided text.

        Args:
            text: Text to embed.

        Returns:
            List[float]: Embedding vector.
        """
        return self.embeddings.embed_query(text)

    def format_documents(self, docs: List[Document], limit: int = 5) -> str:
        """
        Convert documents to a prompt-ready string.

        Args:
            docs: Documents to format.
            limit: Maximum number of documents to include.

        Returns:
            str: Concatenated page content separated by blank lines.
        """
        return "\n\n".join([doc.page_content for doc in docs[:limit]])
