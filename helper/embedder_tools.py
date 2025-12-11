import os
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings


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

    def split_and_filter(self, documents: List[Document], min_length: int = 250) -> List[Document]:
        """
        Create semantic chunks and discard segments shorter than the threshold.

        Args:
            documents: Source documents to split.
            min_length: Minimum allowed character length for a chunk.

        Returns:
            List[Document]: Filtered semantic chunks.
        """
        chunks = self.chunker.split_documents(documents)
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
