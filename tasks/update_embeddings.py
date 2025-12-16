import os
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from logging import Logger
from typing import Dict, List, Tuple

from prefect import get_run_logger, task

from helper.config import cfg
from helper.constants import DOCUMENT_TYPE_CRAN, DOCUMENT_TYPE_OTHER
from helper.embedder_tools import EmbedderTools
from helper.lakefs import download_file
from helper.qdrant_manager import QdrantManager
from tasks.init_db_task import get_connection

# Prevent embedding backends from oversubscribing CPU cores when running in parallel.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


"""
High-level call order (condensed):

update_embeddings   [Prefect task, entry point]
 ├─ Setup QdrantManager / EmbedderTools
 ├─ get_software_items_with_pdf_component.fn()
 ├─ DB query for items to process
 ├─ embed_and_upload_all_PDFs
 │   ├─ DB pre-filter: embeddings_index
 │   ├─ START (2) PARALLEL THREADS:
 │   │   └─ download_and_embed_and_upload_one_PDF   (parallel)
 │   │       ├─ download_file()           (LakeFS → temp file)
 │   │       ├─ _get_embedder()            (thread-local)
 │   │       ├─ load_pdf_file()            (PDF → pages)
 │   │       ├─ split_and_filter()         (semantic chunking)
 │   │       ├─ qdrant_lock
 │   │       │   └─ embed_and_upload_documents()
 │   │       ├─ INSERT embeddings_index
 │   │       └─ cleanup (tmp file, DB close)
 │   └─ collect futures (as_completed)
 └─ return processed
"""

@task(name="update_embeddings")
def update_embeddings(
    max_number_of_pdfs: int | None = None,
    document_type: str = DOCUMENT_TYPE_OTHER,
) -> int:
    """
    MAIN ENTRY POINT:
      * Do configuration
      * Call: get_software_items_with_pdf_component.fn()
      * Call: embed_and_upload_all_PDFs()

    Args:
        max_number_of_pdfs: Optional cap on documents to embed during this run.
        document_type: Label for the document type to store in metadata.

    Returns:
        int: Number of new embedding records created.
    """
    logger = get_run_logger()
    qdrant_cfg = cfg("qdrant")
    qdrant_manager = QdrantManager(
        url=qdrant_cfg.get("url", "http://localhost:6333"),
        api_key=qdrant_cfg.get("api_key"),
        collection_name=qdrant_cfg.get("collection", "software_docs"),
        distance=qdrant_cfg.get("distance", "COSINE"),
    )

    url = qdrant_cfg.get("url", "http://localhost:6333")

    if not qdrant_manager.is_available():
        logger.error(f"Qdrant server is unreachable @: {url}")
        raise RuntimeError(f"Qdrant server is unreachable @: {url}")

    # Prepare embedder once to share embedding dimension across setup and indexing.
    embedding_cfg = cfg("embedding")
    model_name = embedding_cfg.get("model_name", "sentence-transformers/all-MiniLM-L6-v2")
    embedder = EmbedderTools(model_name=model_name, model_kwargs={"device": "cpu"})

    # First get an overview
    current_points = None
    try:
        current_points = qdrant_manager.collection_size()
        logger.info(
            "Qdrant collection '%s' currently holds %s vectors.",
            qdrant_cfg.get("collection", "software_docs"),
            f"{current_points:,}",
        )
    except Exception as exc:
        message = str(exc)
        if "doesn't exist" in message or "Not found" in message or "not exist" in message:
            logger.warning("Qdrant collection missing; creating it now.")
            qdrant_manager.ensure_collection(vector_size=embedder.embedding_dimension)
            try:
                current_points = qdrant_manager.collection_size()
            except Exception:
                current_points = None
        else:
            logger.warning(f"Could not read Qdrant collection size: {exc}")

    get_software_items_with_pdf_component.fn()

    logger.info("Updating embeddings_index from component_index")

    conn = get_connection()
    cursor = conn.cursor()

    # Get components that have a QIDs in software_index and
    # a matching entry in the component_index (=files in lakeFS).
    cursor.execute(
        """
        SELECT si.qid, ci.component
        FROM software_index si
        JOIN component_index ci ON ci.qid = si.qid
        """
    )
    components = cursor.fetchall()
    total_components = len(components)

    cursor.execute(
        "SELECT COUNT(*) FROM embeddings_index"
    )
    already_embedded = cursor.fetchone()[0]
    remaining = max(total_components - already_embedded, 0)

    logger.info(
        f"Found {total_components:,} component records; {remaining:,} pending embeddings"
    )

    if not components:
        conn.close()
        logger.info("No components to process; embeddings_index unchanged.")
        return 0

    conn.close()

    logger.info( f"Starting downloading & embedding of {max_number_of_pdfs} PDFs of type {document_type}." )

    # Call to main embedding logic
    processed = embed_and_upload_all_PDFs(
        components=components,
        qdrant_manager=qdrant_manager,
        max_number_of_pdfs=max_number_of_pdfs,
        document_type=document_type,
        embedder=embedder,
    )

    # Compute changes
    current_points_after_update = None
    try:
        current_points_after_update = qdrant_manager.collection_size()
    except Exception as exc:
        logger.warning(f"Could not read Qdrant collection size: {exc}")

    if current_points is not None and current_points_after_update is not None:
        new_points = current_points_after_update - current_points
        logger.info(f"Embeddings index update complete. Added {new_points:,} vectors.")
    else:
        logger.info("Embeddings index update complete. Vector count unavailable.")

    return processed

@task(name="log_qids_with_components")
def get_software_items_with_pdf_component() -> int:
    """
    Log the overlap between software_index and component_index.

    Prints the first 5 QIDs that exist in component_index among those listed
    in software_index, and returns the total overlap count.
    """
    logger = get_run_logger()
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT si.qid
        FROM software_index si
        JOIN component_index ci ON ci.qid = si.qid
        GROUP BY si.qid
        """
    )
    matching = [row[0] for row in cursor.fetchall()]
    conn.close()

    total = len(matching)
    sample = matching[:5]
    logger.info(f"QIDs with components ({total} total). Sample: {sample}")
    return total


def _get_embedder(
    thread_local: threading.local,
    model_name: str,
    embedder_init_kwargs: Dict,
) -> EmbedderTools:
    """
    Lazily create a per-thread embedder instance.

    Args:
        thread_local: Thread-local namespace to hold the embedder.
        model_name: Model identifier for the embedder.
        embedder_init_kwargs: Keyword args forwarded to the embedder constructor.

    Returns:
        EmbedderTools: Thread-local embedder instance.
    """
    cached = getattr(thread_local, "embedder", None)
    if cached is None:
        thread_local.embedder = EmbedderTools(model_name=model_name, **embedder_init_kwargs)
        cached = thread_local.embedder
    return cached


def download_and_embed_and_upload_one_PDF(
    qid: str,
    component: str,
    document_type: str,
    model_name: str,
    embedder_init_kwargs: Dict,
    qdrant_manager: QdrantManager,
    qdrant_lock: threading.Lock,
    thread_local: threading.local,
    logger: Logger,
    timeout_seconds: int =100,
    max_pages: int = 75,
) -> int:
    """
    Process a single (PDF) file (component) - connected to a qid.
    Main steps: download, chunking, embedding, uploading to qdrant.

    Args:
        qid: Wikibase QID linked to the document.
        component: LakeFS object key for the PDF.
        document_type: Label stored alongside the embedding metadata.
        model_name: Model identifier for the embedder.
        embedder_init_kwargs: Keyword args forwarded to the embedder constructor.
        qdrant_manager: Client wrapper used to upload embeddings.
        qdrant_lock: Mutex guarding Qdrant writes.
        thread_local: Thread-local namespace to hold the embedder instance.
        logger: Prefect/Python logger.
        timeout_seconds: Timeout in seconds for chunking.
        max_pages: If the PDF has more pages than this, it will be skipped (with timeout failure).

    Returns:
        int: 1 when a DB row is written (success or handled failure); 0 otherwise.
    """
    local_conn = get_connection()
    local_cursor = local_conn.cursor()

    # Check whether this component has already been processed
    local_cursor.execute(
        "SELECT 1 FROM embeddings_index WHERE qid = ? AND component = ? LIMIT 1",
        (qid, component),
    )
    if local_cursor.fetchone():
        local_conn.close()
        return 0

    logger.debug(f"Downloading and Embedding PDF for QID: {qid} ...")

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
            tmp_path = tmp_file.name
            logger.debug(f"Downloading PDF for {qid} from {component}")
            download_file(key=component, dest_path=tmp_path)
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Failed to download {component} for {qid}: {exc}")
        local_conn.close()
        return 0

    try:
        # Get thread-specific embedder instance
        local_embedder = _get_embedder(thread_local, model_name, embedder_init_kwargs)

        # Load PDF - documents is a list of the PDF pages
        documents = local_embedder.load_pdf_file(tmp_path)

        # Get PDF title if available
        first_title = documents[0].metadata.get("title") if documents else None

        # Assuming, PDF is a CRAN documentation, try to extract package name and version
        package, version = _extract_package_version(component, title=first_title)

        # Creating base metadata for all chunks
        # TODO: Change "source" to be based on actual type. For now, we assume the source is CRAN
        base_metadata = {
            "qid": qid,
            "source": "CRAN",
            "package": package,
            "version": version,
            "document_type": document_type,
            "page": None,
        }

        # Update the page number in the metadata for each page
        for page in documents:
            page_number = page.metadata.get("page")
            page.metadata.update({**base_metadata, "page": page_number})

        total_docs = len(documents)
        total_chars = sum(len(doc.page_content) for doc in documents)
        if total_docs > max_pages:
            raise TimeoutError(f"PDF (QID: {qid}) too large: {total_docs} pages exceeds limit of {max_pages}")
        logger.info(
            "Starting semantic chunking for %s: %d pages, total_chars=%d, timeout=%ss",
            qid,
            total_docs,
            total_chars,
            timeout_seconds,
        )

        chunk_elapsed = None
        try:
            chunk_start = datetime.now(timezone.utc)

            # Start chunking
            chunks = local_embedder.split_and_filter(documents, timeout_seconds=timeout_seconds)

            # Calculate elapsed time
            elapsed = (datetime.now(timezone.utc) - chunk_start).total_seconds()
            chunk_elapsed = elapsed
        except TimeoutError as exc:
            logger.warning(f"Chunking timed out for {qid} ({component}): {exc}")
            timestamp = datetime.now(timezone.utc).isoformat()
            local_cursor.execute(
                """
                INSERT OR REPLACE INTO embeddings_index
                    (qid, component, updated_at, status)
                VALUES (?, ?, ?, ?)
                """,
                (qid, component, timestamp, "failed - TimeoutError"),
            )
            local_conn.commit()
            return 1

        # Check if any chunks were produced
        if not chunks:
            logger.warning(f"No valid chunks produced for {qid} ({component})")
            return 0

        # Updating page number and chunk index in metadata
        for idx, chunk in enumerate(chunks):
            page_number = chunk.metadata.get("page")
            chunk.metadata.update({**base_metadata, "page": page_number, "chunk_index": idx})

        # Embed chunks and upload chunks to qdrant database
        logger.debug(f"Writing embeddings for {qid} to Qdrant db ...")
        with qdrant_lock:
            qdrant_manager.embed_and_upload_documents(
                documents=chunks,
                embed_fn=local_embedder.embed_document,
                id_prefix=qid,
            )

        # Update entry in embeddings_index
        timestamp = datetime.now(timezone.utc).isoformat()
        local_cursor.execute(
            """
            INSERT OR REPLACE INTO embeddings_index
                (qid, component, updated_at, status)
            VALUES (?, ?, ?, ?)
            """,
            (qid, component, timestamp, "ok"),
        )
        local_conn.commit()


        logger.info(
            "Semantic chunking completed for %s (%s) in %.2fs producing %d chunks",
            qid,
            component,
            chunk_elapsed,
            len(chunks),
        )
        return 1
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Embedding failed for {qid} ({component}): {exc}")
        return 0
    finally:
        try:
            if tmp_path:
                os.remove(tmp_path)
        except OSError:
            pass
        local_conn.close()


def embed_and_upload_all_PDFs(
    components: List[Tuple[str, str]],
    qdrant_manager: QdrantManager | None = None,
    max_number_of_pdfs: int | None = None,
    document_type: str = DOCUMENT_TYPE_OTHER,
    embedder: EmbedderTools | None = None,
) -> int:
    """
    Build embeddings for all given PDFs (components) for the current loop (e.g. list of 50 PDFs)
    and push all embeddings to Qdrant.

    Args:
        components: Iterable of (qid, component) rows from the DB.
        qdrant_manager: Qdrant client wrapper to persist embeddings. If None, a new instance is created from config.
        max_number_of_pdfs: Max number of documents to process
        document_type: Label for the document type to store in metadata.
        embedder: Optional shared EmbedderTools instance; if None, a new one is created.

    Returns:
        int: Count of embedding rows written (including handled failures).
    """
    logger = get_run_logger()

    # Create embedder
    embedding_cfg = cfg("embedding")
    model_name = embedding_cfg.get("model_name", "sentence-transformers/all-MiniLM-L6-v2")
    embedder_init_kwargs = {"model_kwargs": {"device": "cpu"}}
    base_embedder = embedder or EmbedderTools(model_name=model_name, **embedder_init_kwargs)

    # Initialize qdrant manager object
    if qdrant_manager is None:
        qdrant_cfg = cfg("qdrant")
        qdrant_manager = QdrantManager(
            url=qdrant_cfg.get("url", "http://localhost:6333"),
            api_key=qdrant_cfg.get("api_key"),
            collection_name=qdrant_cfg.get("collection", "software_docs"),
            distance=qdrant_cfg.get("distance", "COSINE"),
        )

    qdrant_manager.ensure_collection(vector_size=base_embedder.embedding_dimension)

    # Pre-filter to skip already embedded components and respect max_number_of_pdfs.
    # Components are the files from lakeFS connected to a QID item, e.g. CRAN documentation or paper PDFs/HTML files
    conn = get_connection()
    cursor = conn.cursor()
    components_to_process: List[Tuple[str, str]] = []
    for qid, component in components:
        cursor.execute(
            "SELECT 1 FROM embeddings_index WHERE qid = ? AND component = ? LIMIT 1",
            (qid, component),
        )
        if cursor.fetchone():
            continue
        components_to_process.append((qid, component))
        if max_number_of_pdfs is not None and len(components_to_process) >= max_number_of_pdfs:
            break
    conn.close()

    if not components_to_process:
        logger.info("No new PDFs require embedding; skipping indexing step.")
        return 0

    # Initialize locks for thread critical sections
    qdrant_lock = threading.Lock()
    thread_local = threading.local()

    processed = 0
    max_workers = min(2, len(components_to_process))

    # Start processing components in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_component = {
            executor.submit(
                download_and_embed_and_upload_one_PDF,
                qid,
                component,
                document_type,
                model_name,
                embedder_init_kwargs,
                qdrant_manager,
                qdrant_lock,
                thread_local,
                logger,
            ): (qid, component)
            for qid, component in components_to_process
        }

        # Collect results
        for future in as_completed(future_to_component):
            try:
                # Each finished thread returns 1 when a DB row is written (success or failure)
                processed += future.result()
            except Exception as exc:  # noqa: BLE001
                qid, component = future_to_component[future]
                logger.warning(f"Embedding failed for {qid} ({component}) in worker: {exc}")

    logger.info(f"PDF indexing finished; processed {processed} new items")
    return processed


def _extract_package_version(component: str, title: str | None = None) -> Tuple[str, str]:
    """
    Derive CRAN package name and version from the component filename.

    Args:
        component: LakeFS object key for the PDF.
        title: Optional PDF title metadata to parse for package name.

    Returns:
        Tuple[str, str]: (package, version) with fallbacks when parsing fails.
    """
    filename = os.path.basename(component)
    stem, _ = os.path.splitext(filename)
    # CRAN docs are typically named like `{package}_{version}.pdf`. Only accept that pattern;
    # otherwise fall back to unknown rather than guessing from the path.
    if "_" in stem:
        candidate_pkg, candidate_ver = stem.rsplit("_", 1)
        if candidate_pkg and candidate_ver and candidate_ver[0].isdigit():
            return candidate_pkg, candidate_ver
    # Attempt to extract package name from PDF title if available: e.g., "pkgname: Description"
    if title and ":" in title:
        possible_pkg = title.split(":", 1)[0].strip()
        if possible_pkg:
            return possible_pkg, "unknown"
    return "unknown", "unknown"