import os
import tempfile
from datetime import datetime, timezone
from typing import List, Tuple

from prefect import task, get_run_logger

from helper.config import cfg
from helper.constants import STATE_DB_PATH
from helper.lakefs import download_file
from helper.embedder_tools import EmbedderTools
from helper.qdrant_manager import QdrantManager
from tasks.init_db_task import get_connection


@task(name="log_qids_with_components")
def get_software_items_with_pdf_component(
    db_path: str = str(STATE_DB_PATH),
) -> int:
    """
    Log the overlap between software_index and component_index.

    Prints the first 5 QIDs that exist in component_index among those listed
    in software_index, and returns the total overlap count.

    Args:
        db_path: Path to the workflow's SQLite state database.
    """
    logger = get_run_logger()
    conn = get_connection(db_path)
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


def perform_pdf_indexing(
    components: List[Tuple[str, str]],
    db_path: str,
    qdrant_manager: QdrantManager | None = None,
    max_number_of_pdfs: int | None = None,
) -> int:
    """
    Build embeddings for component PDFs and push them to Qdrant.

    Args:
        components: Iterable of (qid, component) rows from the DB.
        db_path: Path to the workflow's SQLite state database.
        qdrant_manager: Qdrant client wrapper to persist embeddings. If None, a new instance is created from config.
        max_number_of_pdfs: Max number of documents to process

    Raises:
        RuntimeError: When three download timeouts occur within a single run.
    """
    logger = get_run_logger()
    conn = get_connection(db_path)
    cursor = conn.cursor()

    embedding_cfg = cfg("embedding")

    model_name = embedding_cfg.get("model_name", "sentence-transformers/all-MiniLM-L6-v2")
    embedder = EmbedderTools(model_name=model_name)

    logger.debug("Getting qdrant connection...")

    if qdrant_manager is None:
        qdrant_cfg = cfg("qdrant")
        qdrant_manager = QdrantManager(
            url=qdrant_cfg.get("url", "http://localhost:6333"),
            api_key=qdrant_cfg.get("api_key"),
            collection_name=qdrant_cfg.get("collection", "software_docs"),
            distance=qdrant_cfg.get("distance", "COSINE"),
        )

    logger.debug(f"Checking for collection ...")
    qdrant_manager.ensure_collection(vector_size=embedder.embedding_dimension)

    processed = 0
    timeout_failures = 0

    try:
        for qid, component in components:

            cursor.execute(
                "SELECT 1 FROM embeddings_index WHERE qid = ? AND component = ? LIMIT 1",
                (qid, component),
            )
            if cursor.fetchone():
                # logger.debug(f"Skipping {qid} — already embedded")
                continue

            logger.info(f"Downloading and Embedding PDF {processed+1}/{max_number_of_pdfs}  for QID: {qid} ...")

            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
                tmp_path = tmp_file.name
                try:
                    logger.debug(f"Downloading PDF for {qid} from {component}")
                    download_file(key=component, dest_path=tmp_path)
                except Exception as exc:
                    logger.warning(f"Failed to download {component} for {qid}: {exc}")
                    if isinstance(exc, TimeoutError):
                        timeout_failures += 1
                        if timeout_failures >= 3:
                            raise RuntimeError(
                                f"Aborting embeddings after {timeout_failures} download timeouts"
                            ) from exc
                    continue

            try:
                timeout = 60

                logger.info(f"Embedding PDF with a timeout of {timeout}s...")

                documents = embedder.load_pdf_file(tmp_path)
                for doc in documents:
                    doc.metadata.update({
                        "qid": qid,
                        "component": component,
                        "source": "CRAN"
                    })

                # Run the embedder with a timeout of 60 seconds
                chunks = embedder.split_and_filter(documents=documents, timeout_seconds=timeout)
                for chunk in chunks:
                    chunk.metadata.update({"qid": qid, "component": component})

                if not chunks:
                    logger.warning(f"No valid chunks produced for {qid} ({component})")
                    continue

                logger.debug("Writing embeddings to Qdrant db ...")

                qdrant_manager.upload_documents(
                    documents=chunks,
                    embed_fn=embedder.embed_text,
                    id_prefix=qid,
                )

                timestamp = datetime.now(timezone.utc).isoformat()
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO embeddings_index
                        (qid, component, updated_at)
                    VALUES (?, ?, ?)
                    """,
                    (qid, component, timestamp),
                )
                conn.commit()
                processed += 1
                logger.debug(f"Embedded and indexed {qid} ({component})")

                if max_number_of_pdfs is not None and processed >= max_number_of_pdfs:
                    logger.info(f"Reached max_number_of_pdfs={max_number_of_pdfs}; stopping early")
                    break

            except Exception as exc:
                logger.warning(f"Embedding failed for {qid} ({component}): {exc}")
            finally:
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
    finally:
        conn.close()

    logger.info(f"PDF indexing finished; processed {processed} new items")
    return processed



@task(name="update_embeddings")
def update_embeddings(db_path: str = str(STATE_DB_PATH), max_number_of_pdfs: int | None = None) -> int:
    """
    Synchronize embeddings_index rows with the current component_index entries.

    Args:
        db_path: Path to the workflow's SQLite state database.

    Returns:
        int: Number of component records processed into embeddings_index.
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
        logger.warning(f"Could not read Qdrant collection size: {exc}")

    get_software_items_with_pdf_component.fn(db_path=db_path)

    logger.info("Updating embeddings_index from component_index")

    conn = get_connection(db_path)
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

    logger.info( f"Starting downloading & embedding of {max_number_of_pdfs} PDFs." )

    processed = perform_pdf_indexing(
        components=components,
        db_path=db_path,
        qdrant_manager=qdrant_manager,
        max_number_of_pdfs=max_number_of_pdfs,
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
