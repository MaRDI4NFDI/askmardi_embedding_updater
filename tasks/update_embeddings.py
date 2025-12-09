import os
import tempfile
from datetime import datetime, timezone
from typing import List, Tuple

from prefect import task, get_run_logger

from helper.config import cfg
from helper.constants import STATE_DB_PATH
from helper.lakefs import get_lakefs_s3_client
from helper_embedder.embedder_tools import EmbedderTools
from helper_embedder.qdrant_manager import QdrantManager
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
    components: List[Tuple[str, str]], db_path: str
) -> int:
    """
    Build embeddings for component PDFs and push them to Qdrant.

    Args:
        components: Iterable of (qid, component) rows from the DB.
        db_path: Path to the workflow's SQLite state database.
    """
    logger = get_run_logger()
    conn = get_connection(db_path)
    cursor = conn.cursor()

    lakefs_cfg = cfg("lakefs")
    qdrant_cfg = cfg("qdrant")
    embedding_cfg = cfg("embedding")

    s3_client = get_lakefs_s3_client()
    bucket = lakefs_cfg["data_repo"]

    model_name = embedding_cfg.get("model_name", "sentence-transformers/all-MiniLM-L6-v2")
    embedder = EmbedderTools(model_name=model_name)

    qdrant_host = qdrant_cfg.get("host")
    qdrant_port = qdrant_cfg.get("port", 6333)
    qdrant_collection = qdrant_cfg.get("collection", "software_docs")
    qdrant_distance = qdrant_cfg.get("distance", "COSINE")
    qdrant_api_key = qdrant_cfg.get("api_key")
    qdrant_host_cfg = qdrant_cfg.get("host")
    qdrant_url = None
    qdrant_host = qdrant_host_cfg
    if qdrant_host_cfg and str(qdrant_host_cfg).startswith("http"):
        qdrant_url = qdrant_host_cfg.rstrip("/")
        qdrant_host = None

    qdrant_kwargs = {
        "host": qdrant_host,
        "port": qdrant_port,
        "url": qdrant_url,
        "api_key": qdrant_api_key,
        "collection_name": qdrant_collection,
        "distance": qdrant_distance,
    }
    qdrant_manager = QdrantManager(**qdrant_kwargs)
    qdrant_manager.ensure_collection(vector_size=embedder.embedding_dimension)

    processed = 0
    for qid, component in components:

        cursor.execute(
            "SELECT 1 FROM embeddings_index WHERE qid = ? LIMIT 1",
            (qid,),
        )
        if cursor.fetchone():
            logger.debug(f"Skipping {qid} — already embedded")
            continue

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
            tmp_path = tmp_file.name
            try:
                logger.info(f"Downloading PDF for {qid} from {component}")
                obj = s3_client.get_object(Bucket=bucket, Key=component)
                tmp_file.write(obj["Body"].read())
                tmp_file.flush()
            except Exception as exc:
                logger.warning(f"Failed to download {component} for {qid}: {exc}")
                continue

        try:
            documents = embedder.load_pdf_file(tmp_path)
            for doc in documents:
                doc.metadata.update({"qid": qid, "component": component})

            chunks = embedder.split_and_filter(documents)
            for chunk in chunks:
                chunk.metadata.update({"qid": qid, "component": component})

            if not chunks:
                logger.warning(f"No valid chunks produced for {qid} ({component})")
                continue

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
            logger.info(f"Embedded and indexed {qid} ({component})")

        except Exception as exc:
            logger.warning(f"Embedding failed for {qid} ({component}): {exc}")
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    conn.close()
    logger.info(f"PDF indexing finished; processed {processed} new items")
    return processed



@task(name="update_embeddings")
def update_embeddings(db_path: str = str(STATE_DB_PATH)) -> int:
    """
    Synchronize embeddings_index rows with the current component_index entries.

    Args:
        db_path: Path to the workflow's SQLite state database.

    Returns:
        int: Number of component records processed into embeddings_index.
    """
    logger = get_run_logger()

    # First get an overview
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
    logger.info(f"Found {len(components):,} component records to sync")

    if not components:
        conn.close()
        logger.info("No components to process; embeddings_index unchanged.")
        return 0

    conn.close()
    processed = perform_pdf_indexing(components=components, db_path=db_path)


    logger.info("Embeddings index update complete")

    return processed
