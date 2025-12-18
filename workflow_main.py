import logging
from pathlib import Path
from prefect import flow, get_run_logger
from prefect.context import get_run_context
from prefect.exceptions import MissingContextError

from helper import config as config_helper
from helper.config import CONFIG_PATH, check_for_config, get_local_state_db_path, \
    setup_prefect_logging
from helper.constants import DOCUMENT_TYPE_CRAN
from helper.logger import get_logger_safe
from helper.planner_tools import ensure_plan_exists
from tasks.state_pull import pull_state_db_from_lakefs
from tasks.init_db_task import init_db_task
from tasks.state_push import snapshot_table_counts
from tasks.update_software_items import update_software_item_index_from_mardi
from tasks.update_lakefs_file_index import update_file_index_from_lakefs
from tasks.update_embeddings import update_embeddings, get_software_items_with_pdf_component
from tasks.state_push import push_state_db_to_lakefs


setup_prefect_logging()

@flow(name="start_update_embedding_workflow")
def start_update_embedding_workflow(
        update_embeddings_loop_iterations: int = 2,
        update_embeddings_embeddings_per_loop: int = 10,
        timeout_seconds: int = 100,
        max_pages: int = 100,
        use_plan: str | None = None,
):
    """
    Orchestrate the end-to-end software documentation embedding sync flow.

    Args:
        update_embeddings_loop_iterations: Number of iterations to run the embedding loop.
        update_embeddings_embeddings_per_loop: Number of PDFs processed per iteration.
        timeout_seconds: Chunking/embedding timeout per PDF.
        max_pages: Maximum pages allowed per PDF before skipping.
        use_plan: Optional plan filename (e.g., "plan_localworker_01") looked up under
            the LakeFS planned/ prefix. If provided and not found, the workflow exits
            with an error.
    """
    logger = get_logger_safe()

    logger.info(f"Running with: iterations={update_embeddings_loop_iterations}, "
                f"per_loop={update_embeddings_embeddings_per_loop}")

    state_db_path: str = str(get_local_state_db_path())

    Path(state_db_path).parent.mkdir(parents=True, exist_ok=True)

    pulled = pull_state_db_from_lakefs()
    if not pulled:
        init_db_task()

    if use_plan is not None:
        ensure_plan_exists(use_plan, logger)

    baseline_counts = snapshot_table_counts()
    update_software_item_index_from_mardi()
    update_file_index_from_lakefs()

    for iteration in range(update_embeddings_loop_iterations):

        update_embeddings(
            max_number_of_pdfs=update_embeddings_embeddings_per_loop,
            document_type=DOCUMENT_TYPE_CRAN,
            timeout_seconds=timeout_seconds,
            max_pages=max_pages,
        )

        push_state_db_to_lakefs(baseline_counts=baseline_counts)

        completed = iteration + 1
        remaining = update_embeddings_loop_iterations - completed
        logger.info(
            "Completed iteration %s/%s; %s remaining",
            completed,
            update_embeddings_loop_iterations,
            remaining,
        )


if __name__ == "__main__":

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        logger.addHandler(logging.StreamHandler())

    # Check whether we are running in a Prefect environment
    try:
        get_run_context()
    except MissingContextError:
        config_helper.is_prefect_environment = False
        logger.info("Prefect environment not detected; running without Prefect context.")
    else:
        logger.info("Prefect environment detected; Prefect context available.")

    if check_for_config():
        start_update_embedding_workflow()
    else:
        raise SystemExit(1)



# https://github.com/shanojpillai/qdrant-rag-pro
