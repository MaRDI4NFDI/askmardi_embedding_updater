from pathlib import Path
from prefect import flow

from helper.constants import STATE_DB_PATH
from tasks.state_pull import pull_state_db_from_lakefs
from tasks.init_db_task import init_db_task
from tasks.update_software_items import update_software_items_from_mardi
from tasks.update_lakefs_file_index import update_lakefs_file_index
from tasks.update_embeddings import update_embeddings, get_software_items_with_pdf_component
from tasks.state_push import push_state_db_to_lakefs


@flow(name="software-doc-embedding-sync")
def software_doc_embedding_sync():
    """
    Orchestrate the end-to-end software documentation embedding sync flow.
    """
    db_path: str = str(STATE_DB_PATH)

    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    pulled = pull_state_db_from_lakefs()
    if not pulled:
        init_db_task()
    update_software_items_from_mardi()
    update_lakefs_file_index()
    update_embeddings()
    push_state_db_to_lakefs()


if __name__ == "__main__":
    software_doc_embedding_sync()
