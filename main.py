from pathlib import Path
from prefect import flow

from tasks.state_pull import pull_state_db_from_lakefs
from tasks.init_db_task import init_db_task
from tasks.update_software_items import update_software_items
from tasks.update_lakefs_file_index import update_lakefs_file_index
from tasks.update_embeddings import update_embeddings
from tasks.state_push import push_state_db_to_lakefs


@flow(name="software-doc-embedding-sync")
def software_doc_embedding_sync(
    db_path: str = str(Path("state/software_docs_state.db")),
):
    """
    Orchestrate the end-to-end software documentation embedding sync flow.

    Args:
        db_path: Filesystem path to the SQLite state database.
    """

    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    pulled = pull_state_db_from_lakefs(db_path=db_path)
    if not pulled:
        init_db_task(db_path=db_path)
    #qids = update_software_items(db_path=db_path)
    #update_lakefs_file_index(qids=qids, db_path=db_path)
    #update_embeddings(db_path=db_path)
    push_state_db_to_lakefs(db_path=db_path)


if __name__ == "__main__":
    software_doc_embedding_sync()
