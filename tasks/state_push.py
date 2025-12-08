from prefect import task, get_run_logger

from helper.lakefs import upload_state_db, commit_state_db


@task(name="push_state_db_to_lakefs")
def push_state_db_to_lakefs(db_path: str) -> None:
    """
    Upload the local SQLite DB to LakeFS and create a commit.

    Args:
        db_path: Local path of the state DB file to persist.
    """
    logger = get_run_logger()
    logger.info(f"Uploading state DB to LakeFS: {db_path}")

    try:
        upload_state_db(local_path=db_path)
        commit_state_db(message="Update state DB for software docs workflow")
        logger.info("State DB successfully uploaded and committed to LakeFS.")
    except Exception as e:
        logger.error(f"Failed to save state DB to LakeFS: {e}")
        raise
