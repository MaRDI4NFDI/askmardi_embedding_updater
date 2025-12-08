from prefect import task, get_run_logger
from helper.constants import STATE_DB_PATH
from helper.lakefs import download_state_db


@task(name="pull_state_db_from_lakefs")
def pull_state_db_from_lakefs(db_path: str = str(STATE_DB_PATH)) -> bool:
    """
    Restore the SQLite state database from LakeFS if it exists.

    Args:
        db_path: Local path where the state DB should be stored.

    Returns:
        bool: True if a DB was downloaded; False if none existed.
    """
    logger = get_run_logger()
    logger.info(f"Checking for existing state database at LakeFS: {db_path}")

    exists = download_state_db(local_path=db_path)

    if exists:
        logger.info("Successfully restored state DB from LakeFS.")
    else:
        logger.info("No existing state DB in LakeFS (starting fresh).")

    return exists
