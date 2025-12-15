from prefect import task, get_run_logger
from helper.config import cfg, get_local_state_db_path, get_state_db_filename
from helper.lakefs import download_state_db


@task(name="pull_state_db_from_lakefs")
def pull_state_db_from_lakefs() -> bool:
    """
    Restore the SQLite state database from LakeFS if it exists.

    Returns:
        bool: True if a DB was downloaded; False if none existed.
    """
    logger = get_run_logger()
    resolved_path = str(get_local_state_db_path())

    lakefs_cfg = cfg("lakefs")
    repo = lakefs_cfg["state_repo"]
    branch = lakefs_cfg["branch"]
    prefix = lakefs_cfg.get("state_repo_directory", "").strip("/")
    state_db_filename = get_state_db_filename()
    path_in_repo = f"{prefix}/{state_db_filename}" if prefix else state_db_filename
    remote_uri = f"lakefs://{repo}/{branch}/{path_in_repo}"

    logger.info(
        "Checking for existing state database at LakeFS: %s (local target: %s)",
        remote_uri,
        resolved_path,
    )

    exists = download_state_db()

    if exists:
        logger.info("Successfully restored state DB from LakeFS.")
    else:
        logger.info("No existing state DB in LakeFS (starting fresh).")

    return exists
