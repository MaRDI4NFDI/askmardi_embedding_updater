from lakefs_client import ApiException
from prefect import task, get_run_logger

from helper.constants import STATE_DB_PATH
from helper.lakefs import upload_state_db, commit_state_db


@task(name="push_state_db_to_lakefs")
def push_state_db_to_lakefs(db_path: str = str(STATE_DB_PATH)) -> None:
    """
    Upload the local SQLite DB to LakeFS and create a commit when changes occurred.

    Args:
        db_path: Local path of the state DB file to persist.
    """
    logger = get_run_logger()
    logger.info(f"[push_state_db] Uploading state DB: {db_path}")

    try:
        changed = upload_state_db(local_path=db_path)

        # If no changes, skip commit and return success
        if not changed:
            logger.info("[push_state_db] No changes detected — skipping commit.")
            return

        try:
            commit_state_db(message="Updated state database for askmardi_embedding_updater")
            logger.info("[push_state_db] State DB committed successfully.")

        except ApiException as e:
            body = getattr(e, "body", "")
            if "no changes" in str(body).lower():
                logger.info("[push_state_db] Nothing to commit — already up-to-date")
            else:
                raise     # Re-raise real errors

    except Exception as e:
        logger.error(f"[push_state_db] Failed saving state DB: {e}")
        raise

