import hashlib
from pathlib import Path
from typing import List, Tuple, Optional
from datetime import datetime

import boto3
from botocore.config import Config
from lakefs_client import Configuration, ApiException
from lakefs_client.client import LakeFSClient
from prefect import get_run_logger

from helper.config import cfg
from helper.constants import STATE_DB_FILENAME, STATE_DB_PATH
from helper.sharding import shard_qid


def get_lakefs_client() -> LakeFSClient:
    """
    Build a configured LakeFS client from `config.yaml` settings.

    Returns:
        LakeFSClient: Authenticated client instance.

    Raises:
        RuntimeError: If required LakeFS credentials are missing.
    """
    lakefs_cfg = cfg("lakefs")

    if not all(k in lakefs_cfg for k in ["url", "user", "password"]):
        raise RuntimeError("Missing LakeFS credentials in config.yaml")

    config = Configuration()
    config.host = lakefs_cfg["url"].rstrip("/") + "/api/v1"
    config.username = lakefs_cfg["user"]
    config.password = lakefs_cfg["password"]

    return LakeFSClient(configuration=config)


def get_lakefs_s3_client():
    """
    Create a boto3 client for the LakeFS S3 gateway.

    Returns:
        boto3.client: Configured S3-compatible client for LakeFS.
    """
    lakefs_cfg = cfg("lakefs")
    endpoint = f"{lakefs_cfg['url'].rstrip('/')}"

    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=lakefs_cfg["user"],
        aws_secret_access_key=lakefs_cfg["password"],
        region_name="us-east-1",
        config=Config(s3={"addressing_style": "path"}),
    )


# ============================================================
# Persistence of State DB
# ============================================================

def download_state_db(local_path: str = str(STATE_DB_PATH)):
    """
    Download the state SQLite DB from LakeFS if present.
    If a local file already exists, it is renamed to `{name}.backup_<timestamp>`.

    Args:
        local_path: Destination path for the downloaded DB file.

    Returns:
        bool: True when the DB exists and is downloaded; False otherwise.
    """
    logger = get_run_logger()

    lakefs = get_lakefs_client()
    lakefs_cfg = cfg("lakefs")

    repo = lakefs_cfg["state_repo"]
    branch = lakefs_cfg["branch"]
    prefix = lakefs_cfg.get("state_repo_directory", "").strip("/")
    path_in_repo = f"{prefix}/{STATE_DB_FILENAME}" if prefix else STATE_DB_FILENAME

    logger.info(f"[download_state_db] Downloading state DB from {repo}:{branch}/{path_in_repo}")

    try:
        lakefs.objects_api.stat_object(repo, branch, path_in_repo)
    except Exception:
        logger.info(f"[download_state_db] No existing state DB at LakeFS (starting fresh).")
        return False

    try:
        local_path_obj = Path(local_path)
        local_path_obj.parent.mkdir(parents=True, exist_ok=True)

        if local_path_obj.exists():
            timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
            backup_path = local_path_obj.with_name(f"{local_path_obj.name}.backup_{timestamp}")
            local_path_obj.rename(backup_path)
            logger.info(f"[download_state_db] Existing local DB backed up to {backup_path}")

        obj = lakefs.objects_api.get_object(
            repository=repo,
            ref=branch,
            path=path_in_repo,
            _preload_content=False,
        )
        with open(local_path_obj, "wb") as fh:
            fh.write(obj.read())

        logger.info(f"[download_state_db] Successfully downloaded to {local_path_obj}")
        return True

    except Exception as e:
        logger.error(f"[download_state_db] Failed to download state DB: {e}")
        return False


def _file_md5(path: str) -> str:
    """Compute MD5 checksum of a file."""
    md5 = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            md5.update(chunk)
    return md5.hexdigest()


def upload_state_db(local_path: str = str(STATE_DB_PATH)) -> bool:
    """
    Upload the local state SQLite DB to LakeFS only if changed.

    Returns:
        bool: True if upload succeeded or up-to-date; False otherwise.
    """
    logger = get_run_logger()
    lakefs = get_lakefs_client()
    lakefs_cfg = cfg("lakefs")

    repo = lakefs_cfg["state_repo"]
    branch = lakefs_cfg["branch"]
    prefix = lakefs_cfg.get("state_repo_directory", "").strip("/")
    path_in_repo = f"{prefix}/{STATE_DB_FILENAME}" if prefix else STATE_DB_FILENAME

    logger.info(f"[upload_state_db] Checking for DB changes at {repo}:{branch}/{path_in_repo}")

    local_checksum = _file_md5(local_path)
    logger.info(f"[upload_state_db] Local checksum: {local_checksum}")

    # Check remote checksum
    remote_checksum = None
    try:
        meta = lakefs.objects_api.stat_object(repo, branch, path_in_repo)
        remote_checksum = meta.checksum
        logger.info(f"[upload_state_db] Remote checksum: {remote_checksum}")
    except ApiException:
        logger.info("[upload_state_db] No remote object found (upload required)")

    # If unchanged, skip upload
    if remote_checksum == local_checksum:
        logger.info("[upload_state_db] No changes detected — skipping upload")
        return True

    logger.info(f"[upload_state_db] Uploading DB to LakeFS: {repo}:{branch}/{path_in_repo}")

    try:
        with open(local_path, "rb") as fh:
            lakefs.objects_api.upload_object(
                repository=repo,
                branch=branch,
                path=path_in_repo,
                content=fh,
            )
        logger.info("[upload_state_db] Upload finished successfully")
        return True

    except ApiException as e:
        body = getattr(e, "body", "")
        if "no changes" in str(body).lower():
            logger.info("[upload_state_db] No effective changes (already up-to-date)")
            return True

        logger.error(f"[upload_state_db] LakeFS error: {e.status} {e.reason}")
        logger.debug(body)
        return False

    except Exception as e:
        logger.error(f"[upload_state_db] Unexpected error: {e}")
        return False

def commit_state_db(message: str):
    """
    Commit the current state DB change set to LakeFS.

    Args:
        message: Commit message describing the update.
    """
    lakefs = get_lakefs_client()
    lakefs_cfg = cfg("lakefs")

    repo = lakefs_cfg["state_repo"]
    branch = lakefs_cfg["branch"]

    lakefs.commits_api.commit(
        repository=repo,
        branch=branch,
        commit_creation={"message": message},
    )


# ============================================================
# Component Listing for QIDs
# ============================================================

def list_components(qid: str) -> List[Tuple[str, Optional[str]]]:
    """
    List component filenames stored in LakeFS for a given QID.

    Args:
        qid: Wikibase QID whose components should be listed.

    Returns:
        list[tuple[str, Optional[str]]]: Pairs of component name and checksum
            (checksum currently placeholder None).
    """
    lakefs = get_lakefs_client()
    lakefs_cfg = cfg("lakefs")

    repo = lakefs_cfg["data_repo"]
    branch = lakefs_cfg["branch"]

    prefix = f"{shard_qid(qid)}/components/"
    results = []

    try:
        listing = lakefs.objects_api.list_objects(
            repository=repo,
            ref=branch,
            prefix=prefix,
        )
        for obj in listing.results:
            filename = obj.path.split("/")[-1]
            results.append((filename, None))
    except Exception:
        pass

    return results
