from pathlib import Path
from typing import List, Tuple, Optional

from lakefs_client import Configuration
from lakefs_client.client import LakeFSClient

from helper.config import cfg


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


# ============================================================
# Persistence of State DB
# ============================================================

def download_state_db(local_path: str):
    """
    Download the state SQLite DB from LakeFS if present.

    Args:
        local_path: Destination path for the downloaded DB file.

    Returns:
        bool: True when the DB exists and is downloaded; False otherwise.
    """
    lakefs = get_lakefs_client()
    lakefs_cfg = cfg("lakefs")

    repo = lakefs_cfg["state_repo"]
    branch = lakefs_cfg["branch"]
    path_in_repo = "software_docs_state.db"

    try:
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        obj = lakefs.objects_api.get_object(
            repository=repo,
            ref=branch,
            path=path_in_repo,
        )
        with open(local_path, "wb") as fh:
            fh.write(obj.data)
        return True
    except Exception:
        return False


def upload_state_db(local_path: str):
    """
    Upload the local state SQLite DB to LakeFS.

    Args:
        local_path: Path to the local DB file to upload.
    """
    lakefs = get_lakefs_client()
    lakefs_cfg = cfg("lakefs")

    repo = lakefs_cfg["state_repo"]
    branch = lakefs_cfg["branch"]
    path_in_repo = "software_docs_state.db"

    with open(local_path, "rb") as fh:
        lakefs.objects_api.upload_object(
            repository=repo,
            branch=branch,
            path=path_in_repo,
            content=fh,
        )


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
