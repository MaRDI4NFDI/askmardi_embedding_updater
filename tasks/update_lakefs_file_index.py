import re
from datetime import datetime, timezone
from typing import Iterable, List, Optional, Set

from prefect import task, get_run_logger

from helper.config import cfg
from helper.constants import STATE_DB_PATH, ALLOWED_EXTENSIONS
from helper.lakefs import get_lakefs_s3_client
from tasks.init_db_task import get_connection


_QID_PATTERN = re.compile(r"Q\d+$")


def _extract_qid_from_key(key: str) -> Optional[str]:
    """
    Extract the LAST QID-like segment (Q + digits) from the key path.

    Example:
        "main/01/29/Q12961/components/doc.pdf" -> "Q12961"
        "main/Q1/foo/Q200/bar/file.pdf"        -> "Q200"

    Args:
        key: S3/LakeFS object key.

    Returns:
        The most specific QID found or None.
    """
    qids = [part for part in key.split("/") if _QID_PATTERN.fullmatch(part)]
    return qids[-1] if qids else None


def _iter_repo_files(s3_client, bucket: str, prefix: str) -> Set[str]:
    """
    Recursively walk the LakeFS repository (via S3 gateway)
    and collect objects with allowed file extensions.

    Args:
        s3_client: boto3 S3 client
        bucket: LakeFS repository name
        prefix: Branch name prefix, e.g. "main/"

    Returns:
        Set of matching object keys
    """
    paginator = s3_client.get_paginator("list_objects_v2")
    files: Set[str] = set()
    normalized_exts = {
        ext if ext.startswith(".") else f".{ext}" for ext in ALLOWED_EXTENSIONS
    }

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if any(key.lower().endswith(ext) for ext in normalized_exts):
                files.add(key)

    return files


@task(name="update_lakefs_file_index")
def update_lakefs_file_index(db_path: str = str(STATE_DB_PATH)) -> None:
    """
    Index all files found in the LakeFS repository under the given branch.

    Requirements:
      • match any file extension in ALLOWED_EXTENSIONS
      • try to extract QID from the key
      • skip files without QID
      • store full S3 key in component_index table

    Args:
        db_path: path to the workflow state database
    """
    logger = get_run_logger()
    lakefs_cfg = cfg("lakefs")

    s3_client = get_lakefs_s3_client()
    repo = lakefs_cfg["data_repo"]
    branch = lakefs_cfg["branch"]
    prefix = f"{branch}/"

    logger.info(f"Scanning LakeFS repo='{repo}' with prefix='{prefix}' ...")
    files = _iter_repo_files(s3_client, bucket=repo, prefix=prefix)
    logger.info(f"Found {len(files):,} candidate files with allowed extensions")

    if not files:
        logger.info("No matching files found. Nothing to index.")
        return

    conn = get_connection(db_path)
    cursor = conn.cursor()
    timestamp = datetime.now(timezone.utc).isoformat()

    rows_to_write: List[tuple] = []
    skipped_no_qid = 0

    for key in sorted(files):
        qid = _extract_qid_from_key(key)
        if qid:
            rows_to_write.append((qid, key, None, timestamp))
        else:
            skipped_no_qid += 1

    if rows_to_write:
        cursor.executemany(
            """
            INSERT OR REPLACE INTO component_index
                (qid, component, checksum, updated_at)
            VALUES (?, ?, ?, ?)
            """,
            rows_to_write,
        )
        conn.commit()

    conn.close()

    logger.info(
        f"Indexed {len(rows_to_write):,} files. "
        f"Skipped {skipped_no_qid:,} files without identifiable QID."
    )
