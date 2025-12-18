"""
Create per-worker embedding work plans from the current state database.

This script is a lightweight planner (no Prefect flow) that:
1) Pulls the authoritative state DB from LakeFS.
2) Computes pending (qid, component) items that still need embeddings.
3) Splits the backlog into work packages and writes plan_*.json files to ./temp.
4) Optionally uploads those plan files (and an index) to LakeFS when --golive is set.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

from helper.config import check_for_config, get_local_state_db_path, load_config
from helper.constants import DOCUMENT_TYPE_CRAN
from helper.lakefs import _file_md5, get_lakefs_client
from helper.logger import get_logger_safe
from tasks.init_db_task import get_connection
from tasks.state_pull import pull_state_db_from_lakefs
from tasks.state_push import push_state_db_to_lakefs


DEFAULT_PLAN_PREFIX = "planned/"
DEFAULT_TEMP_DIR = Path("temp")


@dataclass
class PlanEntry:
    """
    Single unit of work describing one component to embed.

    Attributes:
        qid: Wikibase identifier.
        component: LakeFS key for the document.
        document_type: Label such as CRAN or OTHER.
    """

    qid: str
    component: str
    document_type: str = DOCUMENT_TYPE_CRAN

    def to_dict(self) -> dict:
        """Return a JSON-serializable representation."""
        return {
            "qid": self.qid,
            "component": self.component,
            "document_type": self.document_type,
        }


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the planner script.

    Returns:
        argparse.Namespace: Parsed arguments including package size, count, and go-live flag.
    """
    parser = argparse.ArgumentParser(description="Create embedding work plans for workers.")
    parser.add_argument(
        "--package-size",
        type=int,
        default=10,
        help="Number of work items per plan file.",
    )
    parser.add_argument(
        "--packages",
        type=int,
        required=True,
        help="Number of plan files to create.",
    )
    parser.add_argument(
        "--golive",
        action="store_true",
        help="Upload plan files to LakeFS in addition to writing them locally.",
    )
    return parser.parse_args()


def ensure_state_db(logger: logging.Logger) -> Path:
    """
    Ensure a local copy of the state database exists and is current.

    The planner always pulls from LakeFS to get the latest snapshot. If no remote DB
    exists yet, the process terminates.

    Args:
        logger (logging.Logger): Logger for status messages.

    Returns:
        Path: Local filesystem path to the state DB.
    """
    state_path = get_local_state_db_path()
    pulled = pull_state_db_from_lakefs()
    if not pulled:
        logger.error(
            "State DB not found in LakeFS. Expected path (local copy would be): %s",
            state_path,
        )
        raise SystemExit(1)
    logger.info("State DB pulled from LakeFS: %s", state_path)
    return state_path


def find_pending_work() -> List[PlanEntry]:
    """
    Query the state DB for components that still need embeddings.

    Returns:
        list[PlanEntry]: Work items ordered as returned by the query.
    """
    conn = get_connection()
    cursor = conn.cursor()
    # We consider anything without a successful 'ok' status as pending.
    cursor.execute(
        """
        SELECT si.qid, ci.component
        FROM software_index si
        JOIN component_index ci ON ci.qid = si.qid
        LEFT JOIN embeddings_index ei
            ON ei.qid = ci.qid AND ei.component = ci.component
        WHERE ei.status IS NULL OR ei.status NOT IN ('ok', 'planned')
        ORDER BY si.qid ASC, ci.component ASC
        """
    )
    rows: Sequence[Tuple[str, str]] = cursor.fetchall()
    conn.close()
    return [PlanEntry(qid=row[0], component=row[1]) for row in rows]


def chunk(items: Iterable[PlanEntry], size: int) -> List[List[PlanEntry]]:
    """
    Split an iterable of work items into fixed-size chunks.

    Args:
        items: Work entries to split.
        size: Maximum chunk size.

    Returns:
        list[list[PlanEntry]]: Chunks preserving order.
    """
    batch: List[PlanEntry] = []
    batches: List[List[PlanEntry]] = []
    for item in items:
        batch.append(item)
        if len(batch) == size:
            batches.append(batch)
            batch = []
    if batch:
        batches.append(batch)
    return batches


def write_plan_files(
    batches: List[List[PlanEntry]],
    temp_dir: Path,
    worker_ids: List[str],
    db_checksum: str,
) -> List[Path]:
    """
    Write plan JSON files to the local temp directory.

    Args:
        batches: Grouped work items.
        temp_dir: Destination directory for plan files.
        worker_ids: Identifiers used in filenames.
        db_checksum: MD5 checksum of the state DB snapshot used for planning.

    Returns:
        list[Path]: Paths to the written plan files.
    """
    temp_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat()
    written: List[Path] = []
    for batch, worker_id in zip(batches, worker_ids):
        payload = {
            "planner_version": "1.0",
            "created_at": timestamp,
            "state_db_checksum": db_checksum,
            "package_id": f"plan_{worker_id}",
            "entries": [entry.to_dict() for entry in batch],
        }
        path = temp_dir / f"plan_{worker_id}.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        written.append(path)
    return written


def upload_plans_to_lakefs(local_paths: List[Path], logger: logging.Logger) -> None:
    """
    Upload generated plan files to LakeFS using the configured repo/branch.

    Args:
        local_paths: Plan and index files to upload.
        logger: Logger for status messages.
    """
    lakefs = get_lakefs_client()
    cfg = load_config()
    lakefs_cfg = cfg.get("lakefs", {})
    repo = lakefs_cfg["state_repo"]
    branch = lakefs_cfg["branch"]
    directory_prefix = lakefs_cfg.get("state_repo_directory", "").strip("/")

    plan_prefix = DEFAULT_PLAN_PREFIX
    for path in local_paths:
        filename = path.name
        remote_path = "/".join(filter(None, [directory_prefix, plan_prefix.strip("/"), filename]))
        logger.info("Uploading plan file %s -> %s:%s/%s", path, repo, branch, remote_path)
        with open(path, "rb") as fh:
            lakefs.objects_api.upload_object(
                repository=repo,
                branch=branch,
                path=remote_path,
                content=fh,
            )


def generate_worker_ids(count: int) -> List[str]:
    """
    Produce sequential worker IDs for the generated batches.

    Args:
        count: Number of required IDs.

    Returns:
        list[str]: Worker IDs aligned with batches using the pattern localworker_XX.
    """
    return [f"localworker_{i:02d}" for i in range(1, count + 1)]


def mark_as_planned(entries: List[PlanEntry], planned_at: datetime) -> None:
    """
    Reserve planned work in the state DB by setting status to 'planned'.

    Args:
        entries: Work items that are being planned.
        planned_at: Timestamp applied to the reservation.
    """
    if not entries:
        return

    conn = get_connection()
    cursor = conn.cursor()
    ts = planned_at.isoformat()
    cursor.executemany(
        """
        INSERT OR REPLACE INTO embeddings_index (qid, component, updated_at, status)
        VALUES (?, ?, ?, 'planned')
        """,
        [(e.qid, e.component, ts) for e in entries],
    )
    conn.commit()
    conn.close()


def main() -> None:
    """
    Run the planner: pull DB, build plans, write locally, optional upload.
    """
    if not check_for_config():
        raise SystemExit(1)

    args = parse_args()
    logger = get_logger_safe()
    logger.info("Planner starting with package_size=%s, packages=%s, golive=%s",
                args.package_size, args.packages, args.golive)

    state_path = ensure_state_db(logger)
    db_checksum = _file_md5(str(state_path))
    pending = find_pending_work()

    if not pending:
        logger.info("No pending work found; no plan files generated.")
        return

    batches = chunk(pending, args.package_size)[: args.packages]
    planned_entries = [entry for batch in batches for entry in batch]
    planned_at = datetime.now(timezone.utc)
    mark_as_planned(planned_entries, planned_at)

    if args.golive:
        push_state_db_to_lakefs.fn()
        logger.info("State DB uploaded to LakeFS after planning.")

    worker_ids = generate_worker_ids(len(batches))
    written_paths = write_plan_files(batches, DEFAULT_TEMP_DIR, worker_ids, db_checksum)
    logger.info("Wrote %d plan files to %s", len(written_paths), DEFAULT_TEMP_DIR)

    if args.golive:
        upload_plans_to_lakefs(written_paths, logger=logger)
        logger.info("Plan files uploaded to LakeFS under prefix '%s'", DEFAULT_PLAN_PREFIX)
    else:
        logger.info("Skipped LakeFS upload (run with --golive to publish plans).")


if __name__ == "__main__":
    main()
