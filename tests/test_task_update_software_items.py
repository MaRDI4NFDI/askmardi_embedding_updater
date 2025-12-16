# pytest -m "not integration"
# pytest -o log_cli=true -o log_cli_level=DEBUG

import logging
from pathlib import Path
from typing import List

import pytest

from helper.config import cfg
from tasks.init_db_task import _init_db, get_connection
from tasks.update_software_items import (
    build_query,
    run_query,
    update_software_item_index_from_mardi,
)


def test_update_software_items_with_mocks(tmp_path, monkeypatch):
    db_path = tmp_path / "state.db"
    monkeypatch.setattr("tasks.init_db_task.get_local_state_db_path", lambda: db_path)
    monkeypatch.setattr("helper.config.get_local_state_db_path", lambda: db_path)
    _init_db()

    all_qids_pool = ["Q1", "Q2", "Q3", "Q4"]
    returned_queries = []

    def fake_cfg(section: str) -> dict:
        if section == "mardi_kg":
            return {"sparql_endpoint": "http://example.test"}
        if section == "sparql":
            return {
                "sparql_max_results": 3,
                "sparql_max_results_per_query": 2,
                "sleep_between_queries": 0,
                "max_retries": 1,
                "timeout": 120,
            }
        raise KeyError(section)

    next_idx = {"start": 0}

    def fake_run_query(endpoint: str, query: str, logger, max_retries: int, timeout: int) -> List[str]:
        returned_queries.append(query)

        limit_val = None
        for line in query.splitlines():
            if line.strip().startswith("LIMIT"):
                try:
                    limit_val = int(line.split()[1])
                except Exception:
                    limit_val = None
                break

        start = next_idx["start"]
        if limit_val is None:
            batch = all_qids_pool[start:]
        else:
            batch = all_qids_pool[start : start + limit_val]
        next_idx["start"] += len(batch)
        return batch

    monkeypatch.setattr("tasks.update_software_items.cfg", fake_cfg)
    monkeypatch.setattr("tasks.update_software_items.run_query", fake_run_query)
    # Ensure globals are present for backward compatibility in the module
    mod = __import__("tasks.update_software_items")
    setattr(mod, "SOFTWARE_PROFILE_QID", "QSOFT")
    setattr(mod, "MARDI_PROFILE_TYPE_PID", "PPID")
    monkeypatch.setattr("tasks.update_software_items.time.sleep", lambda _: None)
    monkeypatch.setattr("tasks.update_software_items.get_run_logger", lambda: logging.getLogger("test_logger"))

    returned_qids = update_software_item_index_from_mardi.fn()

    assert returned_qids == ["Q1", "Q2", "Q3"]
    # Ensure the final query used the remaining total limit (1) instead of the per-query ceiling (2)
    assert any("LIMIT 1" in q for q in returned_queries)

    conn = get_connection()
    cur = conn.execute("SELECT qid FROM software_index ORDER BY qid")
    persisted = [row[0] for row in cur.fetchall()]
    conn.close()

    assert persisted == sorted(returned_qids)


@pytest.mark.integration
def test_run_query_against_real_endpoint(monkeypatch):

    config_path = Path(__file__).resolve().parent.parent / "config.yaml"

    if not config_path.exists():
        pytest.skip("config.yaml not found; skipping LakeFS integration test")

    monkeypatch.setattr("helper.lakefs.get_run_logger", lambda: logging.getLogger("test"))
    monkeypatch.setattr("helper.config.get_run_logger", lambda: logging.getLogger("test"))

    wb_cfg = cfg("mardi_kg")
    wf_cfg = cfg("sparql")

    endpoint = wb_cfg.get("sparql_endpoint")
    if not endpoint:
        pytest.skip("SPARQL endpoint not configured; skipping integration test")

    limit = 5

    query = build_query(offset=0, limit=limit)
    logger = logging.getLogger("test_run_query")

    results = run_query(
        endpoint=endpoint,
        query=query,
        logger=logger,
        max_retries=wf_cfg.get("max_retries", 1),
        timeout=wf_cfg.get("timeout", 120),
    )
    logger.debug(f"SPARQL results: {results}")

    assert len(results) <= limit
    assert all(qid.startswith("Q") for qid in results)
    assert wb_cfg.get("mardi_software_profile_qid", "").startswith("Q")
    assert wb_cfg.get("mardi_profile_type_pid", "").startswith("P")
