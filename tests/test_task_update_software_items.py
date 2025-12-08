# pytest -m "not integration"
# pytest -o log_cli=true -o log_cli_level=DEBUG

import logging
from typing import List

import pytest

from helper.config import cfg
from helper.constants import SOFTWARE_PROFILE_QID, MARDI_PROFILE_TYPE_PID
from tasks.init_db_task import _init_db, get_connection
from tasks.update_software_items import build_query, run_query, update_software_items


def test_update_software_items_with_mocks(tmp_path, monkeypatch):
    db_path = tmp_path / "state.db"
    _init_db(str(db_path))

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
            }
        raise KeyError(section)

    next_idx = {"start": 0}

    def fake_run_query(endpoint: str, query: str, logger, max_retries: int) -> List[str]:
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
    monkeypatch.setattr("tasks.update_software_items.SOFTWARE_PROFILE_QID", "QSOFT")
    monkeypatch.setattr("tasks.update_software_items.MARDI_PROFILE_TYPE_PID", "PPID")
    monkeypatch.setattr("tasks.update_software_items.time.sleep", lambda _: None)
    monkeypatch.setattr("tasks.update_software_items.get_run_logger", lambda: logging.getLogger("test_logger"))

    returned_qids = update_software_items.fn(str(db_path))

    assert returned_qids == ["Q1", "Q2", "Q3"]
    # Ensure the final query used the remaining total limit (1) instead of the per-query ceiling (2)
    assert any("LIMIT 1" in q for q in returned_queries)

    conn = get_connection(str(db_path))
    cur = conn.execute("SELECT qid FROM software_index ORDER BY qid")
    persisted = [row[0] for row in cur.fetchall()]
    conn.close()

    assert persisted == sorted(returned_qids)


@pytest.mark.integration
def test_run_query_against_real_endpoint():
    wb_cfg = cfg("mardi_kg")
    wf_cfg = cfg("sparql")

    endpoint = wb_cfg["sparql_endpoint"]
    limit = 5

    query = build_query(offset=0, limit=limit)
    logger = logging.getLogger("test_run_query")

    results = run_query(endpoint=endpoint, query=query, logger=logger, max_retries=wf_cfg["max_retries"])
    logger.debug(f"SPARQL results: {results}")

    assert len(results) <= limit
    assert all(qid.startswith("Q") for qid in results)
    assert SOFTWARE_PROFILE_QID.startswith("Q")
    assert MARDI_PROFILE_TYPE_PID.startswith("P")
