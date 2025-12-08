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

    fake_qids = ["Q1", "Q2", "Q3"]

    def fake_cfg(section: str) -> dict:
        if section == "wikibase":
            return {"sparql_endpoint": "http://example.test"}
        if section == "workflow":
            return {
                "sparql_page_size": 2,
                "sleep_between_queries": 0,
                "max_retries": 1,
            }
        raise KeyError(section)

    batches = [fake_qids, []]

    def fake_run_query(endpoint: str, query: str, logger, max_retries: int) -> List[str]:
        return batches.pop(0) if batches else []

    monkeypatch.setattr("tasks.update_software_items.cfg", fake_cfg)
    monkeypatch.setattr("tasks.update_software_items.run_query", fake_run_query)
    monkeypatch.setattr("tasks.update_software_items.SOFTWARE_PROFILE_QID", "QSOFT")
    monkeypatch.setattr("tasks.update_software_items.MARDI_PROFILE_TYPE_PID", "PPID")
    monkeypatch.setattr("tasks.update_software_items.time.sleep", lambda _: None)
    monkeypatch.setattr("tasks.update_software_items.get_run_logger", lambda: logging.getLogger("test_logger"))

    returned_qids = update_software_items.fn(str(db_path))

    assert returned_qids == fake_qids

    conn = get_connection(str(db_path))
    cur = conn.execute("SELECT qid FROM software_index ORDER BY qid")
    persisted = [row[0] for row in cur.fetchall()]
    conn.close()

    assert persisted == sorted(fake_qids)


@pytest.mark.integration
def test_run_query_against_real_endpoint():
    wb_cfg = cfg("wikibase")
    wf_cfg = cfg("workflow")

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
