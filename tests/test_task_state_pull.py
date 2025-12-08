import logging

from helper.constants import STATE_DB_PATH
from tasks.state_pull import pull_state_db_from_lakefs


def test_pull_state_db_from_lakefs_invokes_download(monkeypatch):
    calls = []

    def fake_download_state_db(local_path: str):
        calls.append(local_path)
        return True

    monkeypatch.setattr("tasks.state_pull.download_state_db", fake_download_state_db)
    monkeypatch.setattr("tasks.state_pull.get_run_logger", lambda: logging.getLogger("test_logger"))

    result = pull_state_db_from_lakefs.fn()

    assert calls == [str(STATE_DB_PATH)]
    assert result is True
