import logging
from pathlib import Path

from tasks.state_pull import pull_state_db_from_lakefs


def test_pull_state_db_from_lakefs_invokes_download(monkeypatch):
    calls = []

    def fake_download_state_db():
        calls.append("called")
        return True

    monkeypatch.setattr("tasks.state_pull.download_state_db", fake_download_state_db)
    monkeypatch.setattr("tasks.state_pull.get_run_logger", lambda: logging.getLogger("test_logger"))
    monkeypatch.setattr("tasks.state_pull.get_local_state_db_path", lambda: Path("state/test.db"))

    result = pull_state_db_from_lakefs.fn()

    assert calls == ["called"]
    assert result is True
