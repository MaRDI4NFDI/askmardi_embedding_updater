import logging
from pathlib import Path

from tasks.state_push import push_state_db_to_lakefs


def test_push_state_db_to_lakefs_uploads_and_commits(monkeypatch):
    uploads = []
    commits = []

    def fake_upload():
        uploads.append("called")

    def fake_commit(message: str):
        commits.append(message)

    monkeypatch.setattr("tasks.state_push.upload_state_db", fake_upload)
    monkeypatch.setattr("tasks.state_push.commit_state_db", fake_commit)
    monkeypatch.setattr("tasks.state_push.get_run_logger", lambda: logging.getLogger("test_logger"))
    monkeypatch.setattr("tasks.state_push.get_local_state_db_path", lambda: Path("state/test.db"))

    push_state_db_to_lakefs.fn()

    assert uploads == ["called"]
