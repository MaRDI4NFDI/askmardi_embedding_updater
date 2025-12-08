import logging

from tasks.state_push import push_state_db_to_lakefs


def test_push_state_db_to_lakefs_uploads_and_commits(monkeypatch):
    uploads = []
    commits = []

    def fake_upload(local_path: str):
        uploads.append(local_path)

    def fake_commit(message: str):
        commits.append(message)

    monkeypatch.setattr("tasks.state_push.upload_state_db", fake_upload)
    monkeypatch.setattr("tasks.state_push.commit_state_db", fake_commit)
    monkeypatch.setattr("tasks.state_push.get_run_logger", lambda: logging.getLogger("test_logger"))

    push_state_db_to_lakefs.fn("/tmp/fake.db")

    assert uploads == ["/tmp/fake.db"]
