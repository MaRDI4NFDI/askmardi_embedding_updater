import logging
from pathlib import Path

from tasks.state_pull import pull_state_db_from_lakefs


def test_pull_state_db_from_lakefs_invokes_download(monkeypatch):
    """Verify pull_state_db_from_lakefs triggers a download and returns True."""
    calls = []

    def fake_download_state_db():
        """Record invocation and simulate success."""
        calls.append("called")
        return True

    # Patch dependencies used inside pull_state_db_from_lakefs
    monkeypatch.setattr(
        "tasks.state_pull.download_state_db",
        fake_download_state_db,
    )
    monkeypatch.setattr(
        "tasks.state_pull.get_run_logger",
        lambda: logging.getLogger("test_logger"),
    )
    monkeypatch.setattr(
        "tasks.state_pull.get_local_state_db_path",
        lambda: Path("state/test.db"),
    )

    # Patch cfg to avoid reading config.yaml
    monkeypatch.setattr(
        "tasks.state_pull.cfg",
        lambda section: {
            "state_repo": "dummy-repo",
            "branch": "main",
            "state_repo_directory": "",
        },
    )

    # Patch get_state_db_filename to stop config access
    monkeypatch.setattr(
        "tasks.state_pull.get_state_db_filename",
        lambda: "test.db",
    )

    result = pull_state_db_from_lakefs.fn()

    assert calls == ["called"]
    assert result is True
