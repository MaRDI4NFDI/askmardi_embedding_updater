import logging

from helper import lakefs


def test_download_state_db_backs_up_existing(tmp_path, monkeypatch):
    local_path = tmp_path / "state.db"
    local_path.write_text("old-data")

    class FakeObjectsApi:
        def stat_object(self, repo, branch, path):
            return True

        def get_object(self, repository, ref, path, _preload_content=False):
            class FakeResponse:
                def read(self):
                    return b"new-data"

            return FakeResponse()

    class FakeLakefs:
        def __init__(self):
            self.objects_api = FakeObjectsApi()

    class FakeDatetime:
        @staticmethod
        def utcnow():
            class FakeNow:
                @staticmethod
                def strftime(fmt):
                    return "20240101T120000Z"

            return FakeNow()

    monkeypatch.setattr(lakefs, "get_lakefs_client", lambda: FakeLakefs())
    monkeypatch.setattr(
        lakefs,
        "cfg",
        lambda section: (
            {
                "state_repo": "repo",
                "branch": "main",
                "state_repo_directory": "",
                "state_db_filename_prefix": "askmardi_embedding_updater__state",
            }
            if section == "lakefs"
            else {"collection": "test_collection"}
            if section == "qdrant"
            else {}
        ),
    )
    monkeypatch.setattr(lakefs, "datetime", FakeDatetime)
    monkeypatch.setattr(lakefs, "get_run_logger", lambda: logging.getLogger("test"))

    monkeypatch.setattr(lakefs, "get_local_state_db_path", lambda: local_path)

    result = lakefs.download_state_db()

    backup_path = local_path.with_name("state.db.backup_20240101T120000Z")
    assert result is True
    assert backup_path.exists()
    assert backup_path.read_text() == "old-data"
    assert local_path.read_bytes() == b"new-data"
