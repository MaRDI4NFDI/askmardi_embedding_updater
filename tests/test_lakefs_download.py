import logging
import tempfile
from pathlib import Path

import pytest

from helper.lakefs import download_file

@pytest.mark.integration
def test_download_file_real_endpoint(monkeypatch):
    """
    Download a real object from LakeFS via S3 gateway.

    This test is skipped if ../config.yaml is not present.
    """

    config_path = Path(__file__).resolve().parent.parent / "config.yaml"

    if not config_path.exists():
        pytest.skip("config.yaml not found; skipping LakeFS integration test")

    # Patch logging only
    monkeypatch.setattr(
        "helper.lakefs.get_run_logger",
        lambda: logging.getLogger("test"),
    )
    monkeypatch.setattr(
        "helper.config.get_run_logger",
        lambda: logging.getLogger("test"),
    )

    key = "main/01/27/95/Q12795/components/documentation.pdf"

    with tempfile.TemporaryDirectory() as tmpdir:
        dest = Path(tmpdir) / "download.bin"
        download_file(key=key, dest_path=str(dest))

        assert dest.exists()
        assert dest.stat().st_size > 0


# pytest -s -o log_cli=true --log-cli-level=DEBUG .\tests\test_lakefs_download.py