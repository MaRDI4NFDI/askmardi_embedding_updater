"""
Shared constants loaded from configuration.
"""

from pathlib import Path
from typing import Optional

# These values are placeholders and will be populated in ``load_config`` once
# config.yaml is read.
SOFTWARE_PROFILE_QID: Optional[str] = None
MARDI_PROFILE_TYPE_PID: Optional[str] = None

# Config for state database
STATE_DB_FILENAME: str = "askmardi_embedding_updater__state.db"
STATE_DB_PATH: Path = Path("state") / STATE_DB_FILENAME

# Which files should be looked for when doing a full lakeFS scan
ALLOWED_EXTENSIONS = {"pdf", "html", "txt"}
