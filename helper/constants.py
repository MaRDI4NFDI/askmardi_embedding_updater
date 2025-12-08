"""
Shared constants loaded from configuration.
"""

from pathlib import Path

from helper.config import cfg

# QID representing the software class in Wikibase.
SOFTWARE_PROFILE_QID: str = cfg("mardi_kg")["mardi_software_profile_qid"]
MARDI_PROFILE_TYPE_PID: str = cfg("mardi_kg")["mardi_profile_type_pid"]

# LakeFS state DB filename and default local path
STATE_DB_FILENAME: str = "askmardi_embedding_updater__state.db"
STATE_DB_PATH: Path = Path("state") / STATE_DB_FILENAME
