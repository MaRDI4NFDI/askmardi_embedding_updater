"""
Shared constants loaded from configuration.
"""

from helper.config import cfg

# QID representing the software class in Wikibase.
SOFTWARE_PROFILE_QID: str = cfg("mardi_kg")["mardi_software_profile_qid"]
MARDI_PROFILE_TYPE_PID: str = cfg("mardi_kg")["mardi_profile_type_pid"]

# LakeFS state DB filename
STATE_DB_FILENAME: str = "askmardi_embedding_updater__state.db"
