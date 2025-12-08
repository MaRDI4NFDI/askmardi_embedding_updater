"""
Shared constants loaded from configuration.
"""

from helper.config import cfg

# QID representing the software class in Wikibase.
SOFTWARE_PROFILE_QID: str = cfg("wikibase")["mardi_software_profile_qid"]
MARDI_PROFILE_TYPE_PID: str = cfg("wikibase")["mardi_profile_type_pid"]