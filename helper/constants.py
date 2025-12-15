"""
Shared static constants.
"""

# Which files should be looked for when doing a full lakeFS scan
ALLOWED_EXTENSIONS = {"pdf", "html", "txt"}

# Document type values for metadata tagging
DOCUMENT_TYPE_CRAN = "CRAN_PACKAGE"
DOCUMENT_TYPE_MARDI = "MARDI_KG_ITEM"
DOCUMENT_TYPE_PUBLICATION = "PUBLICATION"
DOCUMENT_TYPE_OTHER = "OTHER"
