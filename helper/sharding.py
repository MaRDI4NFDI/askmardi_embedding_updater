def shard_qid(qid: str) -> str:
    """
    Convert a QID into a sharded path format.

    Args:
        qid: Wikibase QID (e.g., "Q12345").

    Returns:
        str: Sharded layout like `NN/NN/NN/Q12345`.
    """
    qid = qid.strip().upper()
    digits = qid[1:].zfill(6)
    return f"{digits[:2]}/{digits[2:4]}/{digits[4:6]}/{qid}"
