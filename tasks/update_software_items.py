import time
import requests
from datetime import datetime, timezone
from typing import List, Optional

from prefect import task, get_run_logger

from tasks.init_db_task import get_connection
from helper.config import cfg
from helper.constants import SOFTWARE_PROFILE_QID
from helper.constants import MARDI_PROFILE_TYPE_PID


def build_query(offset: int, limit: Optional[int]) -> str:
    """
    Construct a SPARQL query to fetch software QIDs from Wikibase.

    Args:
        offset: Pagination offset for the query.
        limit: Maximum number of rows to fetch; None omits LIMIT.

    Returns:
        str: SPARQL query string.
    """
    limit_clause = f"\n    LIMIT {limit}" if limit is not None else ""

    return f"""
    PREFIX wd: <https://portal.mardi4nfdi.de/entity/>
    PREFIX wdt: <https://portal.mardi4nfdi.de/prop/direct/>

    SELECT ?qid
    WHERE {{
      ?item wdt:{MARDI_PROFILE_TYPE_PID} wd:{SOFTWARE_PROFILE_QID} .
      BIND(REPLACE(STR(?item), "^.*/", "") AS ?qid)
    }}
    ORDER BY ?qid
    {limit_clause}
    OFFSET {offset}
    """


def run_query(endpoint: str, query: str, logger, max_retries: int) -> List[str]:
    """
    Execute a SPARQL query with retries and parse QIDs.

    Args:
        endpoint: SPARQL endpoint URL.
        query: SPARQL query text to execute.
        logger: Prefect logger used for status updates.
        max_retries: Maximum retry attempts on failure.

    Returns:
        list[str]: Extracted QIDs from query results.

    Raises:
        RuntimeError: If retries are exhausted without success.
    """
    for attempt in range(1, max_retries + 1):
        try:
            logger.debug(f"Running SPARQL query: {query}")
            resp = requests.get(
                endpoint,
                params={"query": query, "format": "json"},
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()
            bindings = data.get("results", {}).get("bindings", [])
            return [b["qid"]["value"] for b in bindings]

        except Exception as e:
            logger.warning(
                f"[Retry {attempt}/{max_retries}] SPARQL error: {e} — retrying..."
            )
            time.sleep(2 * attempt)

    raise RuntimeError("SPARQL failed after maximum retries")


@task(name="update_software_items")
def update_software_items(db_path: str) -> List[str]:
    """
    Refresh the software_index table with QIDs retrieved from Wikibase.

    Args:
        db_path: Path to the workflow's SQLite state database.

    Returns:
        list[str]: All QIDs fetched during this run.
    """
    logger = get_run_logger()
    logger.info("Updating software_index table from Wikibase")

    wb_cfg = cfg("wikibase")
    wf_cfg = cfg("workflow")

    endpoint = wb_cfg["sparql_endpoint"]

    limit = wf_cfg["sparql_page_size"]
    sleep_between = wf_cfg["sleep_between_queries"]
    max_retries = wf_cfg["max_retries"]

    conn = get_connection(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM software_index")
    existing_rows = cursor.fetchone()[0]
    logger.info(f"Existing software_index rows: {existing_rows:,}")

    offset = existing_rows if existing_rows > 0 else 0

    total_fetched = existing_rows
    all_qids: List[str] = []
    timestamp = datetime.now(timezone.utc).isoformat()

    while True:
        logger.info(f"Querying SPARQL OFFSET={offset} LIMIT={limit}")

        query = build_query(offset=offset, limit=limit)
        batch = run_query(endpoint, query, logger, max_retries)

        if not batch:
            logger.info("Wikibase returned no more software results — complete")
            break

        cursor.executemany(
            """
            INSERT OR REPLACE INTO software_index
                (qid, updated_at)
            VALUES (?, ?)
            """,
            [(qid, timestamp) for qid in batch],
        )
        conn.commit()

        all_qids.extend(batch)
        total_fetched += len(batch)
        offset += limit if limit is not None else len(batch)

        logger.info(f"Fetched {len(batch):,} QIDs — Total: {total_fetched:,}")

        time.sleep(sleep_between)

    conn.close()
    logger.info("Software indexing done")

    return all_qids
