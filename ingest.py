"""
ingest.py
----------
Data ingestion module for the RAG-Powered Search Engine.

Pulls research paper metadata + abstracts from the public arXiv API
(no API key required) and writes them to data/corpus.jsonl.

Design note: this is intentionally written the way you'd structure a
GCP Cloud Function trigger (fetch -> validate -> normalize -> persist),
so it can be lifted almost as-is into a Cloud Function / Cloud Run job
that lands data in GCS or BigQuery instead of a local JSONL file.
"""

import argparse
import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("ingest")

ARXIV_API_URL = "http://export.arxiv.org/api/query"
NS = {"atom": "http://www.w3.org/2005/Atom"}


def fetch_arxiv(
    query: str,
    max_results: int = 50,
    batch_size: int = 25,
    sort_by: str = "relevance",
) -> list[dict]:
    """Fetch paper records from the arXiv API in batches (paginated).

    The query is wrapped in double quotes so arXiv treats it as an exact
    phrase (e.g. "retrieval augmented generation"). Without quotes, arXiv's
    search parser ORs the individual words together, which - especially
    combined with a date sort - can return largely off-topic papers that
    just happen to contain one common word from the query.
    """
    records = []
    start = 0
    while start < max_results:
        n = min(batch_size, max_results - start)
        params = {
            "search_query": f'all:"{query}"',
            "start": start,
            "max_results": n,
            "sortBy": sort_by,
            "sortOrder": "descending",
        }
        url = f"{ARXIV_API_URL}?{urllib.parse.urlencode(params)}"
        logger.info(f"Fetching arXiv batch start={start} n={n}")

        raw = None
        for attempt in range(4):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "capstone-rag-project/1.0"})
                with urllib.request.urlopen(req, timeout=20) as resp:
                    raw = resp.read()
                break
            except urllib.error.HTTPError as e:
                if e.code == 429:
                    wait = 5 * (attempt + 1)
                    logger.warning(f"arXiv rate-limited (429). Backing off {wait}s (attempt {attempt + 1}/4)...")
                    time.sleep(wait)
                    continue
                logger.error(f"arXiv fetch failed: {e}")
                break
            except Exception as e:
                logger.error(f"arXiv fetch failed: {e}")
                break

        if raw is None:
            logger.error("Giving up on this batch after retries.")
            break

        root = ET.fromstring(raw)
        entries = root.findall("atom:entry", NS)
        if not entries:
            break

        for entry in entries:
            arxiv_id = entry.find("atom:id", NS).text.strip()
            title = entry.find("atom:title", NS).text.strip().replace("\n", " ")
            summary = entry.find("atom:summary", NS).text.strip().replace("\n", " ")
            published = entry.find("atom:published", NS).text.strip()
            authors = [a.find("atom:name", NS).text for a in entry.findall("atom:author", NS)]
            categories = [c.attrib.get("term") for c in entry.findall("atom:category", NS)]

            records.append({
                "id": arxiv_id,
                "title": title,
                "abstract": summary,
                "published": published,
                "authors": authors,
                "categories": categories,
                "source": "arxiv",
            })

        start += n
        time.sleep(3)  # arXiv tightened throttling in 2026 (~1 req/3s); be conservative

    return records


def validate_and_normalize(records: list[dict]) -> list[dict]:
    """Drop malformed records, dedupe by id, trim whitespace."""
    seen = set()
    clean = []
    for r in records:
        if not r.get("id") or not r.get("abstract"):
            continue
        if r["id"] in seen:
            continue
        seen.add(r["id"])
        r["title"] = " ".join(r["title"].split())
        r["abstract"] = " ".join(r["abstract"].split())
        clean.append(r)
    return clean


def persist(records: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info(f"Wrote {len(records)} records to {out_path}")


def run(query: str, max_results: int, out_path: str, sort_by: str = "relevance") -> list[dict]:
    raw = fetch_arxiv(query, max_results=max_results, sort_by=sort_by)
    clean = validate_and_normalize(raw)
    persist(clean, Path(out_path))
    return clean


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest arXiv papers into a local JSONL corpus.")
    parser.add_argument("--query", default="retrieval augmented generation", help="arXiv search query")
    parser.add_argument("--max-results", type=int, default=60, help="Number of papers to fetch")
    parser.add_argument("--out", default="data/corpus.jsonl", help="Output JSONL path")
    args = parser.parse_args()

    run(args.query, args.max_results, args.out)
