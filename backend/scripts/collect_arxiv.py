"""
scripts/collect_arxiv.py
========================
Collect papers from the arXiv API by category and date range.

Usage:
    python scripts/collect_arxiv.py --max-papers 250 --output data/raw

Features:
  - Rate-limited (3 req/s per arXiv terms of use)
  - Resumable (skips already-downloaded files)
  - Deduplication by arxiv_id
  - Validates required fields
  - Progress bar
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timedelta
from pathlib import Path

import arxiv
from tqdm import tqdm

# arXiv categories to collect
CATEGORIES = [
    "cs.LG",   # Machine Learning
    "cs.CV",   # Computer Vision
    "cs.CL",   # Computation and Language (NLP)
    "cs.AI",   # Artificial Intelligence
    "cs.CR",   # Cryptography and Security
    "cs.SE",   # Software Engineering
    "stat.ML", # Statistics - Machine Learning
]

# Date range (papers from last 5 years)
DATE_FROM = datetime(2019, 1, 1)
DATE_TO   = datetime(2024, 12, 31)

RATE_LIMIT_SECONDS = 3.0
MAX_PER_QUERY = 2000


def collect_papers(
    max_papers: int,
    output_dir: Path,
    rate_limit: float = RATE_LIMIT_SECONDS,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    seen_ids: set[str] = set()
    total_collected = 0

    # Load already-collected IDs to allow resuming
    for existing_file in output_dir.glob("*.json"):
        try:
            with open(existing_file) as f:
                paper = json.load(f)
                seen_ids.add(paper.get("arxiv_id", ""))
        except Exception:
            pass

    print(f"Resuming from {len(seen_ids)} existing papers.")

    client = arxiv.Client(
        page_size=100,
        delay_seconds=rate_limit,
        num_retries=5,
    )

    pbar = tqdm(total=max_papers, desc="Collecting papers", unit="paper")
    pbar.update(len(seen_ids))

    for category in CATEGORIES:
        if total_collected + len(seen_ids) >= max_papers:
            break

        print(f"\n→ Category: {category}")

        # Slide a 6-month window across the date range
        window_start = DATE_FROM
        while window_start < DATE_TO:
            window_end = min(window_start + timedelta(days=180), DATE_TO)

            query = (
                f"cat:{category} AND "
                f"submittedDate:[{window_start.strftime('%Y%m%d')}0000 TO "
                f"{window_end.strftime('%Y%m%d')}2359]"
            )

            search = arxiv.Search(
                query=query,
                max_results=MAX_PER_QUERY,
                sort_by=arxiv.SortCriterion.SubmittedDate,
                sort_order=arxiv.SortOrder.Descending,
            )

            try:
                for result in client.results(search):
                    arxiv_id = result.entry_id.split("/abs/")[-1].split("v")[0]

                    if arxiv_id in seen_ids:
                        continue

                    # Validate required fields
                    if not result.title or not result.summary:
                        continue

                    paper = {
                        "arxiv_id":        arxiv_id,
                        "title":           result.title.replace("\n", " ").strip(),
                        "abstract":        result.summary.replace("\n", " ").strip(),
                        "authors":         [a.name for a in result.authors],
                        "categories":      result.categories,
                        "primary_category": result.primary_category,
                        "submitted_date":  result.published.date().isoformat(),
                        "doi":             result.doi,
                        "pdf_url":         result.pdf_url,
                    }

                    # Save individual file
                    out_file = output_dir / f"{arxiv_id.replace('/', '_')}.json"
                    with open(out_file, "w", encoding="utf-8") as f:
                        json.dump(paper, f, ensure_ascii=False)

                    seen_ids.add(arxiv_id)
                    total_collected += 1
                    pbar.update(1)

                    if total_collected + len(seen_ids) - total_collected >= max_papers:
                        break

            except Exception as e:
                print(f"Error fetching {category} {window_start}: {e}")
                time.sleep(10)

            window_start = window_end + timedelta(days=1)

    pbar.close()
    total = len(seen_ids)
    print(f"\n✓ Collection complete: {total} papers in {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect arXiv papers")
    # 🚨 CHANGED DEFAULT FROM 25000 TO 250 TO PREVENT RENDER TIMEOUTS 🚨
    parser.add_argument("--max-papers", type=int, default=250)
    parser.add_argument("--output", type=str, default="data/raw")
    parser.add_argument("--rate-limit", type=float, default=RATE_LIMIT_SECONDS)
    args = parser.parse_args()

    collect_papers(
        max_papers=args.max_papers,
        output_dir=Path(args.output),
        rate_limit=args.rate_limit,
    )


if __name__ == "__main__":
    main()
