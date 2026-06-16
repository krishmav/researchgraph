"""
scripts/preprocess_papers.py
=============================
Read raw JSON files → deduplicate → validate → insert into PostgreSQL.

Usage:
    python scripts/preprocess_papers.py --raw-dir data/raw
"""
from __future__ import annotations
from datetime import datetime

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from app.core.config import get_settings
from app.models.orm import Author, Paper, PaperAuthor, Topic

settings = get_settings()


def _async_url(url: str) -> str:
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+asyncpg://", 1)
    if url.startswith("postgres://"):
        return url.replace("postgres://", "postgresql+asyncpg://", 1)
    return url


async def ingest(raw_dir: Path) -> None:
    engine = create_async_engine(_async_url(settings.database_url), echo=False)
    SessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    # Load all raw JSON files
    files = list(raw_dir.glob("*.json"))
    print(f"Found {len(files)} raw paper files.")

    # Deduplicate in memory first
    papers_by_id: Dict[str, dict] = {}
    for fp in tqdm(files, desc="Loading JSON files"):
        try:
            with open(fp, encoding="utf-8") as f:
                p = json.load(f)
            aid = p.get("arxiv_id", "")
            if aid and aid not in papers_by_id:
                papers_by_id[aid] = p
        except Exception:
            pass

    papers = list(papers_by_id.values())
    print(f"Unique papers after dedup: {len(papers)}")

    # Save a processed parquet for later use by ML scripts
    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(papers)
    df.to_parquet(processed_dir / "papers.parquet", index=False)
    print(f"Saved parquet: {processed_dir / 'papers.parquet'}")

    # Insert into DB
    async with SessionLocal() as session:
        # Check what's already in DB
        result = await session.execute(text("SELECT arxiv_id FROM papers"))
        existing = {row[0] for row in result.fetchall()}
        new_papers = [p for p in papers if p["arxiv_id"] not in existing]
        print(f"New papers to insert: {len(new_papers)}")

        author_cache: Dict[str, int] = {}

        BATCH = 500
        for i in tqdm(range(0, len(new_papers), BATCH), desc="Inserting papers"):
            batch = new_papers[i : i + BATCH]
            for p in batch:
                paper_orm = Paper(
                    arxiv_id=p["arxiv_id"],
                    title=p["title"][:500],
                    abstract=p["abstract"],
                    authors=p.get("authors", []),
                    categories=p.get("categories", []),
                    primary_category=p.get("primary_category"),
                    submitted_date=datetime.strptime(str(p["submitted_date"])[:10], "%Y-%m-%d").date(),
                    doi=p.get("doi"),
                    pdf_url=p.get("pdf_url"),
                )
                session.add(paper_orm)
                await session.flush()

                # Authors
                seen_author_ids = set()
                for order, author_name in enumerate(p.get("authors", [])[:20]):
                    name_lower = author_name.lower().strip()
                    if name_lower in author_cache:
                        author_id = author_cache[name_lower]
                    else:
                        res = await session.execute(
                            select(Author).where(Author.name_lower == name_lower)
                        )
                        author = res.scalar_one_or_none()
                        if not author:
                            author = Author(name=author_name, name_lower=name_lower)
                            session.add(author)
                            await session.flush()
                        author_id = author.id
                        author_cache[name_lower] = author_id
                    if author_id in seen_author_ids:
                        continue
                    seen_author_ids.add(author_id)
                    pa = PaperAuthor(
                        paper_id=paper_orm.id,
                        author_id=author_id,
                        author_order=order,
                    )
                    session.add(pa)

            await session.commit()

        # Update author paper counts
        await session.execute(text("""
            UPDATE authors a
            SET paper_count = (
                SELECT COUNT(*) FROM paper_authors pa WHERE pa.author_id = a.id
            )
        """))
        await session.commit()

    await engine.dispose()
    print("✓ Preprocessing complete.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default="data/raw")
    args = parser.parse_args()
    asyncio.run(ingest(Path(args.raw_dir)))


if __name__ == "__main__":
    main()
