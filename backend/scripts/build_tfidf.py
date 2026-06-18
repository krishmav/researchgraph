"""
scripts/build_tfidf.py
=======================
Build TF-IDF matrix and vectorizer from preprocessed papers.

Usage:
    python scripts/build_tfidf.py

Outputs:
    data/tfidf/tfidf_matrix.npz   — Scipy sparse matrix
    data/tfidf/vectorizer.pkl     — Fitted TfidfVectorizer
    data/tfidf/paper_ids.json     — {row_idx: {arxiv_id, title, paper_uuid}}
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import pandas as pd
from scipy.sparse import save_npz
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from app.core.config import get_settings

settings = get_settings()


def build_tfidf(parquet_path: Path) -> None:
    print(f"Loading papers from {parquet_path}…")
    df = pd.read_parquet(parquet_path)
    print(f"  {len(df)} papers loaded.")

    # Title + abstract concatenation
    corpus = [
        f"{row.title} {row.abstract}"
        for row in tqdm(df.itertuples(), total=len(df), desc="Building corpus")
    ]

    print("Fitting TF-IDF vectorizer…")
    vectorizer = TfidfVectorizer(
        max_features=75_000,
        min_df=2,           # ignore terms appearing in fewer than 2 docs
        max_df=0.85,        # ignore terms in >85% of docs
        sublinear_tf=True,  # apply log(1+tf) — better for long texts
        strip_accents="unicode",
        analyzer="word",
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9\-]{2,}\b",
        ngram_range=(1, 2),
    )

    matrix = vectorizer.fit_transform(tqdm(corpus, desc="Vectorizing"))
    print(f"  Matrix shape: {matrix.shape}")
    print(f"  Vocabulary size: {len(vectorizer.vocabulary_)}")
    print(f"  Non-zero entries: {matrix.nnz}")

    # Save
    tfidf_dir = settings.tfidf_dir
    tfidf_dir.mkdir(parents=True, exist_ok=True)

    save_npz(str(tfidf_dir / "tfidf_matrix.npz"), matrix)
    print(f"✓ Matrix saved: {tfidf_dir / 'tfidf_matrix.npz'}")

    with open(tfidf_dir / "vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"✓ Vectorizer saved: {tfidf_dir / 'vectorizer.pkl'}")

    # Paper ID mapping
    paper_ids = {
        i: {
            "arxiv_id":   row.arxiv_id,
            "title":      row.title,
            "paper_uuid": str(getattr(row, "id", "")),
        }
        for i, row in enumerate(df.itertuples())
    }
    with open(tfidf_dir / "paper_ids.json", "w") as f:
        json.dump(paper_ids, f)
    print(f"✓ ID mapping saved: {tfidf_dir / 'paper_ids.json'}")
    print("\n✓ TF-IDF build complete.")


def main() -> None:
    parquet_path = Path("data/processed/papers.parquet")
    if not parquet_path.exists():
        print(f"Error: {parquet_path} not found. Run preprocess_papers.py first.")
        sys.exit(1)
    build_tfidf(parquet_path)


if __name__ == "__main__":
    main()
