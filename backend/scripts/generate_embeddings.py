"""
scripts/generate_embeddings.py
================================
Generate and persist paper embeddings for one or all models.

Usage:
    python scripts/generate_embeddings.py --model miniml
    python scripts/generate_embeddings.py --model all       # runs all 3

CPU-optimised defaults:
  - Batch size 32 (fits in RAM without GPU)
  - Uses float32 (not float16 — CPU doesn't benefit from half-precision)
  - Saves .npy per model; also updates FAISS index
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from app.core.config import get_settings
from app.ml.retrieval import EmbeddingModel, FAISSIndex, MODEL_REGISTRY

settings = get_settings()


def generate_for_model(model_key: str, df: pd.DataFrame) -> None:
    print(f"\n{'='*60}")
    print(f"Model: {model_key}  ({MODEL_REGISTRY[model_key]})")
    print(f"Papers: {len(df)}")
    print(f"{'='*60}")

    emb_dir = settings.embeddings_dir
    emb_dir.mkdir(parents=True, exist_ok=True)
    emb_path = emb_dir / f"{model_key}_embeddings.npy"

    # Concatenate title + abstract for richer representation
    texts = [
        f"{row.title}. {row.abstract}"
        for row in df.itertuples()
    ]

    model = EmbeddingModel(model_key)
    batch_size = 32   # CPU-safe batch size

    # Generate in batches with progress bar
    all_embs = []
    for i in tqdm(range(0, len(texts), batch_size), desc=f"Encoding ({model_key})"):
        batch = texts[i : i + batch_size]
        embs = model.encode(batch, batch_size=batch_size, show_progress=False)
        all_embs.append(embs)

    embeddings = np.vstack(all_embs).astype(np.float32)
    np.save(str(emb_path), embeddings)
    print(f"✓ Embeddings saved: {emb_path}  shape={embeddings.shape}")

    # Build FAISS index
    metadata = [
        {
            "arxiv_id":   row.arxiv_id,
            "paper_uuid": str(getattr(row, "id", "")),
            "title":      row.title,
        }
        for row in df.itertuples()
    ]

    faiss_idx = FAISSIndex(model_key)
    faiss_idx.build(embeddings, metadata)
    print(f"✓ FAISS index saved: {settings.faiss_dir / model_key}_index.faiss")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate paper embeddings")
    parser.add_argument(
        "--model",
        choices=list(MODEL_REGISTRY.keys()) + ["all"],
        default="miniml",
        help="Which embedding model to use (or 'all')",
    )
    parser.add_argument(
        "--parquet",
        default="data/processed/papers.parquet",
        help="Path to preprocessed papers parquet",
    )
    args = parser.parse_args()

    parquet_path = Path(args.parquet)
    if not parquet_path.exists():
        print(f"Error: {parquet_path} not found. Run scripts/preprocess_papers.py first.")
        sys.exit(1)

    df = pd.read_parquet(parquet_path)
    print(f"Loaded {len(df)} papers from {parquet_path}")

    models = list(MODEL_REGISTRY.keys()) if args.model == "all" else [args.model]
    for model_key in models:
        generate_for_model(model_key, df)

    print("\n✓ Embedding generation complete.")


if __name__ == "__main__":
    main()
