"""
scripts/train_bertopic.py
==========================
Train BERTopic on paper embeddings.
Saves model, 2-D UMAP projection, topic assignments, and topic labels.

Usage:
    python scripts/train_bertopic.py --model miniml

Outputs:
    data/processed/topic_model.pkl          — trained BERTopic
    data/processed/umap_2d.npy             — (N,2) 2-D projection
    data/processed/topic_assignments.json  — {arxiv_id: topic_id}
    data/processed/topic_labels.json       — {topic_id: {label, top_words}}
    + inserts topics and topic_trends into PostgreSQL
"""
from __future__ import annotations

import argparse
import asyncio
import json
import pickle
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from app.core.config import get_settings
from app.core.logging import logger

settings = get_settings()


def train_bertopic(model_key: str, parquet_path: Path) -> None:
    from bertopic import BERTopic
    from umap import UMAP
    from hdbscan import HDBSCAN

    print(f"Loading embeddings for model={model_key}…")
    emb_path = settings.embeddings_dir / f"{model_key}_embeddings.npy"
    if not emb_path.exists():
        print(f"Error: {emb_path} not found. Run generate_embeddings.py first.")
        sys.exit(1)

    embeddings = np.load(str(emb_path)).astype(np.float32)
    df = pd.read_parquet(parquet_path)
    abstracts = df["abstract"].tolist()

    print(f"  Papers: {len(df)}, Embeddings shape: {embeddings.shape}")

    # UMAP for dimensionality reduction (5-D for clustering)
    umap_5d = UMAP(
        n_components=5,
        n_neighbors=15,
        min_dist=0.0,
        metric="cosine",
        random_state=42,
        low_memory=True,   # important for large CPU-only runs
    )

    # HDBSCAN for clustering
    hdbscan_model = HDBSCAN(
        min_cluster_size=15,
        min_samples=10,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )

    print("Training BERTopic (this may take 10–30 minutes on CPU)…")
    topic_model = BERTopic(
        umap_model=umap_5d,
        hdbscan_model=hdbscan_model,
        top_n_words=10,
        nr_topics="auto",
        calculate_probabilities=False,
        verbose=True,
    )

    topics, _ = topic_model.fit_transform(abstracts, embeddings=embeddings)
    print(f"✓ BERTopic training complete. Topics discovered: {topic_model.get_topic_info().shape[0]}")

    # 2-D UMAP for visualisation (separate from 5-D clustering)
    print("Computing 2-D UMAP projection for visualisation…")
    umap_2d = UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        metric="cosine",
        random_state=42,
        low_memory=True,
    )
    proj_2d = umap_2d.fit_transform(embeddings)

    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Save model
    with open(processed_dir / "topic_model.pkl", "wb") as f:
        pickle.dump(topic_model, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"✓ Saved topic_model.pkl")

    # Save 2-D projection
    np.save(str(processed_dir / "umap_2d.npy"), proj_2d.astype(np.float32))
    print(f"✓ Saved umap_2d.npy  shape={proj_2d.shape}")

    # Topic assignments
    assignments = {
        df.iloc[i]["arxiv_id"]: int(t)
        for i, t in enumerate(topics)
    }
    with open(processed_dir / "topic_assignments.json", "w") as f:
        json.dump(assignments, f)
    print(f"✓ Saved topic_assignments.json")

    # Topic labels
    topic_info = topic_model.get_topic_info()
    labels: dict = {}
    for _, row in topic_info.iterrows():
        tid = int(row["Topic"])
        top_words = [w for w, _ in topic_model.get_topic(tid)[:10]] if tid != -1 else []
        labels[tid] = {
            "label": f"Topic {tid}: {', '.join(top_words[:3])}",
            "top_words": top_words,
        }
    with open(processed_dir / "topic_labels.json", "w") as f:
        json.dump(labels, f)
    print(f"✓ Saved topic_labels.json ({len(labels)} topics)")

    # Insert topics and monthly trends into PostgreSQL
    asyncio.run(_insert_topics_db(df, topics, labels))

    print("\n✓ BERTopic pipeline complete.")


async def _insert_topics_db(
    df: pd.DataFrame,
    topics: list,
    labels: dict,
) -> None:
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
    from sqlalchemy import text
    from app.models.orm import Topic, TopicTrend, Paper

    def _async_url(url: str) -> str:
        return url.replace("postgresql://", "postgresql+asyncpg://").replace("postgres://", "postgresql+asyncpg://")

    engine = create_async_engine(_async_url(settings.database_url), echo=False)
    SessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with SessionLocal() as session:
        # Clear existing topics
        await session.execute(text("DELETE FROM topic_trends"))
        await session.execute(text("DELETE FROM topics"))
        await session.commit()

        # Insert topics
        topic_id_map: dict = {}   # tid (int) → db id
        for tid, info in tqdm(labels.items(), desc="Inserting topics"):
            t = Topic(
                label=info["label"],
                top_words=info["top_words"],
                paper_count=topics.count(tid) if isinstance(topics, list) else 0,
                is_outlier=(int(tid) == -1),
            )
            session.add(t)
            await session.flush()
            topic_id_map[int(tid)] = t.id

        await session.commit()

        # Update paper topic assignments
        for i, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc="Updating paper topics")):
            topic_id_int = int(topics[i])
            db_topic_id = topic_id_map.get(topic_id_int)
            if db_topic_id:
                await session.execute(
                    text("UPDATE papers SET topic_id = :tid WHERE arxiv_id = :aid"),
                    {"tid": db_topic_id, "aid": row["arxiv_id"]},
                )

        await session.commit()

        # Compute monthly trends per topic
        df_copy = df.copy()
        df_copy["topic_int"] = topics
        df_copy["submitted_date"] = pd.to_datetime(df_copy["submitted_date"])
        df_copy["year_month"] = df_copy["submitted_date"].dt.to_period("M").dt.to_timestamp()

        monthly = (
            df_copy.groupby(["topic_int", "year_month"])
            .size()
            .reset_index(name="count")
        )

        for _, mrow in tqdm(monthly.iterrows(), total=len(monthly), desc="Inserting trends"):
            db_tid = topic_id_map.get(int(mrow["topic_int"]))
            if not db_tid:
                continue
            trend = TopicTrend(
                topic_id=db_tid,
                year_month=mrow["year_month"].date(),
                paper_count=int(mrow["count"]),
            )
            session.add(trend)
        await session.commit()

    await engine.dispose()
    print("✓ Topics and trends inserted into PostgreSQL.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["miniml", "mpnet", "bge"], default="miniml")
    parser.add_argument("--parquet", default="data/processed/papers.parquet")
    args = parser.parse_args()
    train_bertopic(args.model, Path(args.parquet))


if __name__ == "__main__":
    main()
