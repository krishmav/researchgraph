"""
scripts/build_knowledge_graph.py
==================================
Build the knowledge graph from:
  - Paper similarity (top-20 FAISS neighbours per paper)
  - Author–paper relationships
  - Topic memberships
  - Research area memberships

Also computes research gaps (sparse region + structural analysis).

Usage:
    python scripts/build_knowledge_graph.py --model miniml
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from app.core.config import get_settings
from app.ml.knowledge_graph import KnowledgeGraphBuilder
from app.ml.gap_finder import SparseRegionAnalyzer, StructuralGapAnalyzer, _describe_gap

settings = get_settings()


async def load_db_data() -> Tuple[List[dict], List[Tuple[str, str]]]:
    """Load paper metadata and author pairs from PostgreSQL."""
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
    from sqlalchemy import text

    def _async_url(url: str) -> str:
        return url.replace("postgresql://", "postgresql+asyncpg://").replace("postgres://", "postgresql+asyncpg://")

    engine = create_async_engine(_async_url(settings.database_url), echo=False)
    SessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with SessionLocal() as session:
        result = await session.execute(text("""
            SELECT arxiv_id, title, primary_category, categories, topic_id
            FROM papers ORDER BY submitted_date DESC
        """))
        papers = [
            {
                "arxiv_id": r[0],
                "title": r[1],
                "primary_category": r[2] or "",
                "categories": r[3] or [],
                "topic_id": r[4],
            }
            for r in result.fetchall()
        ]

        result2 = await session.execute(text("""
            SELECT p.arxiv_id, a.name
            FROM paper_authors pa
            JOIN papers p ON pa.paper_id = p.id
            JOIN authors a ON pa.author_id = a.id
        """))
        author_pairs = [(r[0], r[1]) for r in result2.fetchall()]

        result3 = await session.execute(text("""
            SELECT id, label, top_words FROM topics WHERE is_outlier = false
        """))
        topics = [
            {"id": r[0], "label": r[1], "top_words": r[2] or []}
            for r in result3.fetchall()
        ]

        result4 = await session.execute(text("""
            SELECT arxiv_id, topic_id FROM papers WHERE topic_id IS NOT NULL
        """))
        paper_topic_pairs = [(r[0], r[1]) for r in result4.fetchall()]

    await engine.dispose()
    return papers, author_pairs, topics, paper_topic_pairs


def compute_similarity_edges(
    embeddings: np.ndarray,
    arxiv_ids: List[str],
    top_k: int = 20,
) -> List[Tuple[str, str, float]]:
    """Compute top-k FAISS neighbours per paper → similarity edges."""
    import faiss

    print(f"Computing similarity edges (top-{top_k} per paper)…")
    embs = embeddings.astype(np.float32)
    # Normalize for cosine similarity
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    embs_normed = embs / (norms + 1e-8)

    index = faiss.IndexFlatIP(embs_normed.shape[1])
    index.add(embs_normed)

    edges = []
    batch_size = 500
    for i in tqdm(range(0, len(arxiv_ids), batch_size), desc="Finding neighbours"):
        batch = embs_normed[i : i + batch_size]
        scores, indices = index.search(batch, top_k + 1)  # +1 because self is returned

        for local_j, (row_scores, row_indices) in enumerate(zip(scores, indices)):
            global_j = i + local_j
            for score, idx in zip(row_scores, row_indices):
                if idx == global_j or idx < 0:
                    continue
                if score < 0.5:   # Only keep meaningful similarities
                    continue
                id_a = arxiv_ids[global_j]
                id_b = arxiv_ids[idx]
                if id_a < id_b:   # deduplicate undirected edges
                    edges.append((id_a, id_b, float(score)))

    print(f"  {len(edges)} similarity edges computed.")
    return edges


def build_graph(
    papers: List[dict],
    author_pairs: List[Tuple[str, str]],
    topics: List[dict],
    paper_topic_pairs: List[Tuple[str, int]],
    similarity_edges: List[Tuple[str, str, float]],
) -> None:
    builder = KnowledgeGraphBuilder()

    print("Adding paper nodes…")
    builder.add_papers(papers)

    print("Adding author nodes and edges…")
    builder.add_authors(author_pairs)

    print("Adding topic nodes…")
    builder.add_topics(topics)

    print("Adding topic memberships…")
    builder.add_topic_memberships(paper_topic_pairs)

    # Research area nodes from primary_category
    area_map: Dict[str, List[str]] = {}
    for p in papers:
        cat = p.get("primary_category") or ""
        if cat:
            area_map.setdefault(cat, []).append(p["arxiv_id"])
    print(f"Adding {len(area_map)} research area nodes…")
    builder.add_research_areas(area_map)

    print("Adding similarity edges…")
    builder.add_similarity_edges(similarity_edges)

    print("Saving knowledge graph…")
    builder.save()


def compute_research_gaps(
    embeddings: np.ndarray,
    arxiv_ids: List[str],
    topic_assignments: Dict[str, int],
    topic_labels: Dict[int, str],
    paper_topic_pairs: List[Tuple[str, int]],
) -> None:
    processed_dir = Path("data/processed")

    # Topic centroids
    topic_centroids: Dict[int, np.ndarray] = {}
    topic_paper_rows: Dict[int, List[int]] = {}
    for i, aid in enumerate(arxiv_ids):
        tid = topic_assignments.get(aid, -1)
        if tid == -1:
            continue
        topic_paper_rows.setdefault(tid, []).append(i)

    for tid, rows in topic_paper_rows.items():
        topic_centroids[tid] = embeddings[rows].mean(axis=0)

    # 2-D UMAP projection (if available)
    umap_2d_path = processed_dir / "umap_2d.npy"
    if not umap_2d_path.exists():
        print("  2-D UMAP not found, skipping sparse region analysis.")
        return

    umap_2d = np.load(str(umap_2d_path))

    # Compute 2-D topic centroids
    topic_centroids_2d: Dict[int, np.ndarray] = {}
    for tid, rows in topic_paper_rows.items():
        valid_rows = [r for r in rows if r < len(umap_2d)]
        if valid_rows:
            topic_centroids_2d[tid] = umap_2d[valid_rows].mean(axis=0)

    print("Computing sparse region gaps…")
    sparse_analyzer = SparseRegionAnalyzer(
        umap_2d=umap_2d,
        arxiv_ids=arxiv_ids,
        topic_assignments=topic_assignments,
        topic_labels=topic_labels,
        topic_centroids=topic_centroids_2d,
    )
    sparse_gaps = sparse_analyzer.find_sparse_regions(n_gaps=15)

    # Add descriptions
    for gap in sparse_gaps:
        gap["description"] = _describe_gap(
            gap.get("flanking_topics", []),
            gap.get("sparse_score", 0.0),
            gap.get("semantic_distance", 0.0),
        )

    with open(processed_dir / "research_gaps_sparse.json", "w") as f:
        json.dump(sparse_gaps, f, default=str)
    print(f"✓ Saved {len(sparse_gaps)} sparse gaps.")

    # Structural gaps
    print("Computing structural graph gaps…")
    co_occ: Dict[Tuple[int, int], int] = {}
    for aid, tid in paper_topic_pairs:
        for aid2, tid2 in paper_topic_pairs:
            if aid == aid2 and tid != tid2:
                key = (min(tid, tid2), max(tid, tid2))
                co_occ[key] = co_occ.get(key, 0) + 1

    struct_analyzer = StructuralGapAnalyzer(
        topic_centroids=topic_centroids,
        topic_labels=topic_labels,
        co_occurrence_counts=co_occ,
    )
    struct_gaps_raw = struct_analyzer.find_structural_gaps(n_gaps=15)
    struct_gaps = [
        {
            "flanking_topics": [g["label_a"], g["label_b"]],
            "sparse_score": g["gap_score"],
            "semantic_distance": float(1.0 - g["semantic_similarity"]),
            "description": _describe_gap(
                [g["label_a"], g["label_b"]],
                g["gap_score"],
                float(1.0 - g["semantic_similarity"]),
            ),
            "evidence_papers": [],
        }
        for g in struct_gaps_raw
    ]

    with open(processed_dir / "research_gaps_structural.json", "w") as f:
        json.dump(struct_gaps, f, default=str)
    print(f"✓ Saved {len(struct_gaps)} structural gaps.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="miniml", choices=["miniml", "mpnet", "bge"])
    args = parser.parse_args()

    print("Loading DB data…")
    papers, author_pairs, topics, paper_topic_pairs = asyncio.run(load_db_data())
    print(f"  Papers: {len(papers)}, Authors: {len(author_pairs)}, Topics: {len(topics)}")

    print(f"Loading {args.model} embeddings…")
    emb_path = settings.embeddings_dir / f"{args.model}_embeddings.npy"
    if not emb_path.exists():
        print(f"Error: {emb_path} not found. Run generate_embeddings.py first.")
        sys.exit(1)

    embeddings = np.load(str(emb_path)).astype(np.float32)
    arxiv_ids = [p["arxiv_id"] for p in papers]

    # Truncate embeddings to match papers loaded from DB
    if len(embeddings) > len(arxiv_ids):
        embeddings = embeddings[:len(arxiv_ids)]
    elif len(embeddings) < len(arxiv_ids):
        arxiv_ids = arxiv_ids[:len(embeddings)]
        papers = papers[:len(embeddings)]

    similarity_edges = compute_similarity_edges(embeddings, arxiv_ids, top_k=20)
    build_graph(papers, author_pairs, topics, paper_topic_pairs, similarity_edges)

    # Load topic labels and assignments for gap analysis
    processed_dir = Path("data/processed")
    assign_path = processed_dir / "topic_assignments.json"
    labels_path = processed_dir / "topic_labels.json"

    if assign_path.exists() and labels_path.exists():
        with open(assign_path) as f:
            topic_assignments = json.load(f)
        with open(labels_path) as f:
            raw_labels = json.load(f)
            topic_labels = {int(k): v["label"] for k, v in raw_labels.items()}

        compute_research_gaps(
            embeddings, arxiv_ids, topic_assignments, topic_labels, paper_topic_pairs
        )
    else:
        print("Topic assignments not found. Run train_bertopic.py before building gaps.")

    print("\n✓ Knowledge graph build complete.")


if __name__ == "__main__":
    main()
