"""
Module 3: Research Recommendation System
==========================================
Implements:
  1. Content-based: cosine similarity via FAISS nearest-neighbor lookup
  2. Graph-enhanced: FAISS similarity + KG neighborhood expansion + re-ranking
  3. Explanation generation for every recommendation
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple
from uuid import UUID

import numpy as np
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import logger
from app.ml.retrieval import RetrievalService
from app.ml.knowledge_graph import KnowledgeGraphService
from app.models.orm import Paper, PaperAuthor, Author


# ── Explanation generator ─────────────────────────────────────────────────────

def _build_explanation(
    score: float,
    shared_authors: List[str],
    same_topic: bool,
    graph_boost: float,
    seed_title: str,
) -> Tuple[str, str]:
    """
    Returns (explanation_text, explanation_type).
    Picks the most salient reason for the recommendation.
    """
    if shared_authors:
        author_str = ", ".join(shared_authors[:2])
        return (
            f"Recommended because co-authored by {author_str}, who also wrote "
            f"papers closely related to your query.",
            "shared_author",
        )
    if same_topic and graph_boost > 0.1:
        return (
            f"Recommended because it belongs to the same research cluster "
            f"and is strongly connected in the knowledge graph "
            f"(graph boost: {graph_boost:.2f}).",
            "shared_topic",
        )
    if same_topic:
        return (
            "Recommended because it belongs to the same research topic cluster.",
            "shared_topic",
        )
    if graph_boost > 0.1:
        return (
            f"Recommended because it is closely connected in the knowledge "
            f"graph to papers similar to your seed paper.",
            "graph_proximity",
        )
    return (
        f"Recommended because it shares strong semantic similarity "
        f"(score: {score:.3f}) with your query paper.",
        "semantic_similarity",
    )


# ── Recommendation Engine ─────────────────────────────────────────────────────

class RecommendationEngine:
    """
    Generates paper recommendations with explanations.
    Uses BGE by default (falls back to best available model).
    """

    def __init__(
        self,
        retrieval_svc: RetrievalService,
        kg_svc: KnowledgeGraphService,
    ) -> None:
        self.retrieval = retrieval_svc
        self.kg = kg_svc

    def _best_model(self) -> str:
        for key in ["bge", "mpnet", "miniml"]:
            if key in self.retrieval.loaded_models:
                return key
        raise RuntimeError("No embedding model loaded for recommendations.")

    async def content_based(
        self,
        seed_paper: Paper,
        session: AsyncSession,
        top_k: int = 10,
    ) -> Tuple[List[Dict], float]:
        """
        Pure content-based: encode seed abstract → FAISS search → return neighbors.
        """
        t0 = time.perf_counter()
        model_key = self._best_model()
        engine = self.retrieval.get_engine(model_key)

        # Encode the seed paper's abstract
        seed_text = f"{seed_paper.title}. {seed_paper.abstract}"
        seed_emb = engine.embedding_model.encode([seed_text], is_query=False)

        # Search (return top_k+1 to exclude the seed itself)
        raw = engine.faiss_index.search(seed_emb, top_k=top_k + 5)

        # Fetch paper details from DB for each result
        results = []
        for score, meta in raw:
            arxiv_id = meta.get("arxiv_id")
            if not arxiv_id or arxiv_id == seed_paper.arxiv_id:
                continue
            paper = await _fetch_paper_by_arxiv(session, arxiv_id)
            if paper is None:
                continue

            same_topic = (
                paper.topic_id is not None
                and paper.topic_id == seed_paper.topic_id
            )
            explanation, exp_type = _build_explanation(
                score=score,
                shared_authors=[],
                same_topic=same_topic,
                graph_boost=0.0,
                seed_title=seed_paper.title,
            )
            results.append({
                "paper": paper,
                "score": score,
                "explanation": explanation,
                "explanation_type": exp_type,
            })
            if len(results) >= top_k:
                break

        latency_ms = (time.perf_counter() - t0) * 1000
        return results, latency_ms

    async def graph_enhanced(
        self,
        seed_paper: Paper,
        session: AsyncSession,
        top_k: int = 10,
        alpha: float = 0.7,
        beta: float = 0.2,
        gamma: float = 0.1,
    ) -> Tuple[List[Dict], float]:
        """
        Graph-enhanced recommendations.

        score(p) = alpha * cosine_sim(seed, p)
                 + beta  * graph_neighbor_overlap(top_k_neighbors)
                 + gamma * pagerank(p)

        alpha + beta + gamma should sum to 1.
        """
        t0 = time.perf_counter()
        model_key = self._best_model()
        engine = self.retrieval.get_engine(model_key)

        seed_text = f"{seed_paper.title}. {seed_paper.abstract}"
        seed_emb = engine.embedding_model.encode([seed_text], is_query=False)

        # Get broader candidate pool for re-ranking
        raw = engine.faiss_index.search(seed_emb, top_k=top_k * 3)

        # Get seed's KG neighbors
        kg_neighbors: set = set()
        if self.kg.is_ready:
            kg_neighbors = self.kg.get_paper_neighbors(
                seed_paper.arxiv_id, radius=1
            )

        # Get PageRank scores
        pagerank = self.kg.pagerank_scores if self.kg.is_ready else {}

        candidate_arxiv_ids = [
            m.get("arxiv_id") for _, m in raw
            if m.get("arxiv_id") and m.get("arxiv_id") != seed_paper.arxiv_id
        ]

        # Re-rank with graph signal
        ranked: List[Tuple[float, str, float]] = []
        for (cos_score, meta) in raw:
            arxiv_id = meta.get("arxiv_id")
            if not arxiv_id or arxiv_id == seed_paper.arxiv_id:
                continue

            kg_in_neighbors = 1.0 if arxiv_id in kg_neighbors else 0.0
            pr_score = pagerank.get(arxiv_id, 0.0)

            # Normalize PageRank to [0,1] range (already small floats)
            combined = alpha * cos_score + beta * kg_in_neighbors + gamma * pr_score
            ranked.append((combined, arxiv_id, cos_score))

        ranked.sort(key=lambda x: x[0], reverse=True)

        # Fetch seed authors for shared-author detection
        seed_author_ids = await _fetch_author_ids(session, seed_paper.id)

        results = []
        for combined_score, arxiv_id, cos_score in ranked[:top_k + 5]:
            paper = await _fetch_paper_by_arxiv(session, arxiv_id)
            if paper is None:
                continue

            # Shared author check
            cand_author_ids = await _fetch_author_ids(session, paper.id)
            shared_ids = seed_author_ids & cand_author_ids
            shared_names: List[str] = []
            if shared_ids:
                shared_names = await _fetch_author_names(session, shared_ids)

            same_topic = (
                paper.topic_id is not None
                and paper.topic_id == seed_paper.topic_id
            )
            graph_boost = float(arxiv_id in kg_neighbors)

            explanation, exp_type = _build_explanation(
                score=cos_score,
                shared_authors=shared_names,
                same_topic=same_topic,
                graph_boost=graph_boost,
                seed_title=seed_paper.title,
            )

            results.append({
                "paper": paper,
                "score": combined_score,
                "explanation": explanation,
                "explanation_type": exp_type,
            })
            if len(results) >= top_k:
                break

        latency_ms = (time.perf_counter() - t0) * 1000
        return results, latency_ms


# ── DB helpers ────────────────────────────────────────────────────────────────

async def _fetch_paper_by_arxiv(
    session: AsyncSession, arxiv_id: str
) -> Optional[Paper]:
    result = await session.execute(
        select(Paper).where(Paper.arxiv_id == arxiv_id)
    )
    return result.scalar_one_or_none()


async def _fetch_author_ids(session: AsyncSession, paper_id) -> set:
    result = await session.execute(
        select(PaperAuthor.author_id).where(PaperAuthor.paper_id == paper_id)
    )
    return {row[0] for row in result.fetchall()}


async def _fetch_author_names(session: AsyncSession, author_ids: set) -> List[str]:
    if not author_ids:
        return []
    result = await session.execute(
        select(Author.name).where(Author.id.in_(author_ids))
    )
    return [row[0] for row in result.fetchall()]
