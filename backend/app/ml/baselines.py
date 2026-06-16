"""
Module 2: Information Retrieval Baselines
==========================================
Implements:
  A. Keyword Search  — PostgreSQL full-text search (GIN index, ts_query)
  B. TF-IDF Retrieval — Scikit-learn vectorizer + cosine similarity
These are baselines against which semantic retrieval is benchmarked.
"""
from __future__ import annotations

import pickle
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.sparse import load_npz, spmatrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.logging import logger

settings = get_settings()


# ── Keyword Search (PostgreSQL FTS) ──────────────────────────────────────────

class KeywordSearchEngine:
    """
    Uses PostgreSQL GIN full-text search index on papers(title, abstract).
    Supports plainto_tsquery for natural language queries.
    """

    @staticmethod
    async def search(
        session: AsyncSession,
        query: str,
        top_k: int = 10,
        category_filter: Optional[str] = None,
    ) -> Tuple[List[Dict], float]:
        """
        Returns (results, latency_ms).
        results = list of dicts with keys: arxiv_id, title, abstract, authors,
                  categories, submitted_date, score.
        """
        t0 = time.perf_counter()

        # Sanitize query — remove characters that break ts_query
        safe_query = " ".join(
            w for w in query.split()
            if w.replace("-", "").replace("_", "").isalnum()
        ) or "research"

        category_clause = (
            "AND :category = ANY(categories)" if category_filter else ""
        )

        sql = text(f"""
            SELECT
                id::text,
                arxiv_id,
                title,
                abstract,
                authors,
                categories,
                submitted_date,
                ts_rank_cd(
                    to_tsvector('english', title || ' ' || abstract),
                    plainto_tsquery('english', :query)
                ) AS score
            FROM papers
            WHERE to_tsvector('english', title || ' ' || abstract)
                @@ plainto_tsquery('english', :query)
            {category_clause}
            ORDER BY score DESC
            LIMIT :top_k
        """)

        params: Dict = {"query": safe_query, "top_k": top_k}
        if category_filter:
            params["category"] = category_filter

        result = await session.execute(sql, params)
        rows = result.fetchall()

        latency_ms = (time.perf_counter() - t0) * 1000

        results = [
            {
                "paper_id": row.id,
                "arxiv_id": row.arxiv_id,
                "title": row.title,
                "abstract": row.abstract,
                "authors": row.authors,
                "categories": row.categories,
                "submitted_date": row.submitted_date,
                "score": float(row.score),
            }
            for row in rows
        ]
        return results, latency_ms


# ── TF-IDF Retrieval ─────────────────────────────────────────────────────────

class TFIDFEngine:
    """
    Scikit-learn TF-IDF retrieval over title+abstract concatenations.

    File layout:
        data/tfidf/tfidf_matrix.npz      — Scipy sparse matrix (N_papers × vocab)
        data/tfidf/vectorizer.pkl        — Fitted TfidfVectorizer
        data/tfidf/paper_ids.json        — {row_idx: {arxiv_id, paper_uuid, title}}
    """

    def __init__(self) -> None:
        self._vectorizer: Optional[TfidfVectorizer] = None
        self._matrix: Optional[spmatrix] = None
        self._paper_ids: Optional[Dict[int, Dict]] = None
        self._loaded = False

    def load(self) -> bool:
        """Load TF-IDF artifacts from disk."""
        tfidf_dir = settings.tfidf_dir
        matrix_path = tfidf_dir / "tfidf_matrix.npz"
        vectorizer_path = tfidf_dir / "vectorizer.pkl"
        ids_path = tfidf_dir / "paper_ids.json"

        if not all(p.exists() for p in [matrix_path, vectorizer_path, ids_path]):
            logger.warning(
                "TF-IDF artifacts not found. "
                "Run scripts/build_tfidf.py first."
            )
            return False

        try:
            self._matrix = load_npz(str(matrix_path))
            with open(vectorizer_path, "rb") as f:
                self._vectorizer = pickle.load(f)
            import json
            with open(ids_path) as f:
                raw = json.load(f)
                self._paper_ids = {int(k): v for k, v in raw.items()}
            self._loaded = True
            logger.info(
                f"TF-IDF loaded: {self._matrix.shape[0]} papers, "
                f"vocab={self._matrix.shape[1]}"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to load TF-IDF: {e}")
            return False

    def search(
        self, query: str, top_k: int = 10
    ) -> Tuple[List[Dict], float]:
        """
        Returns (results, latency_ms).
        results = list of dicts with score and metadata.
        """
        if not self._loaded:
            raise RuntimeError("TF-IDF engine not loaded. Call load() first.")

        t0 = time.perf_counter()

        query_vec = self._vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self._matrix)[0]

        top_indices = np.argsort(scores)[::-1][:top_k]
        latency_ms = (time.perf_counter() - t0) * 1000

        results = []
        for idx in top_indices:
            if scores[idx] <= 0:
                break
            meta = self._paper_ids.get(int(idx), {})
            results.append({
                "paper_id": meta.get("paper_uuid", ""),
                "arxiv_id": meta.get("arxiv_id", ""),
                "title": meta.get("title", ""),
                "score": float(scores[idx]),
            })

        return results, latency_ms

    @property
    def loaded(self) -> bool:
        return self._loaded

    @property
    def corpus_size(self) -> int:
        return self._matrix.shape[0] if self._matrix is not None else 0


# ── Baseline Service (singleton) ──────────────────────────────────────────────

class BaselineService:
    _instance: Optional["BaselineService"] = None

    def __init__(self) -> None:
        self.keyword = KeywordSearchEngine()
        self.tfidf = TFIDFEngine()

    @classmethod
    def get_instance(cls) -> "BaselineService":
        if cls._instance is None:
            cls._instance = BaselineService()
        return cls._instance

    def load_all(self) -> None:
        self.tfidf.load()
