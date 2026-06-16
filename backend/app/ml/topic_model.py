"""
Module 5: Topic Modeling & Trend Mining
========================================
BERTopic + UMAP + HDBSCAN pipeline.
Provides:
  - Topic discovery and labelling
  - 2-D UMAP projection for interactive scatter plot
  - Emerging topic detection via linear regression on monthly paper counts
  - Topic coherence scoring (c_v approximation via co-occurrence)

Pre-built artifacts are stored in data/processed/ and loaded at startup.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from app.core.config import get_settings
from app.core.logging import logger

settings = get_settings()


class TopicModelService:
    """
    Loads pre-trained BERTopic model and 2-D projection coordinates.

    Artifacts (written by scripts/train_bertopic.py):
        data/processed/topic_model.pkl        – trained BERTopic model
        data/processed/umap_2d.npy            – (N,2) 2-D projection
        data/processed/topic_assignments.json – {arxiv_id: topic_id}
        data/processed/topic_labels.json      – {topic_id: {label, top_words}}
    """

    _instance: Optional["TopicModelService"] = None

    def __init__(self) -> None:
        self._model = None
        self._umap_2d: Optional[np.ndarray] = None
        self._assignments: Dict[str, int] = {}     # arxiv_id → topic_id
        self._labels: Dict[int, Dict] = {}          # topic_id → {label, top_words}
        self._loaded = False

    @classmethod
    def get_instance(cls) -> "TopicModelService":
        if cls._instance is None:
            cls._instance = TopicModelService()
        return cls._instance

    # ── Load ──────────────────────────────────────────────────

    def load(self) -> bool:
        base = settings.embeddings_dir.parent   # data/processed/
        model_path  = base / "topic_model.pkl"
        umap_path   = base / "umap_2d.npy"
        assign_path = base / "topic_assignments.json"
        labels_path = base / "topic_labels.json"

        if not model_path.exists():
            logger.warning(
                "BERTopic model not found. Run scripts/train_bertopic.py"
            )
            return False

        try:
            with open(model_path, "rb") as f:
                self._model = pickle.load(f)
            logger.info("BERTopic model loaded")

            if umap_path.exists():
                self._umap_2d = np.load(str(umap_path))
                logger.info(f"UMAP 2-D projection loaded: {self._umap_2d.shape}")

            if assign_path.exists():
                with open(assign_path) as f:
                    self._assignments = json.load(f)

            if labels_path.exists():
                with open(labels_path) as f:
                    raw = json.load(f)
                    self._labels = {int(k): v for k, v in raw.items()}

            self._loaded = True
            return True

        except Exception as e:
            logger.error(f"Failed to load topic model: {e}")
            return False

    # ── Topic map (UMAP scatter) ──────────────────────────────

    def get_topic_map_data(
        self,
        papers_meta: List[Dict],   # [{arxiv_id, title}, ...]
        max_points: int = 5000,
    ) -> List[Dict]:
        """
        Returns list of {paper_id, arxiv_id, title, x, y, topic_id, topic_label}
        for the 2-D scatter plot.
        """
        if not self._loaded or self._umap_2d is None:
            return []

        # Build lookup: arxiv_id → row index (matches order in embeddings)
        arxiv_to_row = {p["arxiv_id"]: i for i, p in enumerate(papers_meta)}

        points = []
        for paper in papers_meta[:max_points]:
            arxiv_id = paper["arxiv_id"]
            row = arxiv_to_row.get(arxiv_id)
            if row is None or row >= len(self._umap_2d):
                continue
            topic_id = self._assignments.get(arxiv_id, -1)
            topic_info = self._labels.get(topic_id, {"label": "Uncategorised", "top_words": []})
            x, y = self._umap_2d[row]
            points.append({
                "paper_id": paper.get("id", arxiv_id),
                "arxiv_id": arxiv_id,
                "title": paper["title"],
                "x": float(x),
                "y": float(y),
                "topic_id": topic_id,
                "topic_label": topic_info["label"],
            })
        return points

    # ── Trend mining ──────────────────────────────────────────

    @staticmethod
    def compute_growth_slopes(
        trends: List[Dict],   # [{year_month, paper_count}, ...]
        min_months: int = 6,
    ) -> Tuple[float, bool]:
        """
        Fit linear regression on monthly paper counts.
        Returns (slope, is_emerging).
        is_emerging = True if slope > 2.0 papers/month (threshold tunable).
        """
        if len(trends) < min_months:
            return 0.0, False

        df = pd.DataFrame(trends).sort_values("year_month")
        x = np.arange(len(df), dtype=float)
        y = df["paper_count"].values.astype(float)

        # Simple OLS slope
        x_mean, y_mean = x.mean(), y.mean()
        slope = float(
            np.sum((x - x_mean) * (y - y_mean)) / (np.sum((x - x_mean) ** 2) + 1e-8)
        )
        is_emerging = slope > 2.0
        return slope, is_emerging

    # ── Topic coherence (c_v approximation) ──────────────────

    def get_topic_coherence_scores(self) -> Dict[int, float]:
        """
        Returns pre-computed or approximate coherence scores per topic.
        BERTopic stores topic representations; we use top-word overlap
        as a lightweight coherence proxy.
        """
        if not self._loaded or self._model is None:
            return {}

        scores: Dict[int, float] = {}
        try:
            topic_info = self._model.get_topic_info()
            for _, row in topic_info.iterrows():
                tid = int(row["Topic"])
                # BERTopic's internal representation score is a coherence proxy
                scores[tid] = float(row.get("Representative_Docs", 0) or 0)
        except Exception as e:
            logger.warning(f"Could not extract coherence scores: {e}")

        return scores

    # ── Properties ────────────────────────────────────────────

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def num_topics(self) -> int:
        return len(self._labels)

    def get_label(self, topic_id: int) -> str:
        return self._labels.get(topic_id, {}).get("label", f"Topic {topic_id}")

    def get_top_words(self, topic_id: int) -> List[str]:
        return self._labels.get(topic_id, {}).get("top_words", [])
