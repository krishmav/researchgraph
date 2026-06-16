"""
Module 6: Research Gap Discovery
===================================
Two complementary strategies:

Strategy A — Sparse embedding region analysis
  1. Load 2-D UMAP projection
  2. Compute KDE (kernel density estimate) on the 2-D plane
  3. Identify density minima → sparse regions
  4. Find nearest topic clusters to each sparse region
  5. Generate natural-language gap description

Strategy B — Structural graph gaps
  1. Build topic-level graph (edge weight = # papers co-occurring)
  2. Identify topic pairs with: low co-occurrence BUT high semantic similarity
  3. These represent areas where research should connect but hasn't
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import gaussian_kde
from sklearn.metrics.pairwise import cosine_similarity

from app.core.config import get_settings
from app.core.logging import logger

settings = get_settings()


# ── Gap templates ─────────────────────────────────────────────────────────────

def _describe_gap(
    flanking_topics: List[str],
    sparse_score: float,
    semantic_distance: float,
) -> str:
    """Generate a natural-language description of a research gap."""
    if len(flanking_topics) >= 2:
        t1, t2 = flanking_topics[0], flanking_topics[1]
        return (
            f"Sparse research area between '{t1}' and '{t2}'. "
            f"Despite a semantic distance of {semantic_distance:.2f}, "
            f"very few papers bridge these areas "
            f"(sparsity score: {sparse_score:.2f}). "
            f"This intersection represents a potentially under-explored "
            f"research direction."
        )
    return (
        f"Isolated sparse region near '{flanking_topics[0]}' "
        f"(sparsity score: {sparse_score:.2f}). "
        f"Few papers explore this neighbourhood."
    )


# ── Strategy A: KDE Sparse Region Analysis ───────────────────────────────────

class SparseRegionAnalyzer:
    """
    Uses kernel density estimation on 2-D UMAP projection to find sparse areas.
    """

    def __init__(
        self,
        umap_2d: np.ndarray,         # (N, 2)
        arxiv_ids: List[str],         # length-N list matching rows
        topic_assignments: Dict[str, int],   # arxiv_id → topic_id
        topic_labels: Dict[int, str],         # topic_id → label
        topic_centroids: Dict[int, np.ndarray],  # topic_id → (2,) centroid
    ) -> None:
        self.umap_2d = umap_2d
        self.arxiv_ids = arxiv_ids
        self.topic_assignments = topic_assignments
        self.topic_labels = topic_labels
        self.topic_centroids = topic_centroids

    def find_sparse_regions(
        self,
        n_grid: int = 50,
        n_gaps: int = 10,
        min_distance_from_points: float = 0.5,
    ) -> List[Dict]:
        """
        1. Fit KDE on 2-D coordinates.
        2. Build a grid over the data range.
        3. Evaluate density at each grid cell.
        4. Select low-density cells that are at least min_distance_from_points
           away from any real data point (i.e. genuinely empty, not just low-density
           due to a tight cluster).
        5. For each gap cell, find the 2–3 closest topic centroids.
        """
        if len(self.umap_2d) < 50:
            return []

        x, y = self.umap_2d[:, 0], self.umap_2d[:, 1]
        kde = gaussian_kde(np.vstack([x, y]), bw_method="scott")

        x_min, x_max = x.min() - 0.5, x.max() + 0.5
        y_min, y_max = y.min() - 0.5, y.max() + 0.5

        xi = np.linspace(x_min, x_max, n_grid)
        yi = np.linspace(y_min, y_max, n_grid)
        XX, YY = np.meshgrid(xi, yi)
        grid_points = np.vstack([XX.ravel(), YY.ravel()])

        density = kde(grid_points).reshape(n_grid, n_grid)
        flat_density = density.ravel()

        # Keep only grid cells far from actual data points
        from scipy.spatial import cKDTree
        tree = cKDTree(self.umap_2d)
        dists, _ = tree.query(grid_points.T)
        sparse_mask = dists > min_distance_from_points

        # Score sparsity: low density + far from points
        sparsity_score = (1.0 / (flat_density + 1e-8)) * sparse_mask

        # Pick top n_gaps grid cells
        top_indices = np.argsort(sparsity_score)[::-1][:n_gaps * 3]

        gaps = []
        seen_cells = set()
        for idx in top_indices:
            if not sparse_mask[idx]:
                continue

            gx = float(grid_points[0, idx])
            gy = float(grid_points[1, idx])
            cell_key = (round(gx, 1), round(gy, 1))
            if cell_key in seen_cells:
                continue
            seen_cells.add(cell_key)

            # Find closest topic centroids
            flanking = self._nearest_topics(gx, gy, n=3)
            if not flanking:
                continue

            # Semantic distance between the two nearest flanking topics
            sem_dist = 0.0
            if len(flanking) >= 2:
                tid1 = flanking[0]["topic_id"]
                tid2 = flanking[1]["topic_id"]
                c1 = self.topic_centroids.get(tid1)
                c2 = self.topic_centroids.get(tid2)
                if c1 is not None and c2 is not None:
                    sim = cosine_similarity([c1], [c2])[0][0]
                    sem_dist = float(1.0 - sim)

            raw_score = float(sparsity_score[idx])
            normalised_score = min(raw_score / (sparsity_score[sparse_mask].max() + 1e-8), 1.0)

            nearby_papers = self._nearest_papers(gx, gy, n=3)

            gaps.append({
                "gap_center": (gx, gy),
                "flanking_topics": [f["label"] for f in flanking],
                "flanking_topic_ids": [f["topic_id"] for f in flanking],
                "evidence_papers": nearby_papers,
                "sparse_score": normalised_score,
                "semantic_distance": sem_dist,
            })

            if len(gaps) >= n_gaps:
                break

        return gaps

    def _nearest_topics(self, gx: float, gy: float, n: int = 3) -> List[Dict]:
        point = np.array([gx, gy])
        results = []
        for tid, centroid in self.topic_centroids.items():
            dist = float(np.linalg.norm(centroid - point))
            results.append({
                "topic_id": tid,
                "label": self.topic_labels.get(tid, f"Topic {tid}"),
                "dist": dist,
            })
        results.sort(key=lambda x: x["dist"])
        return results[:n]

    def _nearest_papers(self, gx: float, gy: float, n: int = 3) -> List[str]:
        point = np.array([gx, gy])
        dists = np.linalg.norm(self.umap_2d - point, axis=1)
        top_idx = np.argsort(dists)[:n]
        return [self.arxiv_ids[i] for i in top_idx]


# ── Strategy B: Structural Graph Gaps ────────────────────────────────────────

class StructuralGapAnalyzer:
    """
    Finds topic pairs that are semantically similar but poorly connected
    in the paper corpus — indicating a bridge research opportunity.
    """

    def __init__(
        self,
        topic_centroids: Dict[int, np.ndarray],  # topic_id → embedding centroid
        topic_labels: Dict[int, str],
        co_occurrence_counts: Dict[Tuple[int, int], int],  # (tid1, tid2) → count
    ) -> None:
        self.centroids = topic_centroids
        self.labels = topic_labels
        self.co_occurrence = co_occurrence_counts

    def find_structural_gaps(
        self,
        n_gaps: int = 10,
        min_semantic_similarity: float = 0.4,
        max_co_occurrence: int = 5,
    ) -> List[Dict]:
        """
        Returns topic pairs with high semantic similarity but low co-occurrence.
        """
        topic_ids = list(self.centroids.keys())
        gaps = []

        for i, tid1 in enumerate(topic_ids):
            for tid2 in topic_ids[i + 1:]:
                c1 = self.centroids[tid1]
                c2 = self.centroids[tid2]
                sim = float(cosine_similarity([c1], [c2])[0][0])

                if sim < min_semantic_similarity:
                    continue

                count = self.co_occurrence.get((tid1, tid2), 0) + \
                        self.co_occurrence.get((tid2, tid1), 0)

                if count <= max_co_occurrence:
                    gaps.append({
                        "topic_id_a": tid1,
                        "topic_id_b": tid2,
                        "label_a": self.labels.get(tid1, f"Topic {tid1}"),
                        "label_b": self.labels.get(tid2, f"Topic {tid2}"),
                        "semantic_similarity": sim,
                        "co_occurrence": count,
                        "gap_score": sim * (1.0 / (count + 1)),
                    })

        gaps.sort(key=lambda x: x["gap_score"], reverse=True)
        return gaps[:n_gaps]


# ── Gap Service (singleton) ───────────────────────────────────────────────────

class GapFinderService:
    _instance: Optional["GapFinderService"] = None

    def __init__(self) -> None:
        self._sparse_gaps: List[Dict] = []
        self._structural_gaps: List[Dict] = []
        self._loaded = False

    @classmethod
    def get_instance(cls) -> "GapFinderService":
        if cls._instance is None:
            cls._instance = GapFinderService()
        return cls._instance

    def load(self) -> bool:
        """Load pre-computed gaps from disk (written by scripts/build_knowledge_graph.py)."""
        base = settings.embeddings_dir.parent
        sparse_path = base / "research_gaps_sparse.json"
        structural_path = base / "research_gaps_structural.json"

        loaded_any = False
        if sparse_path.exists():
            with open(sparse_path) as f:
                self._sparse_gaps = json.load(f)
            loaded_any = True
            logger.info(f"Loaded {len(self._sparse_gaps)} sparse research gaps")

        if structural_path.exists():
            with open(structural_path) as f:
                self._structural_gaps = json.load(f)
            loaded_any = True
            logger.info(f"Loaded {len(self._structural_gaps)} structural research gaps")

        self._loaded = loaded_any
        return loaded_any

    def get_gaps(self, strategy: str = "both") -> List[Dict]:
        if strategy == "sparse":
            return self._sparse_gaps
        if strategy == "structural":
            return self._structural_gaps
        # Merge and de-duplicate by description similarity
        combined = []
        seen_topics: set = set()
        for gap in self._sparse_gaps + self._structural_gaps:
            key = tuple(sorted(gap.get("flanking_topics", [])[:2]))
            if key not in seen_topics:
                seen_topics.add(key)
                combined.append(gap)
        combined.sort(key=lambda g: g.get("sparse_score", g.get("gap_score", 0)), reverse=True)
        return combined

    @property
    def is_loaded(self) -> bool:
        return self._loaded
