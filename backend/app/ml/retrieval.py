"""
Module 1: Semantic Retrieval Engine
=====================================
Implements dense retrieval using Sentence Transformers + FAISS.
Supports three models: MiniLM, MPNet, BGE.
Provides cosine-similarity ranking with optional graph re-ranking.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from app.core.config import get_settings
from app.core.logging import logger

settings = get_settings()


# ── Model registry ────────────────────────────────────────────────────────────

MODEL_REGISTRY: Dict[str, str] = {
    "miniml": "sentence-transformers/all-MiniLM-L6-v2",
    "mpnet":  "sentence-transformers/all-mpnet-base-v2",
    "bge":    "BAAI/bge-large-en-v1.5",
}

# BGE requires an instruction prefix for retrieval tasks
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class EmbeddingModel:
    """Wrapper around SentenceTransformer with lazy loading and caching."""

    def __init__(self, model_key: str) -> None:
        if model_key not in MODEL_REGISTRY:
            raise ValueError(
                f"Unknown model key '{model_key}'. "
                f"Available: {list(MODEL_REGISTRY.keys())}"
            )
        self.model_key = model_key
        self.model_name = MODEL_REGISTRY[model_key]
        self._model: Optional[SentenceTransformer] = None
        self.device = settings.torch_device

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            logger.info(f"Loading embedding model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name, device=self.device)
            logger.info(f"Model loaded on device={self.device}")
        return self._model

    def encode(
        self,
        texts: List[str],
        batch_size: int = 64,
        is_query: bool = False,
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Encode texts into L2-normalized embeddings.
        BGE models need a query prefix for retrieval queries.
        """
        if self.model_key == "bge" and is_query:
            texts = [BGE_QUERY_PREFIX + t for t in texts]

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,   # L2-normalize → cosine sim = dot product
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)

    @property
    def dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()


# ── FAISS Index Manager ───────────────────────────────────────────────────────

class FAISSIndex:
    """
    Manages a FAISS IndexFlatIP (exact cosine similarity via dot product on
    L2-normalized vectors).

    File layout (one set per model):
        data/faiss/{model_key}_index.faiss
        data/faiss/{model_key}_metadata.json   # {faiss_row: {arxiv_id, paper_uuid}}
    """

    def __init__(self, model_key: str) -> None:
        self.model_key = model_key
        self.index_path = settings.faiss_dir / f"{model_key}_index.faiss"
        self.meta_path  = settings.faiss_dir / f"{model_key}_metadata.json"
        self._index: Optional[faiss.IndexFlatIP] = None
        self._metadata: Optional[Dict[int, Dict]] = None

    # ── Persistence ───────────────────────────────────────────

    def build(self, embeddings: np.ndarray, metadata: List[Dict]) -> None:
        """Build and save FAISS index from embeddings array."""
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)

        dim = embeddings.shape[1]
        self._index = faiss.IndexFlatIP(dim)
        self._index.add(embeddings)

        self._metadata = {i: m for i, m in enumerate(metadata)}
        self._save()
        logger.info(
            f"Built FAISS index: model={self.model_key}, "
            f"vectors={self._index.ntotal}, dim={dim}"
        )

    def _save(self) -> None:
        settings.faiss_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(self.index_path))
        with open(self.meta_path, "w") as f:
            json.dump(self._metadata, f)

    def load(self) -> bool:
        """Load index from disk. Returns True if successful."""
        if not self.index_path.exists() or not self.meta_path.exists():
            logger.warning(f"FAISS index not found for model={self.model_key}")
            return False
        try:
            self._index = faiss.read_index(str(self.index_path))
            with open(self.meta_path) as f:
                raw = json.load(f)
                self._metadata = {int(k): v for k, v in raw.items()}
            logger.info(
                f"Loaded FAISS index: model={self.model_key}, "
                f"vectors={self._index.ntotal}"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to load FAISS index: {e}")
            return False

    # ── Search ────────────────────────────────────────────────

    def search(
        self, query_embedding: np.ndarray, top_k: int = 10
    ) -> List[Tuple[float, Dict]]:
        """
        Returns list of (score, metadata_dict) sorted by descending similarity.
        """
        if self._index is None:
            raise RuntimeError("FAISS index not loaded. Call load() first.")

        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        if query_embedding.dtype != np.float32:
            query_embedding = query_embedding.astype(np.float32)

        k = min(top_k, self._index.ntotal)
        scores, indices = self._index.search(query_embedding, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            meta = self._metadata.get(idx, {})
            results.append((float(score), meta))
        return results

    def get_embedding_by_row(self, row_idx: int) -> Optional[np.ndarray]:
        """Retrieve a stored embedding vector by its FAISS row index."""
        if self._index is None or row_idx >= self._index.ntotal:
            return None
        vec = np.zeros((1, self._index.d), dtype=np.float32)
        self._index.reconstruct(row_idx, vec[0])
        return vec[0]

    @property
    def size(self) -> int:
        return self._index.ntotal if self._index else 0

    @property
    def loaded(self) -> bool:
        return self._index is not None


# ── Retrieval Engine ─────────────────────────────────────────────────────────

class SemanticRetrievalEngine:
    """
    Orchestrates query encoding → FAISS search → result assembly.

    One instance per model; the RetrievalService (below) manages
    the pool of engines.
    """

    def __init__(self, model_key: str) -> None:
        self.model_key = model_key
        self.embedding_model = EmbeddingModel(model_key)
        self.faiss_index = FAISSIndex(model_key)

    def is_ready(self) -> bool:
        return self.faiss_index.loaded

    def initialize(self) -> bool:
        return self.faiss_index.load()

    def search(
        self,
        query: str,
        top_k: int = 10,
    ) -> Tuple[List[Tuple[float, Dict]], float]:
        """
        Returns (results, latency_ms).
        results = list of (cosine_score, metadata_dict).
        """
        t0 = time.perf_counter()
        query_emb = self.embedding_model.encode([query], is_query=True)
        results = self.faiss_index.search(query_emb, top_k=top_k)
        latency_ms = (time.perf_counter() - t0) * 1000
        return results, latency_ms

    def encode_paper(self, text: str) -> np.ndarray:
        """Encode a single paper abstract for similarity lookup."""
        return self.embedding_model.encode([text], is_query=False)[0]


# ── Retrieval Service (singleton) ─────────────────────────────────────────────

class RetrievalService:
    """
    Singleton service that holds all loaded engines.
    FastAPI lifespan event loads engines at startup.
    """
    _instance: Optional["RetrievalService"] = None

    def __init__(self) -> None:
        self._engines: Dict[str, SemanticRetrievalEngine] = {}
        self._loaded_models: List[str] = []

    @classmethod
    def get_instance(cls) -> "RetrievalService":
        if cls._instance is None:
            cls._instance = RetrievalService()
        return cls._instance

    def load_all(self) -> None:
        """Try to load all available model indices from disk."""
        for model_key in MODEL_REGISTRY:
            engine = SemanticRetrievalEngine(model_key)
            if engine.initialize():
                self._engines[model_key] = engine
                self._loaded_models.append(model_key)
                logger.info(f"Retrieval engine ready: {model_key}")
            else:
                logger.warning(
                    f"Retrieval engine not available: {model_key} "
                    f"(run scripts/generate_embeddings.py first)"
                )

    def get_engine(self, model_key: str) -> Optional[SemanticRetrievalEngine]:
        return self._engines.get(model_key)

    def search(
        self,
        query: str,
        model_key: str,
        top_k: int = 10,
    ) -> Tuple[List[Tuple[float, Dict]], float]:
        engine = self.get_engine(model_key)
        if engine is None:
            raise RuntimeError(
                f"Model '{model_key}' is not loaded. "
                f"Available: {self._loaded_models}"
            )
        return engine.search(query, top_k=top_k)

    @property
    def loaded_models(self) -> List[str]:
        return list(self._loaded_models)

    @property
    def any_loaded(self) -> bool:
        return len(self._loaded_models) > 0
