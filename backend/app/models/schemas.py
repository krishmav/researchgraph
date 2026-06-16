"""
Pydantic v2 schemas for all API request/response contracts.
"""
from __future__ import annotations

import uuid
from datetime import date, datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


# ── Shared ────────────────────────────────────────────────────────────────────

class PaperBase(BaseModel):
    arxiv_id: str
    title: str
    abstract: str
    authors: List[str]
    categories: List[str]
    primary_category: Optional[str] = None
    submitted_date: date
    doi: Optional[str] = None
    pdf_url: Optional[str] = None


class PaperResponse(PaperBase):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    topic_id: Optional[int] = None
    created_at: datetime


class PaperDetail(PaperResponse):
    """Full paper with topic info and recommendations."""
    topic_label: Optional[str] = None
    topic_top_words: Optional[List[str]] = None


# ── Search ────────────────────────────────────────────────────────────────────

RetrievalMethod = Literal["keyword", "tfidf", "miniml", "mpnet", "bge", "graph"]


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=500)
    method: RetrievalMethod = "bge"
    top_k: int = Field(default=10, ge=1, le=50)
    category_filter: Optional[str] = None
    date_from: Optional[date] = None
    date_to: Optional[date] = None


class SearchResult(BaseModel):
    paper: PaperResponse
    score: float
    explanation: str
    rank: int


class SearchResponse(BaseModel):
    query: str
    method: RetrievalMethod
    total_results: int
    results: List[SearchResult]
    latency_ms: float


class CompareSearchResponse(BaseModel):
    """Side-by-side comparison of multiple retrieval methods."""
    query: str
    methods: Dict[RetrievalMethod, List[SearchResult]]
    latency_ms: Dict[RetrievalMethod, float]


# ── Recommendations ───────────────────────────────────────────────────────────

class RecommendRequest(BaseModel):
    paper_id: uuid.UUID
    method: Literal["content", "graph"] = "graph"
    top_k: int = Field(default=10, ge=1, le=20)


class RecommendResult(BaseModel):
    paper: PaperResponse
    score: float
    explanation: str
    explanation_type: Literal[
        "semantic_similarity",
        "shared_topic",
        "shared_author",
        "graph_proximity",
        "co_citation",
    ]


class RecommendResponse(BaseModel):
    seed_paper: PaperResponse
    method: str
    recommendations: List[RecommendResult]
    latency_ms: float


# ── Topics ────────────────────────────────────────────────────────────────────

class TopicResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    label: str
    top_words: List[str]
    paper_count: int
    is_outlier: bool
    coherence_score: Optional[float] = None


class TopicTrendPoint(BaseModel):
    year_month: date
    paper_count: int


class TopicWithTrend(TopicResponse):
    trend: List[TopicTrendPoint]
    growth_slope: Optional[float] = None
    is_emerging: bool = False


class TopicMapPoint(BaseModel):
    """Single point in the 2D UMAP projection."""
    paper_id: str
    arxiv_id: str
    title: str
    x: float
    y: float
    topic_id: int
    topic_label: str


class TopicMapResponse(BaseModel):
    points: List[TopicMapPoint]
    topics: List[TopicResponse]
    total_papers: int


# ── Knowledge Graph ───────────────────────────────────────────────────────────

class GraphNode(BaseModel):
    id: str
    label: str
    node_type: Literal["paper", "author", "topic", "area"]
    size: float = 10.0
    color: str = "#6366f1"
    metadata: Dict[str, Any] = Field(default_factory=dict)


class GraphEdge(BaseModel):
    source: str
    target: str
    weight: float = 1.0
    edge_type: str


class GraphResponse(BaseModel):
    nodes: List[GraphNode]
    edges: List[GraphEdge]
    stats: Dict[str, Any]


class CentralityResponse(BaseModel):
    paper_id: str
    arxiv_id: str
    title: str
    pagerank: float
    betweenness: float
    degree: int
    community_id: int


# ── Research Gaps ─────────────────────────────────────────────────────────────

class ResearchGap(BaseModel):
    gap_id: int
    description: str
    flanking_topics: List[str]
    evidence_papers: List[str]       # arxiv_ids of nearest papers
    sparse_score: float              # 0-1, higher = more sparse/underexplored
    semantic_distance: float         # distance between flanking topic centroids


class GapResponse(BaseModel):
    gaps: List[ResearchGap]
    methodology: str
    total_gaps_found: int


# ── Evaluation ────────────────────────────────────────────────────────────────

class MetricRow(BaseModel):
    method: RetrievalMethod
    precision_at_5: float
    precision_at_10: float
    recall_at_10: float
    mrr: float
    ndcg_at_10: float
    latency_p50_ms: float
    latency_p95_ms: float
    memory_mb: float


class AblationRow(BaseModel):
    alpha: float
    beta: float
    gamma: float
    ndcg_at_10: float
    mrr: float


class SignificanceRow(BaseModel):
    method_a: str
    method_b: str
    p_value: float
    significant: bool


class EvaluationResponse(BaseModel):
    benchmark: List[MetricRow]
    ablation: List[AblationRow]
    significance: List[SignificanceRow]
    num_queries: int
    corpus_size: int
    methodology_note: str


# ── Pagination / Health ───────────────────────────────────────────────────────

class PaginatedResponse(BaseModel):
    items: List[Any]
    total: int
    page: int
    page_size: int
    pages: int


class HealthResponse(BaseModel):
    status: str
    version: str
    db_connected: bool
    faiss_loaded: bool
    models_loaded: List[str]
    corpus_size: int
