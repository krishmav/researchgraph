"""
Remaining API Routers
  /api/recommend  — paper recommendations
  /api/topics     — topic map, trends, emerging topics
  /api/graph      — knowledge graph queries
  /api/gaps       — research gap discovery
  /api/evaluate   — evaluation framework
"""
from __future__ import annotations

import uuid
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.logging import logger
from app.ml.gap_finder import GapFinderService
from app.ml.knowledge_graph import KnowledgeGraphService
from app.ml.recommender import RecommendationEngine
from app.ml.retrieval import RetrievalService
from app.ml.baselines import BaselineService
from app.ml.topic_model import TopicModelService
from app.ml.evaluation import run_full_benchmark
from app.models.orm import Paper, Topic, TopicTrend
from app.models.schemas import (
    EvaluationResponse,
    GapResponse,
    GraphResponse,
    RecommendRequest,
    RecommendResponse,
    RecommendResult,
    ResearchGap,
    TopicMapResponse,
    TopicResponse,
    TopicTrendPoint,
    TopicWithTrend,
    PaperResponse,
)

# ══════════════════════════════════════════════════════════════════════════════
# RECOMMENDATIONS
# ══════════════════════════════════════════════════════════════════════════════

recommend_router = APIRouter(prefix="/api/recommend", tags=["recommendations"])


@recommend_router.post("", response_model=RecommendResponse)
async def get_recommendations(
    req: RecommendRequest,
    session: AsyncSession = Depends(get_db),
):
    result = await session.execute(
        select(Paper).where(Paper.id == req.paper_id)
    )
    seed = result.scalar_one_or_none()
    if not seed:
        raise HTTPException(404, f"Paper {req.paper_id} not found.")

    retrieval = RetrievalService.get_instance()
    kg = KnowledgeGraphService.get_instance()
    engine = RecommendationEngine(retrieval, kg)

    if req.method == "graph":
        recs, latency_ms = await engine.graph_enhanced(seed, session, req.top_k)
    else:
        recs, latency_ms = await engine.content_based(seed, session, req.top_k)

    return RecommendResponse(
        seed_paper=PaperResponse.model_validate(seed),
        method=req.method,
        recommendations=[
            RecommendResult(
                paper=PaperResponse.model_validate(r["paper"]),
                score=r["score"],
                explanation=r["explanation"],
                explanation_type=r["explanation_type"],
            )
            for r in recs
        ],
        latency_ms=round(latency_ms, 2),
    )


# ══════════════════════════════════════════════════════════════════════════════
# TOPICS
# ══════════════════════════════════════════════════════════════════════════════

topics_router = APIRouter(prefix="/api/topics", tags=["topics"])


@topics_router.get("", response_model=list)
async def list_topics(
    include_outliers: bool = False,
    session: AsyncSession = Depends(get_db),
):
    query = select(Topic).order_by(Topic.paper_count.desc())
    if not include_outliers:
        query = query.where(Topic.is_outlier.is_(False))
    result = await session.execute(query)
    topics = result.scalars().all()
    return [TopicResponse.model_validate(t) for t in topics]


@topics_router.get("/map", response_model=TopicMapResponse)
async def get_topic_map(
    max_points: int = Query(3000, ge=100, le=10000),
    session: AsyncSession = Depends(get_db),
):
    topic_svc = TopicModelService.get_instance()
    if not topic_svc.is_loaded:
        raise HTTPException(503, "Topic model not loaded. Run scripts/train_bertopic.py")

    result = await session.execute(
        select(Paper.id, Paper.arxiv_id, Paper.title).limit(max_points)
    )
    papers_meta = [
        {"id": str(row.id), "arxiv_id": row.arxiv_id, "title": row.title}
        for row in result.fetchall()
    ]

    points = topic_svc.get_topic_map_data(papers_meta, max_points=max_points)

    topics_result = await session.execute(
        select(Topic).where(Topic.is_outlier.is_(False))
    )
    topics = [TopicResponse.model_validate(t) for t in topics_result.scalars().all()]

    return TopicMapResponse(
        points=points,
        topics=topics,
        total_papers=len(papers_meta),
    )


@topics_router.get("/trending", response_model=list)
async def get_trending_topics(
    min_months: int = 6,
    session: AsyncSession = Depends(get_db),
):
    result = await session.execute(
        select(Topic).where(Topic.is_outlier.is_(False)).order_by(Topic.paper_count.desc())
    )
    topics = result.scalars().all()

    trending = []
    for topic in topics:
        trend_result = await session.execute(
            select(TopicTrend)
            .where(TopicTrend.topic_id == topic.id)
            .order_by(TopicTrend.year_month)
        )
        trend_rows = trend_result.scalars().all()
        trend_data = [
            {"year_month": t.year_month, "paper_count": t.paper_count}
            for t in trend_rows
        ]
        slope, is_emerging = TopicModelService.compute_growth_slopes(trend_data, min_months)

        trending.append(
            TopicWithTrend(
                **TopicResponse.model_validate(topic).model_dump(),
                trend=[TopicTrendPoint(**td) for td in trend_data],
                growth_slope=slope,
                is_emerging=is_emerging,
            )
        )

    trending.sort(key=lambda t: t.growth_slope or 0, reverse=True)
    return trending


@topics_router.get("/{topic_id}", response_model=TopicWithTrend)
async def get_topic(
    topic_id: int,
    session: AsyncSession = Depends(get_db),
):
    result = await session.execute(select(Topic).where(Topic.id == topic_id))
    topic = result.scalar_one_or_none()
    if not topic:
        raise HTTPException(404, f"Topic {topic_id} not found.")

    trend_result = await session.execute(
        select(TopicTrend)
        .where(TopicTrend.topic_id == topic_id)
        .order_by(TopicTrend.year_month)
    )
    trend_rows = trend_result.scalars().all()
    trend_data = [
        {"year_month": t.year_month, "paper_count": t.paper_count}
        for t in trend_rows
    ]
    slope, is_emerging = TopicModelService.compute_growth_slopes(trend_data)

    return TopicWithTrend(
        **TopicResponse.model_validate(topic).model_dump(),
        trend=[TopicTrendPoint(**td) for td in trend_data],
        growth_slope=slope,
        is_emerging=is_emerging,
    )


# ══════════════════════════════════════════════════════════════════════════════
# KNOWLEDGE GRAPH
# ══════════════════════════════════════════════════════════════════════════════

graph_router = APIRouter(prefix="/api/graph", tags=["knowledge-graph"])


@graph_router.get("/paper/{arxiv_id}", response_model=GraphResponse)
async def get_paper_graph(
    arxiv_id: str,
    radius: int = Query(2, ge=1, le=3),
    max_nodes: int = Query(60, ge=10, le=120),
):
    kg = KnowledgeGraphService.get_instance()
    if not kg.is_ready:
        raise HTTPException(503, "Knowledge graph not loaded. Run scripts/build_knowledge_graph.py")

    nodes, edges = kg.get_nodes_and_edges_for_api(arxiv_id, radius, max_nodes)
    if not nodes:
        raise HTTPException(404, f"Paper '{arxiv_id}' not found in knowledge graph.")

    stats = {
        "total_nodes": len(nodes),
        "total_edges": len(edges),
        "center": arxiv_id,
        "radius": radius,
    }
    return GraphResponse(nodes=nodes, edges=edges, stats=stats)


@graph_router.get("/paper/{arxiv_id}/html")
async def get_paper_graph_html(
    arxiv_id: str,
    radius: int = Query(2, ge=1, le=3),
):
    """Return interactive PyVis HTML for embedding in an iframe."""
    from fastapi.responses import HTMLResponse
    kg = KnowledgeGraphService.get_instance()
    if not kg.is_ready:
        return HTMLResponse("<p>Knowledge graph not loaded.</p>")
    html = kg.render_paper_subgraph(arxiv_id, radius=radius)
    return HTMLResponse(html)


@graph_router.get("/stats")
async def get_graph_stats():
    kg = KnowledgeGraphService.get_instance()
    return kg.get_graph_stats()


@graph_router.get("/top-papers")
async def get_top_papers_by_pagerank(
    n: int = Query(20, ge=5, le=100),
    session: AsyncSession = Depends(get_db),
):
    kg = KnowledgeGraphService.get_instance()
    if not kg.is_ready:
        raise HTTPException(503, "Knowledge graph not loaded.")

    top = kg.get_top_papers_by_pagerank(n=n)
    arxiv_ids = [item["arxiv_id"] for item in top]
    result = await session.execute(
        select(Paper).where(Paper.arxiv_id.in_(arxiv_ids))
    )
    paper_map = {p.arxiv_id: p for p in result.scalars().all()}

    enriched = []
    for item in top:
        paper = paper_map.get(item["arxiv_id"])
        if paper:
            enriched.append({
                **item,
                "title": paper.title,
                "categories": paper.categories,
                "submitted_date": paper.submitted_date.isoformat(),
            })
    return enriched


# ══════════════════════════════════════════════════════════════════════════════
# RESEARCH GAPS
# ══════════════════════════════════════════════════════════════════════════════

gaps_router = APIRouter(prefix="/api/gaps", tags=["research-gaps"])


@gaps_router.get("", response_model=GapResponse)
async def get_research_gaps(
    strategy: str = Query("both", regex="^(both|sparse|structural)$"),
    limit: int = Query(10, ge=1, le=30),
):
    gap_svc = GapFinderService.get_instance()
    if not gap_svc.is_loaded:
        raise HTTPException(
            503,
            "Research gap analysis not available. "
            "Run scripts/build_knowledge_graph.py to compute gaps."
        )

    raw_gaps = gap_svc.get_gaps(strategy)[:limit]

    formatted = []
    for i, g in enumerate(raw_gaps):
        formatted.append(
            ResearchGap(
                gap_id=i + 1,
                description=g.get("description", "") or _auto_describe(g),
                flanking_topics=g.get("flanking_topics", []),
                evidence_papers=g.get("evidence_papers", []),
                sparse_score=float(g.get("sparse_score", g.get("gap_score", 0.0))),
                semantic_distance=float(g.get("semantic_distance", 0.0)),
            )
        )

    return GapResponse(
        gaps=formatted,
        methodology=(
            "Gaps identified via two methods: "
            "(A) Kernel density estimation on 2-D UMAP projection — sparse regions "
            "with few nearby papers are flagged as under-explored. "
            "(B) Topic pairs with high semantic centroid similarity but low "
            "co-occurrence in the paper corpus."
        ),
        total_gaps_found=len(raw_gaps),
    )


def _auto_describe(gap: dict) -> str:
    topics = gap.get("flanking_topics", ["Unknown"])
    score = gap.get("sparse_score", gap.get("gap_score", 0.0))
    if len(topics) >= 2:
        return (
            f"Sparse research area between '{topics[0]}' and '{topics[1]}'. "
            f"Sparsity score: {score:.2f}."
        )
    return f"Under-explored area near '{topics[0]}'. Sparsity score: {score:.2f}."


# ══════════════════════════════════════════════════════════════════════════════
# EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

evaluate_router = APIRouter(prefix="/api/evaluate", tags=["evaluation"])


@evaluate_router.post("", response_model=EvaluationResponse)
async def run_evaluation(
    num_queries: int = Query(100, ge=10, le=500),
    session: AsyncSession = Depends(get_db),
):
    """
    Run full benchmark evaluation across all loaded retrieval methods.
    May take several minutes depending on corpus size and loaded models.
    """
    retrieval = RetrievalService.get_instance()
    baselines = BaselineService.get_instance()
    kg = KnowledgeGraphService.get_instance()

    results = await run_full_benchmark(
        session=session,
        retrieval_svc=retrieval,
        tfidf_engine=baselines.tfidf,
        kg_svc=kg,
        num_queries=num_queries,
    )

    from app.models.schemas import MetricRow, AblationRow, SignificanceRow
    return EvaluationResponse(
        benchmark=[MetricRow(**r) for r in results["benchmark"]],
        ablation=[AblationRow(**r) for r in results["ablation"]],
        significance=[SignificanceRow(**r) for r in results["significance"]],
        num_queries=results["num_queries"],
        corpus_size=results["corpus_size"],
        methodology_note=results["methodology_note"],
    )


@evaluate_router.get("/cached", response_model=EvaluationResponse)
async def get_cached_evaluation():
    """Return last saved evaluation results from disk (if available)."""
    import json
    from pathlib import Path

    results_path = Path("../research/results/benchmark_table.json")
    if not results_path.exists():
        raise HTTPException(
            404,
            "No cached evaluation results found. POST /api/evaluate to run benchmark."
        )

    with open(results_path) as f:
        data = json.load(f)

    from app.models.schemas import MetricRow, AblationRow, SignificanceRow
    return EvaluationResponse(
        benchmark=[MetricRow(**r) for r in data.get("benchmark", [])],
        ablation=[AblationRow(**r) for r in data.get("ablation", [])],
        significance=[SignificanceRow(**r) for r in data.get("significance", [])],
        num_queries=data.get("num_queries", 0),
        corpus_size=data.get("corpus_size", 0),
        methodology_note=data.get("methodology_note", ""),
    )
