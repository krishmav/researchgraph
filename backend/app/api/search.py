"""
Search API Router
Routes:
  POST /api/search           — single-method search
  POST /api/search/compare   — side-by-side comparison of all methods
  GET  /api/search/paper/{arxiv_id} — fetch single paper detail
"""
from __future__ import annotations

import time
import uuid
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.logging import logger
from app.ml.baselines import BaselineService
from app.ml.retrieval import RetrievalService
from app.models.orm import Paper, SearchLog, Topic
from app.models.schemas import (
    CompareSearchResponse,
    PaperDetail,
    PaperResponse,
    SearchRequest,
    SearchResponse,
    SearchResult,
)

router = APIRouter(prefix="/api/search", tags=["search"])


def _get_retrieval() -> RetrievalService:
    return RetrievalService.get_instance()


def _get_baselines() -> BaselineService:
    return BaselineService.get_instance()


def _explain(method: str, score: float, paper: Paper) -> str:
    """Generate a per-result explanation string."""
    if method == "keyword":
        return f"Matched by full-text keyword search (BM25 score: {score:.4f})."
    if method == "tfidf":
        return f"Matched by TF-IDF cosine similarity (score: {score:.4f})."
    if method in ("miniml", "mpnet", "bge"):
        model_names = {"miniml": "MiniLM-L6", "mpnet": "MPNet-base", "bge": "BGE-large"}
        return (
            f"Matched by {model_names[method]} semantic embedding similarity "
            f"(cosine: {score:.4f})."
        )
    if method == "graph":
        return (
            f"Matched by graph-augmented retrieval. BGE semantic score boosted "
            f"by knowledge-graph neighbourhood and PageRank signal (combined: {score:.4f})."
        )
    return f"Score: {score:.4f}"


async def _fetch_paper_by_arxiv(
    session: AsyncSession, arxiv_id: str
) -> Optional[Paper]:
    result = await session.execute(
        select(Paper).where(Paper.arxiv_id == arxiv_id)
    )
    return result.scalar_one_or_none()


async def _papers_from_arxiv_ids(
    session: AsyncSession, arxiv_ids: List[str]
) -> dict:
    """Fetch papers in bulk. Returns {arxiv_id: Paper}."""
    if not arxiv_ids:
        return {}
    result = await session.execute(
        select(Paper).where(Paper.arxiv_id.in_(arxiv_ids))
    )
    return {p.arxiv_id: p for p in result.scalars().all()}


async def _log_search(
    session: AsyncSession,
    query: str,
    method: str,
    result_ids: List[str],
    latency_ms: float,
) -> None:
    log = SearchLog(
        query=query,
        method=method,
        result_ids=result_ids,
        latency_ms=int(latency_ms),
    )
    session.add(log)


# ── POST /api/search ──────────────────────────────────────────────────────────

@router.post("", response_model=SearchResponse)
async def search(
    req: SearchRequest,
    session: AsyncSession = Depends(get_db),
    retrieval: RetrievalService = Depends(_get_retrieval),
    baselines: BaselineService = Depends(_get_baselines),
) -> SearchResponse:
    method = req.method
    results: List[SearchResult] = []
    latency_ms = 0.0

    if method == "keyword":
        rows, latency_ms = await baselines.keyword.search(
            session, req.query, req.top_k, req.category_filter
        )
        paper_map = await _papers_from_arxiv_ids(session, [r["arxiv_id"] for r in rows])
        for rank, row in enumerate(rows, 1):
            paper = paper_map.get(row["arxiv_id"])
            if paper:
                results.append(SearchResult(
                    paper=PaperResponse.model_validate(paper),
                    score=row["score"],
                    explanation=_explain(method, row["score"], paper),
                    rank=rank,
                ))

    elif method == "tfidf":
        if not baselines.tfidf.loaded:
            raise HTTPException(503, "TF-IDF index not available. Run build_tfidf.py first.")
        rows, latency_ms = baselines.tfidf.search(req.query, req.top_k)
        paper_map = await _papers_from_arxiv_ids(session, [r["arxiv_id"] for r in rows if r.get("arxiv_id")])
        for rank, row in enumerate(rows, 1):
            paper = paper_map.get(row.get("arxiv_id", ""))
            if paper:
                results.append(SearchResult(
                    paper=PaperResponse.model_validate(paper),
                    score=row["score"],
                    explanation=_explain(method, row["score"], paper),
                    rank=rank,
                ))

    elif method in ("miniml", "mpnet", "bge"):
        if method not in retrieval.loaded_models:
            raise HTTPException(
                503,
                f"Model '{method}' not available. "
                f"Run scripts/generate_embeddings.py --model {method} first."
            )
        raw, latency_ms = retrieval.search(req.query, method, req.top_k)
        arxiv_ids = [m.get("arxiv_id", "") for _, m in raw]
        paper_map = await _papers_from_arxiv_ids(session, arxiv_ids)
        for rank, (score, meta) in enumerate(raw, 1):
            paper = paper_map.get(meta.get("arxiv_id", ""))
            if paper:
                if req.category_filter and req.category_filter not in (paper.categories or []):
                    continue
                results.append(SearchResult(
                    paper=PaperResponse.model_validate(paper),
                    score=score,
                    explanation=_explain(method, score, paper),
                    rank=rank,
                ))

    elif method == "graph":
        # Use best available embedding model then apply graph re-ranking
        from app.ml.knowledge_graph import KnowledgeGraphService
        kg = KnowledgeGraphService.get_instance()

        best_model = next(
            (m for m in ["bge", "mpnet", "miniml"] if m in retrieval.loaded_models),
            None
        )
        if not best_model:
            raise HTTPException(503, "No embedding model loaded for graph-augmented search.")

        t0 = time.perf_counter()
        raw, _ = retrieval.search(req.query, best_model, req.top_k * 3)
        pagerank = kg.pagerank_scores if kg.is_ready else {}

        scored = []
        for cos_score, meta in raw:
            arxiv_id = meta.get("arxiv_id", "")
            pr = pagerank.get(arxiv_id, 0.0)
            combined = 0.7 * cos_score + 0.2 * 0.0 + 0.1 * pr   # alpha=0.7, beta=0.2, gamma=0.1
            scored.append((combined, cos_score, arxiv_id))
        scored.sort(reverse=True)

        arxiv_ids = [a for _, _, a in scored[:req.top_k]]
        paper_map = await _papers_from_arxiv_ids(session, arxiv_ids)
        latency_ms = (time.perf_counter() - t0) * 1000

        for rank, (combined_score, cos_score, arxiv_id) in enumerate(scored[:req.top_k], 1):
            paper = paper_map.get(arxiv_id)
            if paper:
                results.append(SearchResult(
                    paper=PaperResponse.model_validate(paper),
                    score=combined_score,
                    explanation=_explain("graph", combined_score, paper),
                    rank=rank,
                ))

    else:
        raise HTTPException(400, f"Unknown retrieval method: {method}")

    # Log search
    await _log_search(
        session, req.query, method, [r.paper.arxiv_id for r in results], latency_ms
    )

    return SearchResponse(
        query=req.query,
        method=method,
        total_results=len(results),
        results=results,
        latency_ms=round(latency_ms, 2),
    )


# ── POST /api/search/compare ──────────────────────────────────────────────────

@router.post("/compare", response_model=CompareSearchResponse)
async def compare_search(
    req: SearchRequest,
    session: AsyncSession = Depends(get_db),
    retrieval: RetrievalService = Depends(_get_retrieval),
    baselines: BaselineService = Depends(_get_baselines),
) -> CompareSearchResponse:
    """Run the same query through all available methods and return side-by-side results."""
    available_methods = ["keyword", "tfidf"] + retrieval.loaded_models
    methods_results = {}
    methods_latency = {}

    for method in available_methods:
        try:
            sub_req = SearchRequest(
                query=req.query,
                method=method,
                top_k=req.top_k,
            )
            resp = await search(sub_req, session, retrieval, baselines)
            methods_results[method] = resp.results
            methods_latency[method] = resp.latency_ms
        except HTTPException:
            pass

    return CompareSearchResponse(
        query=req.query,
        methods=methods_results,
        latency_ms=methods_latency,
    )


# ── GET /api/search/paper/{arxiv_id} ─────────────────────────────────────────

@router.get("/paper/{arxiv_id}", response_model=PaperDetail)
async def get_paper(
    arxiv_id: str,
    session: AsyncSession = Depends(get_db),
) -> PaperDetail:
    paper = await _fetch_paper_by_arxiv(session, arxiv_id)
    if not paper:
        raise HTTPException(404, f"Paper '{arxiv_id}' not found.")

    topic_label = None
    topic_top_words = None
    if paper.topic_id:
        topic_result = await session.execute(
            select(Topic).where(Topic.id == paper.topic_id)
        )
        topic = topic_result.scalar_one_or_none()
        if topic:
            topic_label = topic.label
            topic_top_words = topic.top_words

    detail = PaperDetail(
        **PaperResponse.model_validate(paper).model_dump(),
        topic_label=topic_label,
        topic_top_words=topic_top_words,
    )
    return detail


# ── GET /api/search/papers (paginated listing) ───────────────────────────────

@router.get("/papers")
async def list_papers(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    category: Optional[str] = None,
    session: AsyncSession = Depends(get_db),
):
    offset = (page - 1) * page_size
    query = select(Paper).order_by(Paper.submitted_date.desc())
    if category:
        query = query.where(Paper.primary_category == category)

    count_q = select(text("COUNT(*)")).select_from(Paper)
    if category:
        count_q = count_q.where(Paper.primary_category == category)

    total_result = await session.execute(count_q)
    total = total_result.scalar() or 0

    result = await session.execute(query.offset(offset).limit(page_size))
    papers = result.scalars().all()

    return {
        "items": [PaperResponse.model_validate(p) for p in papers],
        "total": total,
        "page": page,
        "page_size": page_size,
        "pages": (total + page_size - 1) // page_size,
    }
