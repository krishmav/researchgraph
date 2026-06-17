"""
ResearchGraph FastAPI Application
===================================
Entry point. Configures:
  - Lifespan events (load all ML models at startup)
  - CORS middleware
  - Routers
  - Health check endpoint
  - Global exception handler
"""
from __future__ import annotations

import traceback
import threading
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core.config import get_settings
from app.core.logging import logger, setup_logging
from app.ml.baselines import BaselineService
from app.ml.gap_finder import GapFinderService
from app.ml.knowledge_graph import KnowledgeGraphService
from app.ml.retrieval import RetrievalService
from app.ml.topic_model import TopicModelService
from app.models.schemas import HealthResponse

settings = get_settings()


# ── Background Model Loader ───────────────────────────────────────────────────

def load_heavy_models_in_background():
    """Loads ML pipelines on a separate thread to prevent 502 timeouts.
       Heavy auxiliary models are disabled to survive the 512MB RAM limit."""
    try:
        logger.info("Background thread: Loading retrieval engines…")
        RetrievalService.get_instance().load_all()

        logger.info("Background thread: Loading TF-IDF baseline…")
        BaselineService.get_instance().load_all()

        logger.info("Background thread: Loading knowledge graph…")
        KnowledgeGraphService.get_instance().load()

        # 🚨 TEMPORARILY DISABLED TO PREVENT RAM CRASHES ON FREE TIER 🚨
        # logger.info("Background thread: Loading topic model…")
        # TopicModelService.get_instance().load()

        # logger.info("Background thread: Loading research gaps…")
        # GapFinderService.get_instance().load()
        
        logger.info("🎉 Core ML artifacts successfully loaded in background!")
    except Exception as e:
        logger.error(f"🚨 Background loading failed: {e}")


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Start background loader on startup; clean up on shutdown."""
    setup_logging()
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")

    # Offload the heavy lifting to a background thread so the port opens instantly
    thread = threading.Thread(target=load_heavy_models_in_background)
    thread.start()

    yield

    # Shutdown
    from app.core.database import close_db
    await close_db()
    logger.info("Database connections closed. Shutdown complete.")


# ── App factory ───────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "Semantic Research Discovery & Knowledge Intelligence Platform. "
            "Transformer-based retrieval over 25k–50k arXiv papers with "
            "knowledge graph augmentation, topic modeling, and rigorous "
            "IR evaluation."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # ── CORS ──────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Global exception handler ──────────────────────────────
    @app.exception_handler(Exception)
    async def unhandled_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        logger.error(
            f"Unhandled exception on {request.method} {request.url}: "
            f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
        )
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error. Check server logs."},
        )

    # ── Routers ───────────────────────────────────────────────
    from app.api.search import router as search_router
    from app.api.routes import (
        recommend_router,
        topics_router,
        graph_router,
        gaps_router,
        evaluate_router,
    )

    app.include_router(search_router)
    app.include_router(recommend_router)
    app.include_router(topics_router)
    app.include_router(graph_router)
    app.include_router(gaps_router)
    app.include_router(evaluate_router)

    # ── Health check ──────────────────────────────────────────
    @app.get("/health", response_model=HealthResponse, tags=["system"])
    async def health_check() -> HealthResponse:
        from sqlalchemy import text
        from app.core.database import AsyncSessionLocal

        db_ok = False
        try:
            async with AsyncSessionLocal() as session:
                await session.execute(text("SELECT 1"))
                db_ok = True
        except Exception:
            pass

        retrieval = RetrievalService.get_instance()

        # Get corpus size
        corpus_size = 0
        try:
            async with AsyncSessionLocal() as session:
                result = await session.execute(text("SELECT COUNT(*) FROM papers"))
                corpus_size = result.scalar() or 0
        except Exception:
            pass

        return HealthResponse(
            status="ok" if db_ok else "degraded",
            version=settings.app_version,
            db_connected=db_ok,
            faiss_loaded=retrieval.any_loaded,
            models_loaded=retrieval.loaded_models,
            corpus_size=corpus_size,
        )

    @app.get("/", tags=["system"])
    async def root():
        return {
            "name": settings.app_name,
            "version": settings.app_version,
            "docs": "/docs",
            "health": "/health",
        }

    return app


app = create_app()