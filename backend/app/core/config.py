"""
Application configuration using pydantic-settings.
All values are read from environment variables or .env file.
"""
from functools import lru_cache
from pathlib import Path
from typing import List

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Application ──────────────────────────────────────────
    app_name: str = "ResearchGraph API"
    app_version: str = "1.0.0"
    debug: bool = False
    log_level: str = "INFO"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    secret_key: str = "change-this-in-production"

    # ── CORS ─────────────────────────────────────────────────
    allowed_origins: str = "http://localhost:3000"

    @property
    def cors_origins(self) -> List[str]:
        return [o.strip() for o in self.allowed_origins.split(",")]

    # ── Database ─────────────────────────────────────────────
    database_url: str = "postgresql://postgres:postgres@localhost:5432/researchgraph"

    # ── ML ───────────────────────────────────────────────────
    default_embedding_model: str = "miniml"
    embedding_batch_size: int = 64
    torch_device: str = "cpu"

    # ── Paths ─────────────────────────────────────────────────
    faiss_index_path: str = "../data/faiss"
    graph_path: str = "../data/graphs"
    tfidf_path: str = "../data/tfidf"
    embeddings_path: str = "../data/processed/embeddings"

    @property
    def faiss_dir(self) -> Path:
        return Path(self.faiss_index_path)

    @property
    def graph_dir(self) -> Path:
        return Path(self.graph_path)

    @property
    def tfidf_dir(self) -> Path:
        return Path(self.tfidf_path)

    @property
    def embeddings_dir(self) -> Path:
        return Path(self.embeddings_path)

    # ── Model registry ───────────────────────────────────────
    EMBEDDING_MODELS: dict = {
        "miniml": "sentence-transformers/all-MiniLM-L6-v2",
        "mpnet": "sentence-transformers/all-mpnet-base-v2",
        "bge": "BAAI/bge-large-en-v1.5",
    }

    @property
    def embedding_model_name(self) -> str:
        return self.EMBEDDING_MODELS.get(
            self.default_embedding_model,
            self.EMBEDDING_MODELS["miniml"],
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
