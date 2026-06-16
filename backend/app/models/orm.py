"""
SQLAlchemy ORM models.
Mirrors exactly the schema in alembic/versions/001_initial.py.
"""
import uuid
from datetime import date, datetime
from typing import List, Optional

from sqlalchemy import (
    ARRAY,
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


def _uuid() -> uuid.UUID:
    return uuid.uuid4()


# ── Papers ────────────────────────────────────────────────────────────────────

class Paper(Base):
    __tablename__ = "papers"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=_uuid
    )
    arxiv_id: Mapped[str] = mapped_column(String(30), unique=True, nullable=False, index=True)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    abstract: Mapped[str] = mapped_column(Text, nullable=False)
    authors: Mapped[List[str]] = mapped_column(ARRAY(Text), nullable=False)
    categories: Mapped[List[str]] = mapped_column(ARRAY(Text), nullable=False)
    primary_category: Mapped[Optional[str]] = mapped_column(String(20))
    submitted_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    doi: Mapped[Optional[str]] = mapped_column(String(150))
    pdf_url: Mapped[Optional[str]] = mapped_column(Text)
    embedding_id: Mapped[Optional[int]] = mapped_column(Integer)      # row idx in FAISS
    topic_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("topics.id", ondelete="SET NULL"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # Relationships
    topic: Mapped[Optional["Topic"]] = relationship(back_populates="papers")
    paper_authors: Mapped[List["PaperAuthor"]] = relationship(
        back_populates="paper", cascade="all, delete-orphan"
    )
    similarity_edges_a: Mapped[List["SimilarityEdge"]] = relationship(
        "SimilarityEdge",
        foreign_keys="[SimilarityEdge.paper_id_a]",
        back_populates="paper_a",
        cascade="all, delete-orphan",
    )
    similarity_edges_b: Mapped[List["SimilarityEdge"]] = relationship(
        "SimilarityEdge",
        foreign_keys="[SimilarityEdge.paper_id_b]",
        back_populates="paper_b",
        cascade="all, delete-orphan",
    )

    __table_args__ = (
        Index(
            "papers_fts_idx",
            text("to_tsvector('english', title || ' ' || abstract)"),
            postgresql_using="gin",
        ),
        Index("papers_categories_idx", "categories", postgresql_using="gin"),
    )

    def __repr__(self) -> str:
        return f"<Paper {self.arxiv_id}: {self.title[:50]}>"


# ── Authors ───────────────────────────────────────────────────────────────────

class Author(Base):
    __tablename__ = "authors"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    name_lower: Mapped[str] = mapped_column(Text, nullable=False)
    paper_count: Mapped[int] = mapped_column(Integer, default=0)

    paper_authors: Mapped[List["PaperAuthor"]] = relationship(back_populates="author")

    __table_args__ = (
        UniqueConstraint("name_lower", name="uq_authors_name_lower"),
        Index("authors_name_lower_idx", "name_lower"),
    )


class PaperAuthor(Base):
    __tablename__ = "paper_authors"

    paper_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("papers.id", ondelete="CASCADE"), primary_key=True
    )
    author_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("authors.id", ondelete="CASCADE"), primary_key=True
    )
    author_order: Mapped[int] = mapped_column(Integer, nullable=False)

    paper: Mapped["Paper"] = relationship(back_populates="paper_authors")
    author: Mapped["Author"] = relationship(back_populates="paper_authors")


# ── Topics ────────────────────────────────────────────────────────────────────

class Topic(Base):
    __tablename__ = "topics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    label: Mapped[str] = mapped_column(Text, nullable=False)
    top_words: Mapped[List[str]] = mapped_column(ARRAY(Text), nullable=False)
    paper_count: Mapped[int] = mapped_column(Integer, default=0)
    is_outlier: Mapped[bool] = mapped_column(Boolean, default=False)
    coherence_score: Mapped[Optional[float]] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    papers: Mapped[List["Paper"]] = relationship(back_populates="topic")
    trends: Mapped[List["TopicTrend"]] = relationship(
        back_populates="topic", cascade="all, delete-orphan"
    )


class TopicTrend(Base):
    __tablename__ = "topic_trends"

    topic_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("topics.id", ondelete="CASCADE"), primary_key=True
    )
    year_month: Mapped[date] = mapped_column(Date, primary_key=True)
    paper_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    topic: Mapped["Topic"] = relationship(back_populates="trends")


# ── Graph edges ───────────────────────────────────────────────────────────────

class SimilarityEdge(Base):
    """Pre-computed top-20 semantic neighbors per paper."""
    __tablename__ = "similarity_edges"

    paper_id_a: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("papers.id", ondelete="CASCADE"), primary_key=True
    )
    paper_id_b: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("papers.id", ondelete="CASCADE"), primary_key=True
    )
    similarity: Mapped[float] = mapped_column(Float, nullable=False)

    paper_a: Mapped["Paper"] = relationship(
        "Paper", foreign_keys=[paper_id_a], back_populates="similarity_edges_a"
    )
    paper_b: Mapped["Paper"] = relationship(
        "Paper", foreign_keys=[paper_id_b], back_populates="similarity_edges_b"
    )

    __table_args__ = (
        Index("sim_edges_a_idx", "paper_id_a"),
        Index("sim_edges_b_idx", "paper_id_b"),
    )


class CitationEdge(Base):
    __tablename__ = "citation_edges"

    citing_paper_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("papers.id", ondelete="CASCADE"), primary_key=True
    )
    cited_paper_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("papers.id", ondelete="CASCADE"), primary_key=True
    )


class SearchLog(Base):
    __tablename__ = "search_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    query: Mapped[str] = mapped_column(Text, nullable=False)
    method: Mapped[str] = mapped_column(String(20), nullable=False)
    result_ids: Mapped[Optional[List[str]]] = mapped_column(ARRAY(Text))
    latency_ms: Mapped[Optional[int]] = mapped_column(Integer)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )
