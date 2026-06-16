"""Initial schema

Revision ID: 001
Revises:
Create Date: 2026-01-01
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "topics",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("label", sa.Text(), nullable=False),
        sa.Column("top_words", postgresql.ARRAY(sa.Text()), nullable=False),
        sa.Column("paper_count", sa.Integer(), nullable=True, default=0),
        sa.Column("is_outlier", sa.Boolean(), nullable=True, default=False),
        sa.Column("coherence_score", sa.Float(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_table(
        "papers",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("arxiv_id", sa.String(30), nullable=False),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("abstract", sa.Text(), nullable=False),
        sa.Column("authors", postgresql.ARRAY(sa.Text()), nullable=False),
        sa.Column("categories", postgresql.ARRAY(sa.Text()), nullable=False),
        sa.Column("primary_category", sa.String(20), nullable=True),
        sa.Column("submitted_date", sa.Date(), nullable=False),
        sa.Column("doi", sa.String(150), nullable=True),
        sa.Column("pdf_url", sa.Text(), nullable=True),
        sa.Column("embedding_id", sa.Integer(), nullable=True),
        sa.Column("topic_id", sa.Integer(), sa.ForeignKey("topics.id", ondelete="SET NULL"), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("arxiv_id", name="uq_papers_arxiv_id"),
    )
    op.create_index("papers_arxiv_id_idx", "papers", ["arxiv_id"])
    op.create_index("papers_date_idx", "papers", [sa.text("submitted_date DESC")])
    op.create_index(
        "papers_fts_idx", "papers",
        [sa.text("to_tsvector('english', title || ' ' || abstract)")],
        postgresql_using="gin",
    )
    op.create_index("papers_categories_idx", "papers", ["categories"], postgresql_using="gin")

    op.create_table(
        "authors",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("name_lower", sa.Text(), nullable=False),
        sa.Column("paper_count", sa.Integer(), nullable=True, default=0),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("name_lower", name="uq_authors_name_lower"),
    )
    op.create_index("authors_name_lower_idx", "authors", ["name_lower"])

    op.create_table(
        "paper_authors",
        sa.Column("paper_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("papers.id", ondelete="CASCADE"), nullable=False),
        sa.Column("author_id", sa.Integer(), sa.ForeignKey("authors.id", ondelete="CASCADE"), nullable=False),
        sa.Column("author_order", sa.Integer(), nullable=False),
        sa.PrimaryKeyConstraint("paper_id", "author_id"),
    )

    op.create_table(
        "topic_trends",
        sa.Column("topic_id", sa.Integer(), sa.ForeignKey("topics.id", ondelete="CASCADE"), nullable=False),
        sa.Column("year_month", sa.Date(), nullable=False),
        sa.Column("paper_count", sa.Integer(), nullable=False, default=0),
        sa.PrimaryKeyConstraint("topic_id", "year_month"),
    )

    op.create_table(
        "similarity_edges",
        sa.Column("paper_id_a", postgresql.UUID(as_uuid=True), sa.ForeignKey("papers.id", ondelete="CASCADE"), nullable=False),
        sa.Column("paper_id_b", postgresql.UUID(as_uuid=True), sa.ForeignKey("papers.id", ondelete="CASCADE"), nullable=False),
        sa.Column("similarity", sa.Float(), nullable=False),
        sa.PrimaryKeyConstraint("paper_id_a", "paper_id_b"),
    )
    op.create_index("sim_edges_a_idx", "similarity_edges", ["paper_id_a"])
    op.create_index("sim_edges_b_idx", "similarity_edges", ["paper_id_b"])

    op.create_table(
        "citation_edges",
        sa.Column("citing_paper_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("papers.id", ondelete="CASCADE"), nullable=False),
        sa.Column("cited_paper_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("papers.id", ondelete="CASCADE"), nullable=False),
        sa.PrimaryKeyConstraint("citing_paper_id", "cited_paper_id"),
    )

    op.create_table(
        "search_logs",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("query", sa.Text(), nullable=False),
        sa.Column("method", sa.String(20), nullable=False),
        sa.Column("result_ids", postgresql.ARRAY(sa.Text()), nullable=True),
        sa.Column("latency_ms", sa.Integer(), nullable=True),
        sa.Column("timestamp", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("search_logs_timestamp_idx", "search_logs", ["timestamp"])


def downgrade() -> None:
    op.drop_table("search_logs")
    op.drop_table("citation_edges")
    op.drop_table("similarity_edges")
    op.drop_table("topic_trends")
    op.drop_table("paper_authors")
    op.drop_table("authors")
    op.drop_table("papers")
    op.drop_table("topics")
