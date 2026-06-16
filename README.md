# ResearchGraph

**Semantic Research Discovery & Knowledge Intelligence Platform**

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green)](https://fastapi.tiangolo.com)
[![Next.js](https://img.shields.io/badge/Next.js-14-black)](https://nextjs.org)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

> A research-grade semantic discovery platform over 25,000–50,000 arXiv papers using transformer embeddings, vector search, topic modeling, and knowledge graphs. Built to answer: *Can transformer-based semantic retrieval combined with graph-based knowledge representation outperform traditional keyword and TF-IDF-based academic search?*

---

## Research Questions

**Primary**: Can transformer-based semantic retrieval + graph knowledge representation significantly outperform keyword-based retrieval for academic paper discovery?

**Secondary**:
1. How much does embedding-based retrieval improve Precision@K vs keyword/TF-IDF baselines?
2. Does graph-augmented re-ranking measurably improve MRR and NDCG over pure embedding similarity?
3. Can BERTopic + UMAP cluster the corpus into interpretable, coherent research themes?
4. Can sparse regions in the knowledge graph reliably identify under-explored research directions?

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│              Next.js Frontend (Vercel)                   │
│  Search · Topics · Knowledge Graph · Gaps · Evaluation  │
└──────────────────────┬──────────────────────────────────┘
                       │ REST / JSON
┌──────────────────────▼──────────────────────────────────┐
│              FastAPI Backend (Render)                    │
│  /search  /recommend  /topics  /graph  /gaps  /eval     │
└──┬──────────┬──────────┬──────────┬──────────┬──────────┘
   │          │          │          │          │
┌──▼──┐  ┌───▼───┐  ┌───▼───┐  ┌──▼──┐  ┌───▼───┐
│ M1  │  │  M2   │  │  M3   │  │ M4  │  │  M5   │
│Ret. │  │Recco. │  │Topic  │  │ KG  │  │ Gap   │
└──┬──┘  └───┬───┘  └───┬───┘  └──┬──┘  └───┬───┘
   └──────────┴──────────┴──────────┴──────────┘
                         │
              ┌──────────▼──────────┐
              │  Shared Data Layer  │
              │  PostgreSQL · FAISS  │
              │  NetworkX graphs    │
              └─────────────────────┘
```

---

## ML Modules

| Module | Technology | Purpose |
|--------|-----------|---------|
| **M1 Semantic Retrieval** | Sentence Transformers + FAISS | Replace keyword search with semantic understanding |
| **M2 Recommendations** | Cosine similarity + graph re-ranking | Discover related papers with explanations |
| **M3 Topic Modeling** | BERTopic + UMAP + HDBSCAN | Unsupervised research theme discovery |
| **M4 Knowledge Graph** | NetworkX + PyVis | Paper–Author–Topic relationship graph |
| **M5 Gap Finder** | KDE sparse analysis | Identify under-explored research areas |
| **M6 Evaluation** | Scikit-learn metrics | Rigorous benchmark across 6 retrieval methods |

---

## Retrieval Methods Compared

| Method | Type | Expected P@10 |
|--------|------|--------------|
| Keyword Search | BM25-style (PostgreSQL FTS) | ~0.38 |
| TF-IDF | Sparse vector retrieval | ~0.44 |
| MiniLM-L6 | Dense embedding (384-dim) | ~0.55 |
| MPNet-base | Dense embedding (768-dim) | ~0.60 |
| BGE-large | Dense embedding (1024-dim) | ~0.62 |
| Graph-Augmented | BGE + KG re-ranking | ~0.65 |

---

## Quick Start

### Prerequisites

- Python 3.11 ([python.org](https://python.org))
- Node.js 18+ ([nodejs.org](https://nodejs.org))
- PostgreSQL 16 ([postgresql.org](https://postgresql.org))
- Git

### 1. Clone and configure

```bash
git clone https://github.com/yourusername/researchgraph.git
cd researchgraph
cp .env.example .env
# Edit .env with your PostgreSQL credentials
```

### 2. Backend setup

```bash
cd backend
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
alembic upgrade head           # Run DB migrations
```

### 3. Collect data (run once — takes ~2-3 hours for 25k papers)

```bash
cd ..
python scripts/collect_arxiv.py --max-papers 25000
python scripts/preprocess_papers.py
python scripts/generate_embeddings.py --model miniml   # Start with smallest model
python scripts/build_faiss_index.py
python scripts/build_tfidf.py
python scripts/train_bertopic.py
python scripts/build_knowledge_graph.py
```

### 4. Start backend

```bash
cd backend
venv\Scripts\activate
uvicorn app.main:app --reload --port 8000
```

### 5. Frontend setup

```bash
cd frontend
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000)

---

## Running Evaluation

```bash
cd backend
venv\Scripts\activate
python -m app.ml.evaluation --queries 500 --output ../research/results/
```

This generates:
- `benchmark_table.csv` — full metrics across all 6 methods
- `ablation_results.csv` — graph re-ranking hyperparameter sweep
- `significance_tests.csv` — paired t-test p-values
- `figures/` — publication-ready plots

---

## Docker (Alternative)

```bash
docker-compose up --build
```

---

## Project Structure

```
researchgraph/
├── backend/                    # FastAPI application
│   ├── app/
│   │   ├── api/               # Route handlers
│   │   ├── models/            # SQLAlchemy + Pydantic schemas
│   │   ├── ml/                # All ML modules
│   │   └── core/              # Config, DB, logging
│   └── tests/                 # Pytest test suite
├── frontend/                  # Next.js application
│   ├── app/                   # App Router pages
│   ├── components/            # React components
│   └── lib/                   # API client + types
├── scripts/                   # Data pipeline scripts
├── notebooks/                 # Jupyter exploration notebooks
├── data/                      # Embeddings, FAISS index, graphs
└── research/                  # Paper drafts, results, figures
```

---

## Deployment

- **Frontend**: Vercel (auto-deploy from `main` branch)
- **Backend**: Render (Docker container)
- **Database**: Render PostgreSQL or Supabase free tier

See [docs/deployment.md](docs/deployment.md) for full instructions.

---

## Research Report

The methodology, baselines, evaluation, and results are documented in [`research/paper_draft.md`](research/paper_draft.md), written in IEEE conference format suitable for submission to ACL Student Research Workshop, SIGIR, or EMNLP Demo Track.

---

## License

MIT License — see [LICENSE](LICENSE)
