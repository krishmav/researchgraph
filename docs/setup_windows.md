# Developer Setup Guide (Windows Native)

Complete step-by-step guide to run ResearchGraph locally on Windows without Docker.

---

## Step 1 — Install Prerequisites

### Python 3.11
1. Download from https://www.python.org/downloads/release/python-3119/
2. Run installer — **check "Add Python to PATH"**
3. Verify: open Command Prompt → `python --version`

### Node.js 20 LTS
1. Download from https://nodejs.org/en/download
2. Run installer (accept defaults)
3. Verify: `node --version` and `npm --version`

### PostgreSQL 16
1. Download from https://www.postgresql.org/download/windows/
2. Run installer:
   - Port: **5432** (default)
   - Password for postgres user: **remember this** — you'll put it in `.env`
   - Locale: default
3. Verify: open pgAdmin or run `psql -U postgres -c "SELECT version();"` in CMD

### Git
1. Download from https://git-scm.com/download/win
2. Install with defaults

---

## Step 2 — Clone and Configure

```cmd
git clone https://github.com/yourusername/researchgraph.git
cd researchgraph
copy .env.example .env
```

Open `.env` in Notepad and update:
```
DATABASE_URL=postgresql://postgres:YOURPASSWORD@localhost:5432/researchgraph
POSTGRES_PASSWORD=YOURPASSWORD
```

---

## Step 3 — Create the Database

```cmd
psql -U postgres -c "CREATE DATABASE researchgraph;"
```

---

## Step 4 — Backend Setup

```cmd
cd backend
python -m venv venv
venv\Scripts\activate
pip install torch==2.3.0 --extra-index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

Run migrations:
```cmd
alembic upgrade head
```

---

## Step 5 — Collect Data (Run Once — ~2-3 hours for 25k papers)

Open a new CMD window in the project root:

```cmd
cd backend
venv\Scripts\activate
cd ..

REM Collect papers from arXiv
python scripts/collect_arxiv.py --max-papers 25000

REM Insert into PostgreSQL
python scripts/preprocess_papers.py

REM Generate embeddings (MiniLM is fastest on CPU — ~30 min for 25k papers)
python scripts/generate_embeddings.py --model miniml

REM Build FAISS index
REM (Already done as part of generate_embeddings.py)

REM Build TF-IDF baseline
python scripts/build_tfidf.py

REM Train BERTopic topic model (~20-40 min on CPU)
python scripts/train_bertopic.py --model miniml

REM Build knowledge graph
python scripts/build_knowledge_graph.py --model miniml
```

> **Tip**: Start with `--max-papers 5000` for a quick first run (~15 min total).
> The system works with any corpus size.

---

## Step 6 — Start the Backend

```cmd
cd backend
venv\Scripts\activate
uvicorn app.main:app --reload --port 8000
```

Visit: http://localhost:8000/docs — you should see the Swagger UI.

---

## Step 7 — Start the Frontend

Open a new CMD window:

```cmd
cd frontend
npm install
npm run dev
```

Visit: http://localhost:3000

---

## Step 8 — Run Evaluation (Optional)

```cmd
cd backend
venv\Scripts\activate
python -m pytest tests/ -v
```

To run the full benchmark:
```cmd
REM Via the API (starts automatically when backend is running)
REM POST http://localhost:8000/api/evaluate?num_queries=100
```

---

## Troubleshooting

### `psycopg2` install fails
Install the binary version already in requirements.txt (`psycopg2-binary`).
If it still fails: `pip install psycopg2-binary --no-binary psycopg2`

### `faiss-cpu` install error
```cmd
pip install faiss-cpu==1.8.0 --no-deps
pip install numpy==1.26.4
```

### Port 5432 not available
Another process is using PostgreSQL. Check with:
```cmd
netstat -ano | findstr :5432
```

### Slow embedding generation
CPU embedding is slow. For 25k papers at batch_size=32:
- MiniLM: ~30-45 min
- MPNet: ~60-90 min
- BGE: ~90-120 min

Run MiniLM first to get the system working, then add other models later.

### Memory errors during BERTopic
Reduce corpus size: `--max-papers 10000`
Or set in scripts/train_bertopic.py: reduce UMAP `n_neighbors` to 10.

---

## VS Code Setup

Recommended extensions:
- Python (ms-python.python)
- Pylance
- ESLint
- Tailwind CSS IntelliSense
- REST Client (for API testing)

Python interpreter: select `backend/venv/Scripts/python.exe`
