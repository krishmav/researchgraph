# ResearchGraph: Semantic Research Discovery via Transformer Embeddings and Knowledge Graphs

**Draft — Research Methodology**

---

## Abstract

We present ResearchGraph, a semantic research discovery platform that combines transformer-based dense retrieval with graph-based knowledge representation over a corpus of 25,000–50,000 arXiv papers. We empirically evaluate six retrieval methods — keyword search, TF-IDF, and three transformer embedding models (MiniLM, MPNet, BGE-large) with and without knowledge graph augmentation — using category-based pseudo-relevance labels as ground truth. Our primary research question asks whether semantic retrieval with graph augmentation significantly outperforms keyword and TF-IDF baselines on standard IR metrics (Precision@K, MRR, NDCG@10). We further investigate automated topic discovery via BERTopic+UMAP+HDBSCAN and a novel research gap identification method based on kernel density estimation over 2-D UMAP projections.

---

## 1. Introduction

Academic search systems remain predominantly keyword-driven, relying on BM25-style term matching (Robertson & Zaragoza, 2009) and citation count signals. This creates systematic blind spots: (1) vocabulary mismatch between query and document terms; (2) poor recall for emerging topics lacking established keyword density; (3) invisible interdisciplinary connections where identical methods appear under different nomenclature across fields.

Dense retrieval with pre-trained language models (Karpukhin et al., 2020; Ni et al., 2021) has demonstrated strong performance on open-domain question answering and information retrieval benchmarks (Thakur et al., 2021). However, their application to systematic academic paper discovery—with rigorous comparison against classical baselines and augmentation with knowledge graph signals—remains understudied in the context of multi-domain arXiv corpora.

This work makes the following contributions:

1. A reproducible evaluation framework comparing 6 retrieval methods over 25k–50k arXiv papers, using category overlap as pseudo-relevance labels.
2. A graph-augmented re-ranking method combining cosine similarity, KG neighbourhood overlap, and PageRank.
3. An unsupervised topic discovery pipeline (BERTopic+UMAP+HDBSCAN) with emerging topic detection via linear regression on monthly paper counts.
4. A novel research gap identification method using KDE on 2-D UMAP projections to locate under-explored areas.
5. A fully open-source, deployable platform exposing all components via a REST API and interactive web interface.

---

## 2. Related Work

### 2.1 Dense Retrieval
Bi-encoder models (Karpukhin et al., 2020) encode queries and passages independently into a shared embedding space, enabling efficient approximate nearest-neighbour search. Sentence-BERT (Reimers & Gurevych, 2019) extended this to general sentence similarity, providing strong baselines for semantic search. BGE (Chen et al., 2024) and E5 (Wang et al., 2022) represent the current state of the art on the MTEB and BEIR benchmarks.

### 2.2 Academic Search
Semantic Scholar (Lo et al., 2020) uses citation-augmented representations. SPECTER (Cohan et al., 2020) fine-tunes SciBERT on citation graphs. Our work differs in scope: we focus on evaluating off-the-shelf retrieval models without fine-tuning on citation data, and augment with a locally-constructed knowledge graph.

### 2.3 Topic Modelling
LDA (Blei et al., 2003) requires a bag-of-words assumption and is sensitive to vocabulary. BERTopic (Grootendorst, 2022) uses pre-trained embeddings + UMAP dimensionality reduction + HDBSCAN clustering + c-TF-IDF labelling, producing more coherent and interpretable topics without requiring a predetermined topic count.

### 2.4 Knowledge Graphs for IR
Graph-augmented retrieval has been studied in the context of entity retrieval (Xiong et al., 2021) and knowledge-intensive NLP (Lewis et al., 2020). Our approach applies graph structural signals (PageRank, neighbourhood overlap) as a retrieval re-ranking signal, rather than as a generation condition.

---

## 3. Dataset

**Source**: arXiv API (export.arxiv.org/api/query).

**Coverage**: Papers from January 2019 to December 2024 in six domains:
- cs.LG (Machine Learning): ~15,000 papers
- cs.CV (Computer Vision): ~12,000 papers
- cs.CL (Computation and Language): ~10,000 papers
- cs.AI (Artificial Intelligence): ~5,000 papers
- cs.CR (Cryptography and Security): ~5,000 papers
- cs.SE (Software Engineering): ~3,000 papers

**Collection**: Rate-limited (3 req/s) using 6-month sliding windows per category. Deduplicated by arXiv ID. Validated for non-empty title and abstract.

**Storage**: PostgreSQL 16 with GIN full-text index on title+abstract concatenation.

---

## 4. Retrieval Methods

### 4.1 Keyword Search (Baseline A)
PostgreSQL ts_rank_cd over GIN-indexed `to_tsvector('english', title || ' ' || abstract)` using `plainto_tsquery`. Equivalent to BM25-style retrieval.

### 4.2 TF-IDF (Baseline B)
Scikit-learn TfidfVectorizer with max_features=75,000, min_df=2, max_df=0.85, sublinear TF, unigram+bigram tokens. Cosine similarity between query and stored sparse matrix.

### 4.3 MiniLM-L6 (Model C)
Sentence-Transformers `all-MiniLM-L6-v2`. 384-dimensional embeddings, L2-normalised. FAISS IndexFlatIP (exact cosine search). Encodes title+abstract concatenation.

### 4.4 MPNet-base (Model D)
`all-mpnet-base-v2`. 768-dimensional embeddings. Same indexing procedure.

### 4.5 BGE-large (Model E)
`BAAI/bge-large-en-v1.5`. 1024-dimensional embeddings. Instruction prefix added for retrieval queries: *"Represent this sentence for searching relevant passages: "*. State-of-the-art on MTEB retrieval as of 2024.

### 4.6 Graph-Augmented (Model F)
BGE embeddings re-ranked with:

```
score(p) = α · cos(q, p) + β · KG_signal(p) + γ · PageRank(p)
```

Where:
- `cos(q, p)` = cosine similarity between query and paper embedding
- `KG_signal(p)` = 1 if p is a 1-hop neighbour of any top-10 BGE result, else 0
- `PageRank(p)` = damping=0.85 PageRank score on paper similarity graph
- Default: α=0.7, β=0.2, γ=0.1

---

## 5. Evaluation Methodology

### 5.1 Pseudo-Relevance Labels
**Strategy**: A paper is relevant to a query paper if it shares ≥1 arXiv primary or secondary category. This is standard practice in academic IR evaluation when gold relevance labels are unavailable (Thakur et al., 2021; Cohan et al., 2020).

**Limitation**: Category-level relevance is coarse. Two papers may share `cs.LG` without being topically related. This biases all methods toward recall over precision. We mitigate this by also reporting MRR (which rewards highly-ranked relevant results).

### 5.2 Test Set Construction
1. Sample N=500 papers uniformly from the corpus
2. Construct query text as paper title
3. Compute relevant set from category overlap (excluding query paper)
4. Remove queries with zero relevant papers

### 5.3 Metrics
- **Precision@5, @10**: |relevant ∩ top-K| / K
- **Recall@10**: |relevant ∩ top-10| / |relevant|
- **MRR**: 1 / rank of first relevant result
- **NDCG@10**: Normalised Discounted Cumulative Gain with binary relevance

### 5.4 Statistical Testing
Paired t-test (per-query NDCG@10) comparing each method against Keyword baseline. Threshold: p < 0.05.

### 5.5 Latency
Median (p50) and 95th percentile (p95) query latency over 100 repeated queries on a CPU-only server.

---

## 6. Topic Modelling Pipeline

### 6.1 Embedding Reuse
Topic modelling reuses the embeddings generated for retrieval (no additional model loading).

### 6.2 Dimensionality Reduction
UMAP (McInnes et al., 2018): `n_components=5`, `n_neighbors=15`, `min_dist=0.0`, `metric=cosine`, `random_state=42`. Low-memory mode for CPU compatibility.

### 6.3 Clustering
HDBSCAN: `min_cluster_size=15`, `min_samples=10`. Papers assigned to topic -1 are flagged as outliers and excluded from trend analysis.

### 6.4 Topic Representation
BERTopic c-TF-IDF: class-level TF-IDF that upweights terms appearing frequently within a topic relative to the full corpus. Top-10 words per topic form the topic representation.

### 6.5 Emerging Topic Detection
For each topic with ≥6 months of data:
- Aggregate paper count by calendar month
- Fit OLS linear regression: `count ~ month_index`
- Topics with slope > 2.0 papers/month are flagged as "emerging"

---

## 7. Research Gap Discovery

### 7.1 Sparse Region Analysis (Strategy A)
1. Project all embeddings to 2-D with UMAP (`n_components=2`, `min_dist=0.1`)
2. Fit Gaussian KDE on 2-D coordinates (Scott's rule bandwidth)
3. Evaluate density on 50×50 grid
4. Find grid cells with: density < 5th percentile AND distance from nearest data point > 0.5 units
5. For each sparse cell: find the 3 nearest topic centroids by Euclidean distance in 2-D space
6. Report as gap: flanking topics, sparsity score (normalised), semantic distance between flanking topic centroids in full embedding space

### 7.2 Structural Graph Gaps (Strategy B)
1. Build topic co-occurrence count: for each paper, increment count for all pairs of topics it belongs to (e.g., a paper tagged cs.LG and cs.CV contributes to the LG-CV pair)
2. Compute cosine similarity between full-dimensional topic centroids
3. Flag topic pairs where: cosine_sim > 0.4 AND co_occurrence ≤ 5
4. Score = cosine_sim / (co_occurrence + 1) — higher = stronger gap

---

## 8. Expected Results

(To be filled in with actual experimental results)

| Method      | P@5  | P@10 | R@10 | MRR  | NDCG@10 | p95 (ms) |
|-------------|------|------|------|------|---------|---------|
| Keyword     | —    | —    | —    | —    | —       | —       |
| TF-IDF      | —    | —    | —    | —    | —       | —       |
| MiniLM      | —    | —    | —    | —    | —       | —       |
| MPNet       | —    | —    | —    | —    | —       | —       |
| BGE         | —    | —    | —    | —    | —       | —       |
| Graph+      | —    | —    | —    | —    | —       | —       |

---

## 9. References

- Blei, D. M., Ng, A. Y., & Jordan, M. I. (2003). Latent Dirichlet Allocation. *JMLR*.
- Chen, J., et al. (2024). BGE M3-Embedding. *arXiv:2402.03216*.
- Cohan, A., et al. (2020). SPECTER. *ACL 2020*.
- Grootendorst, M. (2022). BERTopic. *arXiv:2203.05794*.
- Karpukhin, V., et al. (2020). Dense Passage Retrieval. *EMNLP 2020*.
- Lewis, P., et al. (2020). Retrieval-Augmented Generation. *NeurIPS 2020*.
- Lo, K., et al. (2020). S2ORC. *ACL 2020*.
- McInnes, L., Healy, J., & Melville, J. (2018). UMAP. *arXiv:1802.03426*.
- Reimers, N., & Gurevych, I. (2019). Sentence-BERT. *EMNLP 2019*.
- Robertson, S., & Zaragoza, H. (2009). The Probabilistic Relevance Framework: BM25. *FnTIR*.
- Thakur, N., et al. (2021). BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation. *NeurIPS 2021*.
- Wang, L., et al. (2022). Text Embeddings by Weakly-Supervised Contrastive Pre-training. *arXiv:2212.03533*.
- Xiong, W., et al. (2021). Answering Complex Open-Domain Questions with Multi-Hop Dense Retrieval. *ICLR 2021*.
