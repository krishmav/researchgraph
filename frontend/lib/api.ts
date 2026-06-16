// lib/api.ts
import type {
  SearchResponse,
  CompareSearchResponse,
  RecommendResponse,
  TopicWithTrend,
  TopicMapResponse,
  GraphResponse,
  GapResponse,
  EvaluationResponse,
  HealthResponse,
  PaperDetail,
  RetrievalMethod,
} from "./types";

const BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

async function request<T>(
  path: string,
  options: RequestInit = {}
): Promise<T> {
  const url = `${BASE}${path}`;
  const res = await fetch(url, {
    headers: { "Content-Type": "application/json", ...options.headers },
    ...options,
  });
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`API error ${res.status}: ${body}`);
  }
  return res.json();
}

// ── Search ────────────────────────────────────────────────────────────────────

export async function search(
  query: string,
  method: RetrievalMethod = "bge",
  topK: number = 10,
  categoryFilter?: string
): Promise<SearchResponse> {
  return request("/api/search", {
    method: "POST",
    body: JSON.stringify({
      query,
      method,
      top_k: topK,
      category_filter: categoryFilter,
    }),
  });
}

export async function compareSearch(
  query: string,
  topK: number = 10
): Promise<CompareSearchResponse> {
  return request("/api/search/compare", {
    method: "POST",
    body: JSON.stringify({ query, method: "bge", top_k: topK }),
  });
}

export async function getPaper(arxivId: string): Promise<PaperDetail> {
  return request(`/api/search/paper/${arxivId}`);
}

// ── Recommendations ───────────────────────────────────────────────────────────

export async function getRecommendations(
  paperId: string,
  method: "content" | "graph" = "graph",
  topK: number = 10
): Promise<RecommendResponse> {
  return request("/api/recommend", {
    method: "POST",
    body: JSON.stringify({ paper_id: paperId, method, top_k: topK }),
  });
}

// ── Topics ────────────────────────────────────────────────────────────────────

export async function getTopics(): Promise<TopicWithTrend[]> {
  return request("/api/topics");
}

export async function getTrendingTopics(): Promise<TopicWithTrend[]> {
  return request("/api/topics/trending");
}

export async function getTopicMap(maxPoints: number = 3000): Promise<TopicMapResponse> {
  return request(`/api/topics/map?max_points=${maxPoints}`);
}

export async function getTopic(topicId: number): Promise<TopicWithTrend> {
  return request(`/api/topics/${topicId}`);
}

// ── Knowledge Graph ───────────────────────────────────────────────────────────

export async function getPaperGraph(
  arxivId: string,
  radius: number = 2
): Promise<GraphResponse> {
  return request(`/api/graph/paper/${arxivId}?radius=${radius}`);
}

export async function getGraphStats(): Promise<Record<string, unknown>> {
  return request("/api/graph/stats");
}

export async function getTopPapersByPageRank(n: number = 20) {
  return request(`/api/graph/top-papers?n=${n}`);
}

// ── Research Gaps ─────────────────────────────────────────────────────────────

export async function getResearchGaps(
  strategy: "both" | "sparse" | "structural" = "both",
  limit: number = 10
): Promise<GapResponse> {
  return request(`/api/gaps?strategy=${strategy}&limit=${limit}`);
}

// ── Evaluation ────────────────────────────────────────────────────────────────

export async function getCachedEvaluation(): Promise<EvaluationResponse> {
  return request("/api/evaluate/cached");
}

export async function runEvaluation(numQueries: number = 100): Promise<EvaluationResponse> {
  return request(`/api/evaluate?num_queries=${numQueries}`, { method: "POST" });
}

// ── Health ────────────────────────────────────────────────────────────────────

export async function getHealth(): Promise<HealthResponse> {
  return request("/health");
}
