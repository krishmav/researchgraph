// lib/types.ts
// ─────────────────────────────────────────────────────────────────────────────

export type RetrievalMethod = "keyword" | "tfidf" | "miniml" | "mpnet" | "bge" | "graph";

export interface Paper {
  id: string;
  arxiv_id: string;
  title: string;
  abstract: string;
  authors: string[];
  categories: string[];
  primary_category: string | null;
  submitted_date: string;
  doi: string | null;
  pdf_url: string | null;
  topic_id: number | null;
  created_at: string;
}

export interface PaperDetail extends Paper {
  topic_label: string | null;
  topic_top_words: string[] | null;
}

export interface SearchResult {
  paper: Paper;
  score: number;
  explanation: string;
  rank: number;
}

export interface SearchResponse {
  query: string;
  method: RetrievalMethod;
  total_results: number;
  results: SearchResult[];
  latency_ms: number;
}

export interface CompareSearchResponse {
  query: string;
  methods: Record<RetrievalMethod, SearchResult[]>;
  latency_ms: Record<RetrievalMethod, number>;
}

export interface RecommendResult {
  paper: Paper;
  score: number;
  explanation: string;
  explanation_type: string;
}

export interface RecommendResponse {
  seed_paper: Paper;
  method: string;
  recommendations: RecommendResult[];
  latency_ms: number;
}

export interface Topic {
  id: number;
  label: string;
  top_words: string[];
  paper_count: number;
  is_outlier: boolean;
  coherence_score: number | null;
}

export interface TopicTrendPoint {
  year_month: string;
  paper_count: number;
}

export interface TopicWithTrend extends Topic {
  trend: TopicTrendPoint[];
  growth_slope: number | null;
  is_emerging: boolean;
}

export interface TopicMapPoint {
  paper_id: string;
  arxiv_id: string;
  title: string;
  x: number;
  y: number;
  topic_id: number;
  topic_label: string;
}

export interface TopicMapResponse {
  points: TopicMapPoint[];
  topics: Topic[];
  total_papers: number;
}

export interface GraphNode {
  id: string;
  label: string;
  node_type: "paper" | "author" | "topic" | "area";
  size: number;
  color: string;
  metadata: Record<string, unknown>;
}

export interface GraphEdge {
  source: string;
  target: string;
  weight: number;
  edge_type: string;
}

export interface GraphResponse {
  nodes: GraphNode[];
  edges: GraphEdge[];
  stats: Record<string, unknown>;
}

export interface ResearchGap {
  gap_id: number;
  description: string;
  flanking_topics: string[];
  evidence_papers: string[];
  sparse_score: number;
  semantic_distance: number;
}

export interface GapResponse {
  gaps: ResearchGap[];
  methodology: string;
  total_gaps_found: number;
}

export interface MetricRow {
  method: RetrievalMethod;
  precision_at_5: number;
  precision_at_10: number;
  recall_at_10: number;
  mrr: number;
  ndcg_at_10: number;
  latency_p50_ms: number;
  latency_p95_ms: number;
  memory_mb: number;
}

export interface EvaluationResponse {
  benchmark: MetricRow[];
  ablation: unknown[];
  significance: unknown[];
  num_queries: number;
  corpus_size: number;
  methodology_note: string;
}

export interface HealthResponse {
  status: string;
  version: string;
  db_connected: boolean;
  faiss_loaded: boolean;
  models_loaded: string[];
  corpus_size: number;
}
