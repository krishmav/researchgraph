// app/page.tsx
"use client";
import { useState } from "react";
import { Search, Zap, GitCompare } from "lucide-react";
import SearchBar from "@/components/search/SearchBar";
import ResultsList from "@/components/search/ResultsList";
import MethodSelector from "@/components/search/MethodSelector";
import SideBySideView from "@/components/search/SideBySideView";
import type { SearchResponse, CompareSearchResponse, RetrievalMethod } from "@/lib/types";
import { search, compareSearch } from "@/lib/api";
import { LoadingSpinner, ErrorBanner } from "@/components/shared/PaperCard";

type Mode = "single" | "compare";

export default function HomePage() {
  const [mode, setMode]       = useState<Mode>("single");
  const [method, setMethod]   = useState<RetrievalMethod>("bge");
  const [query, setQuery]     = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError]     = useState<string | null>(null);
  const [result, setResult]   = useState<SearchResponse | null>(null);
  const [compareResult, setCompareResult] = useState<CompareSearchResponse | null>(null);

  async function handleSearch(q: string) {
    if (!q.trim()) return;
    setQuery(q);
    setLoading(true);
    setError(null);
    try {
      if (mode === "compare") {
        const res = await compareSearch(q, 10);
        setCompareResult(res);
        setResult(null);
      } else {
        const res = await search(q, method, 10);
        setResult(res);
        setCompareResult(null);
      }
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Search failed.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="max-w-4xl mx-auto px-6 py-10">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-white mb-1">
          Semantic Research Search
        </h1>
        <p className="text-gray-400 text-sm">
          Query 25k+ arXiv papers using transformer embeddings, TF-IDF, or keyword search.
        </p>
      </div>

      {/* Mode toggle */}
      <div className="flex items-center gap-2 mb-4">
        <button
          onClick={() => setMode("single")}
          className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium
                      transition-colors ${mode === "single"
                        ? "bg-brand-500/20 text-brand-300 border border-brand-500/30"
                        : "text-gray-500 hover:text-gray-300"}`}
        >
          <Zap className="w-3.5 h-3.5" /> Single method
        </button>
        <button
          onClick={() => setMode("compare")}
          className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium
                      transition-colors ${mode === "compare"
                        ? "bg-brand-500/20 text-brand-300 border border-brand-500/30"
                        : "text-gray-500 hover:text-gray-300"}`}
        >
          <GitCompare className="w-3.5 h-3.5" /> Compare all methods
        </button>
      </div>

      {/* Search bar */}
      <SearchBar onSearch={handleSearch} loading={loading} />

      {/* Method selector (single mode only) */}
      {mode === "single" && (
        <div className="mt-3">
          <MethodSelector selected={method} onChange={setMethod} />
        </div>
      )}

      {/* Error */}
      {error && <div className="mt-4"><ErrorBanner message={error} /></div>}

      {/* Loading */}
      {loading && (
        <div className="flex items-center gap-3 mt-8 text-gray-400 text-sm">
          <LoadingSpinner size="sm" />
          <span>
            {mode === "compare"
              ? "Running all retrieval methods…"
              : `Searching with ${method.toUpperCase()}…`}
          </span>
        </div>
      )}

      {/* Results */}
      {!loading && result && mode === "single" && (
        <div className="mt-6">
          <div className="flex items-center justify-between mb-3">
            <p className="text-xs text-gray-500">
              {result.total_results} results · {result.latency_ms.toFixed(1)} ms ·{" "}
              <span className="text-brand-400">{result.method.toUpperCase()}</span>
            </p>
          </div>
          <ResultsList results={result.results} />
        </div>
      )}

      {!loading && compareResult && mode === "compare" && (
        <div className="mt-6">
          <SideBySideView data={compareResult} />
        </div>
      )}

      {/* Empty state hero */}
      {!loading && !result && !compareResult && !error && (
        <div className="mt-16 text-center">
          <Search className="w-12 h-12 text-gray-700 mx-auto mb-4" />
          <p className="text-gray-500 text-sm">
            Try searching for{" "}
            {[
              '"graph neural networks"',
              '"transformer attention"',
              '"adversarial robustness"',
              '"federated learning privacy"',
            ].map((s, i) => (
              <button
                key={i}
                onClick={() => handleSearch(s.replace(/"/g, ""))}
                className="text-brand-400 hover:text-brand-300 mx-1 underline
                           underline-offset-2 transition-colors"
              >
                {s}
              </button>
            ))}
          </p>
        </div>
      )}
    </div>
  );
}
