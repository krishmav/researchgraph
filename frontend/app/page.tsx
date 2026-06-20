// app/page.tsx
"use client";
import { useState } from "react";
import SearchBar from "@/components/search/SearchBar";
import ResultsList from "@/components/search/ResultsList";
import type { SearchResponse } from "@/lib/types";
import { search } from "@/lib/api";
import { LoadingSpinner, ErrorBanner } from "@/components/shared/PaperCard";

export default function HomePage() {
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<SearchResponse | null>(null);

  async function handleSearch(q: string) {
    if (!q.trim()) return;
    setQuery(q);
    setLoading(true);
    setError(null);
    try {
      // Silently routing all searches to the optimal hybrid/dense pipeline
      const res = await search(q, "miniml" as any, 10);
      setResult(res);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Search failed.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="max-w-5xl mx-auto px-6 py-10">
      
      {/* SaaS Landing Hero (Only visible when no results and not loading) */}
      {!result && !loading && (
        <div className="flex flex-col items-center justify-center py-20 text-center">
          <h1 className="text-5xl font-extrabold text-white tracking-tight mb-4">
            ResearchGraph
          </h1>
          <p className="text-xl text-gray-400 mb-10 max-w-2xl">
            Search and explore 25,000+ research papers using semantic search.
          </p>
        </div>
      )}

      {/* Mini-Header for Results View */}
      {result && !loading && (
        <div className="mb-8">
          <h1 
            className="text-2xl font-bold text-white cursor-pointer inline-block"
            onClick={() => {
              setResult(null);
              setQuery("");
            }}
          >
            ResearchGraph
          </h1>
        </div>
      )}

      {/* Main Search Bar */}
      <div className={!result && !loading ? "max-w-3xl mx-auto w-full" : "w-full"}>
        <SearchBar onSearch={handleSearch} loading={loading} />
      </div>

      {/* Suggested Topics (Only visible on the landing page) */}
      {!loading && !result && !error && (
        <div className="mt-12 text-center">
          <div className="flex flex-wrap justify-center gap-3 max-w-3xl mx-auto">
            {[
              "Graph Neural Networks",
              "Large Language Models",
              "Reinforcement Learning",
              "Computer Vision",
              "Cybersecurity",
              "Multi-Agent Systems"
            ].map((topic) => (
              <button
                key={topic}
                onClick={() => handleSearch(topic)}
                className="px-4 py-2 bg-gray-800/40 text-gray-300 rounded-full text-sm font-medium hover:bg-gray-700/60 border border-gray-700/50 transition-all"
              >
                {topic}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Error Banner */}
      {error && <div className="mt-8 max-w-3xl mx-auto"><ErrorBanner message={error} /></div>}

      {/* Loading State */}
      {loading && (
        <div className="flex flex-col items-center justify-center mt-24 text-gray-400">
          <LoadingSpinner size="lg" />
          <span className="mt-4 text-sm font-medium">
            Searching 25,000+ indexed papers…
          </span>
        </div>
      )}

      {/* Results View */}
      {!loading && result && (
        <div className="mt-10">
          {/* Clean User-Focused Analytics */}
          <div className="flex items-center justify-between pb-4 mb-6 border-b border-gray-800 text-sm text-gray-400">
            <span className="font-semibold text-white">
              {result.total_results} Results Found
            </span>
            <div className="flex gap-6">
              <span>Search Time: {result.latency_ms.toFixed(1)}ms</span>
              <span>Indexed Corpus: 25,000+ Papers</span>
            </div>
          </div>
          
          <ResultsList results={result.results} />
        </div>
      )}

    </div>
  );
}