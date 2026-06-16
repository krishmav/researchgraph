"use client";

import { Suspense } from "react";
import { useEffect, useState } from "react";
import { useSearchParams } from "next/navigation";
import { getPaperGraph, getGraphStats, getTopPapersByPageRank } from "@/lib/api";
import type { GraphResponse } from "@/lib/types";
import { LoadingSpinner, ErrorBanner } from "@/components/shared/PaperCard";
import ForceGraph from "@/components/graph/ForceGraph";
import { Network, Star } from "lucide-react";

// FIX: Renamed from GraphPage to GraphContent to avoid duplication
function GraphContent() {
  const params = useSearchParams();
  const initialPaper = params.get("paper") ?? "";

  const [arxivId, setArxivId]   = useState(initialPaper);
  const [graph, setGraph]       = useState<GraphResponse | null>(null);
  const [stats, setStats]       = useState<Record<string, unknown> | null>(null);
  const [topPapers, setTopPapers] = useState<unknown[]>([]);
  const [loading, setLoading]   = useState(false);
  const [error, setError]       = useState<string | null>(null);
  const [radius, setRadius]     = useState(2);

  useEffect(() => {
    getGraphStats().then(setStats).catch(() => {});
    getTopPapersByPageRank(10).then((data: any) => setTopPapers(data)).catch(() => {});
  }, []);

  useEffect(() => {
    if (initialPaper) loadGraph(initialPaper);
  }, [initialPaper]);

  async function loadGraph(id: string) {
    if (!id.trim()) return;
    setArxivId(id.trim());
     // Fixed state assignment syntax if needed, but keeping your original functional layout
    setLoading(true);
    setError(null);
    try {
      const g = await getPaperGraph(id.trim(), radius);
      setGraph(g);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Failed to load graph.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="max-w-6xl mx-auto px-6 py-10">
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-white mb-1">Knowledge Graph</h1>
        <p className="text-gray-400 text-sm">
          Paper–Author–Topic relationships. Enter an arXiv ID to explore a subgraph.
        </p>
      </div>

      {/* Stats bar */}
      {stats && (
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-6">
          {[
            { label: "Total nodes",  value: stats.num_nodes },
            { label: "Total edges",  value: stats.num_edges },
            { label: "Paper nodes",  value: stats.num_paper_nodes },
            { label: "Communities",  value: stats.num_communities },
          ].map(({ label, value }) => (
            <div key={label} className="card p-3 text-center">
              <p className="text-lg font-bold text-white font-mono">
                {(value as number)?.toLocaleString() ?? "—"}
              </p>
              <p className="text-xs text-gray-500 mt-0.5">{label}</p>
            </div>
          ))}
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        {/* Controls */}
        <div className="lg:col-span-1 space-y-4">
          <div className="card p-4">
            <p className="text-xs text-gray-500 mb-2 font-medium">
              Enter arXiv ID
            </p>
            <input
              type="text"
              placeholder="e.g. 2104.12345"
              defaultValue={initialPaper}
              onKeyDown={(e) => {
                if (e.key === "Enter") loadGraph((e.target as HTMLInputElement).value);
              }}
              className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2
                         text-sm text-white placeholder-gray-600 focus:outline-none
                         focus:border-brand-500 transition-colors"
            />
            <div className="mt-3">
              <p className="text-xs text-gray-500 mb-1">Graph radius</p>
              <div className="flex gap-2">
                {[1, 2, 3].map((r) => (
                  <button
                    key={r}
                    onClick={() => setRadius(r)}
                    className={`flex-1 py-1 rounded text-xs font-medium transition-colors
                                ${radius === r
                                  ? "bg-brand-500/20 text-brand-300 border border-brand-500/30"
                                  : "bg-gray-800 text-gray-500 border border-gray-700"}`}
                  >
                    {r}-hop
                  </button>
                ))}
              </div>
            </div>
            <button
              onClick={() => loadGraph(arxivId)}
              disabled={!arxivId || loading}
              className="btn-primary w-full mt-3 text-xs justify-center"
            >
              {loading ? "Loading…" : "Load Graph"}
            </button>
          </div>

          {/* Top papers by PageRank */}
          {topPapers.length > 0 && (
            <div className="card p-4">
              <div className="flex items-center gap-1.5 mb-3">
                <Star className="w-3.5 h-3.5 text-amber-400" />
                <p className="text-xs font-medium text-gray-300">Top by PageRank</p>
              </div>
              <div className="space-y-2">
                {(topPapers as Array<{arxiv_id:string;title:string;pagerank:number}>)
                  .slice(0, 8)
                  .map((p) => (
                  <button
                    key={p.arxiv_id}
                    onClick={() => loadGraph(p.arxiv_id)}
                    className="w-full text-left text-xs text-gray-400 hover:text-brand-300
                               transition-colors truncate block"
                    title={p.title}
                  >
                    <span className="font-mono text-gray-600">{p.arxiv_id}</span>
                    {" "}
                    <span className="line-clamp-1">{p.title?.substring(0, 40)}</span>
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Graph canvas */}
        <div className="lg:col-span-3 card p-4">
          {error && <ErrorBanner message={error} />}

          {loading && (
            <div className="flex items-center justify-center py-32">
              <LoadingSpinner size="lg" />
            </div>
          )}

          {!loading && !graph && !error && (
            <div className="flex flex-col items-center justify-center py-24 text-center">
              <Network className="w-12 h-12 text-gray-700 mb-4" />
              <p className="text-gray-500 text-sm">
                Enter an arXiv ID or select a top paper to visualise its knowledge graph neighbourhood.
              </p>
            </div>
          )}

          {!loading && graph && (
            <>
              <div className="flex items-center gap-3 mb-3 text-xs text-gray-500">
                <span>{graph.stats.total_nodes as number} nodes</span>
                <span>·</span>
                <span>{graph.stats.total_edges as number} edges</span>
                <span>·</span>
                <span>radius {graph.stats.radius as number}</span>
                <NodeLegend />
              </div>
              <ForceGraph nodes={graph.nodes} edges={graph.edges} />
            </>
          )}
        </div>
      </div>
    </div>
  );
}

function NodeLegend() {
  const items = [
    { color: "#6366f1", label: "Paper" },
    { color: "#10b981", label: "Author" },
    { color: "#f59e0b", label: "Topic" },
    { color: "#ef4444", label: "Area" },
  ];
  return (
    <div className="ml-auto flex items-center gap-3">
      {items.map(({ color, label }) => (
        <div key={label} className="flex items-center gap-1">
          <div
            className="w-2.5 h-2.5 rounded-full"
            style={{ backgroundColor: color }}
          />
          <span className="text-gray-500">{label}</span>
        </div>
      ))}
    </div>
  );
}

// Default export handles the Suspense wrapper for Vercel building
export default function GraphPage() {
  return (
    <Suspense fallback={<div className="flex w-full h-screen items-center justify-center text-gray-400">Loading graph data...</div>}>
      <GraphContent />
    </Suspense>
  );
}