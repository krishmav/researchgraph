// app/gaps/page.tsx
"use client";
import { useEffect, useState } from "react";
import { getResearchGaps } from "@/lib/api";
import type { GapResponse, ResearchGap } from "@/lib/types";
import { LoadingSpinner, ErrorBanner } from "@/components/shared/PaperCard";
import { Lightbulb, TrendingUp, AlertCircle } from "lucide-react";

export default function GapsPage() {
  const [data, setData]         = useState<GapResponse | null>(null);
  const [loading, setLoading]   = useState(true);
  const [error, setError]       = useState<string | null>(null);
  const [strategy, setStrategy] = useState<"both" | "sparse" | "structural">("both");

  useEffect(() => {
    setLoading(true);
    getResearchGaps(strategy, 15)
      .then(setData)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, [strategy]);

  return (
    <div className="max-w-4xl mx-auto px-6 py-10">
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-white mb-1">Research Gap Discovery</h1>
        <p className="text-gray-400 text-sm">
          Under-explored areas identified from sparse embedding regions and
          weakly-connected topic pairs in the knowledge graph.
        </p>
      </div>

      {/* Strategy selector */}
      <div className="flex gap-2 mb-6">
        {(["both", "sparse", "structural"] as const).map((s) => (
          <button
            key={s}
            onClick={() => setStrategy(s)}
            className={`px-3 py-1.5 rounded-lg text-xs font-medium border transition-colors
                        ${strategy === s
                          ? "bg-brand-500/20 text-brand-300 border-brand-500/40"
                          : "text-gray-400 border-gray-700 hover:border-gray-600"}`}
          >
            {s === "both" ? "Both methods" : s === "sparse" ? "Sparse regions" : "Graph gaps"}
          </button>
        ))}
      </div>

      {loading && (
        <div className="flex items-center gap-2 text-gray-500 text-sm py-8">
          <LoadingSpinner size="sm" /> Analysing research landscape…
        </div>
      )}

      {error && <ErrorBanner message={error} />}

      {!loading && data && (
        <>
          {/* Methodology note */}
          <div className="card p-4 mb-6 bg-blue-500/5 border-blue-500/20">
            <div className="flex items-start gap-2">
              <AlertCircle className="w-4 h-4 text-blue-400 shrink-0 mt-0.5" />
              <p className="text-xs text-gray-400 leading-relaxed">
                <span className="text-blue-300 font-medium">Methodology: </span>
                {data.methodology}
              </p>
            </div>
          </div>

          <p className="text-xs text-gray-500 mb-4">
            {data.total_gaps_found} research gaps identified
          </p>

          <div className="space-y-4">
            {data.gaps.map((gap) => (
              <GapCard key={gap.gap_id} gap={gap} />
            ))}
          </div>

          {data.gaps.length === 0 && (
            <div className="text-center py-16 text-gray-600 text-sm">
              No research gaps found. Run{" "}
              <code className="font-mono bg-gray-800 px-1 rounded">
                scripts/build_knowledge_graph.py
              </code>{" "}
              to compute gaps.
            </div>
          )}
        </>
      )}
    </div>
  );
}

function GapCard({ gap }: { gap: ResearchGap }) {
  const sparsityPct = Math.round(gap.sparse_score * 100);

  return (
    <div className="card p-5 hover:border-gray-700 transition-colors">
      <div className="flex items-start gap-3">
        <div className="shrink-0 w-8 h-8 rounded-lg bg-amber-500/20 flex items-center
                        justify-center border border-amber-500/30">
          <Lightbulb className="w-4 h-4 text-amber-400" />
        </div>
        <div className="flex-1">
          <div className="flex items-center justify-between mb-1">
            <span className="text-xs font-medium text-gray-400">
              Gap #{gap.gap_id}
            </span>
            <div className="flex items-center gap-3">
              <SparsityMeter score={sparsityPct} />
            </div>
          </div>

          <p className="text-sm text-white leading-relaxed">{gap.description}</p>

          {/* Flanking topics */}
          {gap.flanking_topics.length > 0 && (
            <div className="flex flex-wrap gap-2 mt-3">
              {gap.flanking_topics.map((topic, i) => (
                <span
                  key={i}
                  className="badge bg-amber-500/10 text-amber-300 border border-amber-500/20"
                >
                  {topic.split(":")[1]?.trim() ?? topic}
                </span>
              ))}
            </div>
          )}

          {/* Metrics */}
          <div className="flex gap-4 mt-3 text-xs text-gray-500">
            <span>
              Sparsity:{" "}
              <span className="text-amber-400 font-medium">{sparsityPct}%</span>
            </span>
            {gap.semantic_distance > 0 && (
              <span>
                Semantic distance:{" "}
                <span className="text-brand-400 font-medium">
                  {gap.semantic_distance.toFixed(3)}
                </span>
              </span>
            )}
            {gap.evidence_papers.length > 0 && (
              <span>
                Nearest papers:{" "}
                <span className="text-gray-400">
                  {gap.evidence_papers.slice(0, 2).join(", ")}
                </span>
              </span>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

function SparsityMeter({ score }: { score: number }) {
  return (
    <div className="flex items-center gap-1.5">
      <div className="w-20 h-1.5 bg-gray-800 rounded-full overflow-hidden">
        <div
          className="h-full rounded-full bg-gradient-to-r from-emerald-500 to-amber-500"
          style={{ width: `${score}%` }}
        />
      </div>
      <span className="text-xs text-gray-400 font-mono">{score}%</span>
    </div>
  );
}
