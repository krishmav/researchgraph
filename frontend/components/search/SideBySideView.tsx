// components/search/SideBySideView.tsx
"use client";
import { useState } from "react";
import type { CompareSearchResponse, RetrievalMethod } from "@/lib/types";
import PaperCard from "@/components/shared/PaperCard";
import clsx from "clsx";

const METHOD_LABELS: Record<string, string> = {
  keyword: "Keyword",
  tfidf:   "TF-IDF",
  miniml:  "MiniLM",
  mpnet:   "MPNet",
  bge:     "BGE",
  graph:   "Graph+",
};

interface Props {
  data: CompareSearchResponse;
}

export default function SideBySideView({ data }: Props) {
  const methods = Object.keys(data.methods) as RetrievalMethod[];
  const [activeMethod, setActiveMethod] = useState<RetrievalMethod>(methods[0]);

  return (
    <div>
      {/* Method tabs */}
      <div className="flex flex-wrap gap-2 mb-4">
        {methods.map((m) => {
          const latency = data.latency_ms[m];
          const count = data.methods[m]?.length ?? 0;
          return (
            <button
              key={m}
              onClick={() => setActiveMethod(m)}
              className={clsx(
                "flex flex-col items-start px-3 py-2 rounded-lg border text-xs transition-colors",
                activeMethod === m
                  ? "bg-brand-500/20 border-brand-500/40 text-brand-300"
                  : "bg-gray-900 border-gray-700 text-gray-400 hover:border-gray-600"
              )}
            >
              <span className="font-semibold">{METHOD_LABELS[m] ?? m}</span>
              <span className="text-gray-500 mt-0.5">
                {count} results · {latency?.toFixed(1)}ms
              </span>
            </button>
          );
        })}
      </div>

      {/* Overlap stats */}
      <OverlapStats data={data} methods={methods} />

      {/* Results for active method */}
      <div className="space-y-3 mt-4">
        {(data.methods[activeMethod] ?? []).map((r) => (
          <PaperCard
            key={r.paper.arxiv_id}
            paper={r.paper}
            score={r.score}
            explanation={r.explanation}
            rank={r.rank}
          />
        ))}
      </div>
    </div>
  );
}

function OverlapStats({
  data,
  methods,
}: {
  data: CompareSearchResponse;
  methods: RetrievalMethod[];
}) {
  if (methods.length < 2) return null;

  // Compute pairwise Jaccard overlap between first two non-empty methods
  const m1 = methods[0];
  const m2 = methods[methods.length - 1];
  const set1 = new Set((data.methods[m1] ?? []).map((r) => r.paper.arxiv_id));
  const set2 = new Set((data.methods[m2] ?? []).map((r) => r.paper.arxiv_id));
  const intersection = [...set1].filter((id) => set2.has(id)).length;
  const union = new Set([...set1, ...set2]).size;
  const jaccard = union > 0 ? intersection / union : 0;

  return (
    <div className="flex flex-wrap gap-4 p-3 bg-gray-900/60 rounded-lg border border-gray-800 text-xs text-gray-400">
      <span>
        <span className="text-gray-300 font-medium">{METHOD_LABELS[m1]}</span>
        {" vs "}
        <span className="text-gray-300 font-medium">{METHOD_LABELS[m2]}</span>
      </span>
      <span>
        Overlap:{" "}
        <span className="text-emerald-400 font-medium">{intersection}</span>
        {" shared results"}
      </span>
      <span>
        Jaccard:{" "}
        <span className="text-brand-400 font-medium">{(jaccard * 100).toFixed(1)}%</span>
      </span>
      <span className="text-gray-600 italic">
        Low overlap = methods find different papers
      </span>
    </div>
  );
}
