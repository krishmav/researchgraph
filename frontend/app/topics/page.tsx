// app/topics/page.tsx
"use client";
import { useEffect, useState } from "react";
import { getTrendingTopics, getTopicMap } from "@/lib/api";
import type { TopicWithTrend, TopicMapResponse } from "@/lib/types";
import TopicScatter from "@/components/topics/TopicScatter";
import TrendChart from "@/components/topics/TrendChart";
import { LoadingSpinner, ErrorBanner } from "@/components/shared/PaperCard";
import { TrendingUp, Layers } from "lucide-react";

export default function TopicsPage() {
  const [topics, setTopics]     = useState<TopicWithTrend[]>([]);
  const [mapData, setMapData]   = useState<TopicMapResponse | null>(null);
  const [selected, setSelected] = useState<TopicWithTrend | null>(null);
  const [loading, setLoading]   = useState(true);
  const [error, setError]       = useState<string | null>(null);

  useEffect(() => {
    Promise.all([getTrendingTopics(), getTopicMap(2000)])
      .then(([t, m]) => {
        setTopics(t);
        setMapData(m);
        if (t.length > 0) setSelected(t[0]);
      })
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center py-32">
        <LoadingSpinner size="lg" />
      </div>
    );
  }

  if (error) {
    return (
      <div className="max-w-4xl mx-auto px-6 py-10">
        <ErrorBanner message={error} />
      </div>
    );
  }

  const emerging = topics.filter((t) => t.is_emerging);

  return (
    <div className="max-w-6xl mx-auto px-6 py-10">
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-white mb-1">Topic Explorer</h1>
        <p className="text-gray-400 text-sm">
          {topics.length} research clusters discovered by BERTopic + UMAP + HDBSCAN
          across {mapData?.total_papers.toLocaleString()} papers.
        </p>
      </div>

      {/* Emerging topics banner */}
      {emerging.length > 0 && (
        <div className="mb-6 p-4 bg-amber-500/10 border border-amber-500/30 rounded-xl">
          <div className="flex items-center gap-2 mb-2">
            <TrendingUp className="w-4 h-4 text-amber-400" />
            <span className="text-sm font-medium text-amber-300">
              {emerging.length} Emerging Topic{emerging.length > 1 ? "s" : ""} Detected
            </span>
          </div>
          <div className="flex flex-wrap gap-2">
            {emerging.slice(0, 5).map((t) => (
              <button
                key={t.id}
                onClick={() => setSelected(t)}
                className="text-xs bg-amber-500/20 text-amber-300 border border-amber-500/30
                           px-2 py-1 rounded-lg hover:bg-amber-500/30 transition-colors"
              >
                {t.label.split(":")[1]?.trim() ?? t.label} ↑{(t.growth_slope ?? 0).toFixed(1)}/mo
              </button>
            ))}
          </div>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* UMAP scatter */}
        <div className="lg:col-span-2 card p-4">
          <div className="flex items-center gap-2 mb-3">
            <Layers className="w-4 h-4 text-gray-400" />
            <span className="text-sm font-medium text-gray-300">
              2-D UMAP Projection
            </span>
            <span className="text-xs text-gray-600 ml-auto">
              Click a point to inspect
            </span>
          </div>
          {mapData ? (
            <TopicScatter
              points={mapData.points}
              topics={mapData.topics}
              onTopicSelect={(tid) => {
                const t = topics.find((t) => t.id === tid);
                if (t) setSelected(t);
              }}
            />
          ) : (
            <div className="h-64 flex items-center justify-center text-gray-600 text-sm">
              Topic map not available. Run scripts/train_bertopic.py first.
            </div>
          )}
        </div>

        {/* Topic list */}
        <div className="card p-4 overflow-y-auto max-h-[520px]">
          <p className="text-xs text-gray-500 mb-3">All topics by paper count</p>
          <div className="space-y-1">
            {topics.map((t) => (
              <button
                key={t.id}
                onClick={() => setSelected(t)}
                className={`w-full text-left px-3 py-2 rounded-lg text-xs transition-colors
                            ${selected?.id === t.id
                              ? "bg-brand-500/20 text-brand-300 border border-brand-500/30"
                              : "text-gray-400 hover:bg-gray-800"}`}
              >
                <div className="flex items-center justify-between">
                  <span className="line-clamp-1 font-medium">
                    {t.label.split(":")[1]?.trim() ?? t.label}
                  </span>
                  <div className="flex items-center gap-1.5 shrink-0">
                    {t.is_emerging && (
                      <TrendingUp className="w-3 h-3 text-amber-400" />
                    )}
                    <span className="text-gray-600">{t.paper_count}</span>
                  </div>
                </div>
                <p className="text-gray-600 truncate mt-0.5">
                  {t.top_words.slice(0, 4).join(" · ")}
                </p>
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Selected topic trend chart */}
      {selected && (
        <div className="card p-6 mt-6">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h2 className="text-sm font-semibold text-white">
                {selected.label}
              </h2>
              <p className="text-xs text-gray-500 mt-0.5">
                {selected.paper_count} papers ·{" "}
                {selected.top_words.slice(0, 6).join(", ")}
              </p>
            </div>
            {selected.is_emerging && (
              <div className="flex items-center gap-1.5 bg-amber-500/20 border
                              border-amber-500/30 text-amber-300 text-xs px-2.5 py-1 rounded-lg">
                <TrendingUp className="w-3.5 h-3.5" />
                Emerging · +{(selected.growth_slope ?? 0).toFixed(1)} papers/mo
              </div>
            )}
          </div>
          <TrendChart trend={selected.trend} label={selected.label} />
        </div>
      )}
    </div>
  );
}
