// app/paper/[arxiv_id]/page.tsx
"use client";
import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import Link from "next/link";
import {
  ArrowLeft, ExternalLink, Network, BookOpen, Users, Tag,
} from "lucide-react";
import { getPaper, getRecommendations } from "@/lib/api";
import type { PaperDetail, RecommendResponse } from "@/lib/types";
import PaperCard from "@/components/shared/PaperCard";
import { LoadingSpinner, Badge, ErrorBanner } from "@/components/shared/PaperCard";

export default function PaperPage() {
  const { arxiv_id } = useParams<{ arxiv_id: string }>();
  const [paper, setPaper]       = useState<PaperDetail | null>(null);
  const [recs, setRecs]         = useState<RecommendResponse | null>(null);
  const [recMethod, setRecMethod] = useState<"content" | "graph">("graph");
  const [loading, setLoading]   = useState(true);
  const [recLoading, setRecLoading] = useState(false);
  const [error, setError]       = useState<string | null>(null);

  useEffect(() => {
    if (!arxiv_id) return;
    setLoading(true);
    getPaper(arxiv_id)
      .then(setPaper)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, [arxiv_id]);

  useEffect(() => {
    if (!paper) return;
    setRecLoading(true);
    getRecommendations(paper.id, recMethod, 8)
      .then(setRecs)
      .catch(() => {})
      .finally(() => setRecLoading(false));
  }, [paper, recMethod]);

  if (loading) {
    return (
      <div className="flex items-center justify-center py-32">
        <LoadingSpinner size="lg" />
      </div>
    );
  }

  if (error || !paper) {
    return (
      <div className="max-w-3xl mx-auto px-6 py-10">
        <ErrorBanner message={error ?? "Paper not found."} />
        <Link href="/" className="text-brand-400 text-sm mt-4 inline-block">
          ← Back to search
        </Link>
      </div>
    );
  }

  const year = new Date(paper.submitted_date).getFullYear();

  return (
    <div className="max-w-3xl mx-auto px-6 py-10">
      {/* Back */}
      <Link
        href="/"
        className="flex items-center gap-1.5 text-gray-500 hover:text-gray-300
                   text-sm mb-6 transition-colors"
      >
        <ArrowLeft className="w-4 h-4" /> Back to search
      </Link>

      {/* Paper header */}
      <div className="card p-6 mb-6">
        <div className="flex flex-wrap gap-2 mb-3">
          {paper.categories.slice(0, 4).map((cat) => (
            <span
              key={cat}
              className="badge bg-brand-500/20 text-brand-300 border border-brand-500/30"
            >
              {cat}
            </span>
          ))}
          <span className="badge bg-gray-700 text-gray-300">{year}</span>
        </div>

        <h1 className="text-lg font-bold text-white leading-snug mb-3">
          {paper.title}
        </h1>

        {/* Authors */}
        <div className="flex items-start gap-2 mb-4">
          <Users className="w-4 h-4 text-gray-500 shrink-0 mt-0.5" />
          <p className="text-sm text-gray-400">
            {paper.authors.slice(0, 6).join(", ")}
            {paper.authors.length > 6 && ` +${paper.authors.length - 6} more`}
          </p>
        </div>

        {/* Abstract */}
        <div className="flex items-start gap-2">
          <BookOpen className="w-4 h-4 text-gray-500 shrink-0 mt-0.5" />
          <p className="text-sm text-gray-300 leading-relaxed">{paper.abstract}</p>
        </div>

        {/* Topic */}
        {paper.topic_label && (
          <div className="flex items-center gap-2 mt-4 pt-4 border-t border-gray-800">
            <Tag className="w-4 h-4 text-amber-400" />
            <span className="text-xs text-amber-300">{paper.topic_label}</span>
            {paper.topic_top_words && (
              <span className="text-xs text-gray-500">
                · {paper.topic_top_words.slice(0, 5).join(", ")}
              </span>
            )}
          </div>
        )}

        {/* Links */}
        <div className="flex gap-3 mt-4 pt-4 border-t border-gray-800">
          <a
            href={`https://arxiv.org/abs/${paper.arxiv_id}`}
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-1.5 text-xs text-gray-400 hover:text-brand-300
                       transition-colors"
          >
            <ExternalLink className="w-3.5 h-3.5" />
            arXiv: {paper.arxiv_id}
          </a>
          {paper.pdf_url && (
            <a
              href={paper.pdf_url}
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-1.5 text-xs text-gray-400 hover:text-brand-300
                         transition-colors"
            >
              <ExternalLink className="w-3.5 h-3.5" />
              PDF
            </a>
          )}
          <Link
            href={`/graph?paper=${paper.arxiv_id}`}
            className="flex items-center gap-1.5 text-xs text-gray-400 hover:text-brand-300
                       transition-colors ml-auto"
          >
            <Network className="w-3.5 h-3.5" />
            View in Knowledge Graph
          </Link>
        </div>
      </div>

      {/* Recommendations */}
      <div>
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-sm font-semibold text-white">Similar Papers</h2>
          <div className="flex gap-2">
            {(["content", "graph"] as const).map((m) => (
              <button
                key={m}
                onClick={() => setRecMethod(m)}
                className={`text-xs px-3 py-1.5 rounded-lg border transition-colors ${
                  recMethod === m
                    ? "bg-brand-500/20 text-brand-300 border-brand-500/40"
                    : "text-gray-500 border-gray-700 hover:border-gray-600"
                }`}
              >
                {m === "content" ? "Content-based" : "Graph-enhanced"}
              </button>
            ))}
          </div>
        </div>

        {recLoading && (
          <div className="flex items-center gap-2 text-gray-500 text-sm py-4">
            <LoadingSpinner size="sm" />
            Finding similar papers…
          </div>
        )}

        {!recLoading && recs && (
          <div className="space-y-3">
            {recs.recommendations.map((r) => (
              <PaperCard
                key={r.paper.arxiv_id}
                paper={r.paper}
                score={r.score}
                explanation={r.explanation}
                compact
              />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
