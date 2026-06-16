// components/shared/PaperCard.tsx
import Link from "next/link";
import type { Paper } from "@/lib/types";
import clsx from "clsx";

interface PaperCardProps {
  paper: Paper;
  score?: number;
  explanation?: string;
  rank?: number;
  compact?: boolean;
}

const CATEGORY_COLORS: Record<string, string> = {
  "cs.LG": "bg-violet-500/20 text-violet-300 border-violet-500/30",
  "cs.CV": "bg-blue-500/20 text-blue-300 border-blue-500/30",
  "cs.CL": "bg-emerald-500/20 text-emerald-300 border-emerald-500/30",
  "cs.AI": "bg-amber-500/20 text-amber-300 border-amber-500/30",
  "cs.CR": "bg-red-500/20 text-red-300 border-red-500/30",
  "cs.SE": "bg-pink-500/20 text-pink-300 border-pink-500/30",
  "stat.ML": "bg-cyan-500/20 text-cyan-300 border-cyan-500/30",
};

function getCategoryColor(cat: string) {
  return CATEGORY_COLORS[cat] ?? "bg-gray-500/20 text-gray-300 border-gray-500/30";
}

export default function PaperCard({
  paper,
  score,
  explanation,
  rank,
  compact = false,
}: PaperCardProps) {
  const year = paper.submitted_date
    ? new Date(paper.submitted_date).getFullYear()
    : "";

  return (
    <div className="card p-4 hover:border-gray-700 transition-colors group">
      <div className="flex items-start gap-3">
        {rank && (
          <span className="shrink-0 w-6 h-6 rounded-full bg-gray-800 flex items-center
                           justify-center text-xs text-gray-500 font-mono mt-0.5">
            {rank}
          </span>
        )}
        <div className="flex-1 min-w-0">
          {/* Title */}
          <Link
            href={`/paper/${paper.arxiv_id}`}
            className="text-sm font-semibold text-white group-hover:text-brand-300
                       transition-colors line-clamp-2 leading-snug block"
          >
            {paper.title}
          </Link>

          {/* Meta row */}
          <div className="flex flex-wrap items-center gap-2 mt-1.5">
            {paper.primary_category && (
              <span
                className={clsx(
                  "badge border",
                  getCategoryColor(paper.primary_category)
                )}
              >
                {paper.primary_category}
              </span>
            )}
            {year && (
              <span className="text-xs text-gray-500">{year}</span>
            )}
            {paper.authors.length > 0 && (
              <span className="text-xs text-gray-500 truncate max-w-[200px]">
                {paper.authors[0]}
                {paper.authors.length > 1 && ` +${paper.authors.length - 1}`}
              </span>
            )}
            {score !== undefined && (
              <span className="ml-auto text-xs font-mono text-emerald-400">
                {score.toFixed(4)}
              </span>
            )}
          </div>

          {/* Abstract (truncated) */}
          {!compact && (
            <p className="text-xs text-gray-400 mt-2 line-clamp-2 leading-relaxed">
              {paper.abstract}
            </p>
          )}

          {/* Explanation */}
          {explanation && (
            <p className="text-xs text-brand-400/80 mt-2 italic leading-relaxed">
              {explanation}
            </p>
          )}

          {/* Links */}
          <div className="flex items-center gap-3 mt-2.5">
            <a
              href={`https://arxiv.org/abs/${paper.arxiv_id}`}
              target="_blank"
              rel="noopener noreferrer"
              className="text-xs text-gray-500 hover:text-brand-300 transition-colors"
            >
              arXiv ↗
            </a>
            {paper.pdf_url && (
              <a
                href={paper.pdf_url}
                target="_blank"
                rel="noopener noreferrer"
                className="text-xs text-gray-500 hover:text-brand-300 transition-colors"
              >
                PDF ↗
              </a>
            )}
            <Link
              href={`/paper/${paper.arxiv_id}`}
              className="ml-auto text-xs text-gray-500 hover:text-brand-300 transition-colors"
            >
              Details & Recommendations →
            </Link>
          </div>
        </div>
      </div>
    </div>
  );
}

// ── Badge ──────────────────────────────────────────────────────────────────────

interface BadgeProps {
  children: React.ReactNode;
  variant?: "default" | "success" | "warning" | "error" | "info";
}

const BADGE_VARIANTS: Record<string, string> = {
  default: "bg-gray-700 text-gray-300",
  success: "bg-emerald-500/20 text-emerald-300 border border-emerald-500/30",
  warning: "bg-amber-500/20 text-amber-300 border border-amber-500/30",
  error:   "bg-red-500/20 text-red-300 border border-red-500/30",
  info:    "bg-blue-500/20 text-blue-300 border border-blue-500/30",
};

export function Badge({ children, variant = "default" }: BadgeProps) {
  return (
    <span className={clsx("badge", BADGE_VARIANTS[variant])}>
      {children}
    </span>
  );
}

// ── LoadingSpinner ─────────────────────────────────────────────────────────────

export function LoadingSpinner({ size = "md" }: { size?: "sm" | "md" | "lg" }) {
  const sizes = { sm: "w-4 h-4", md: "w-6 h-6", lg: "w-10 h-10" };
  return (
    <div
      className={clsx(
        "border-2 border-gray-700 border-t-brand-500 rounded-full animate-spin",
        sizes[size]
      )}
    />
  );
}

// ── Empty State ────────────────────────────────────────────────────────────────

export function EmptyState({
  title,
  description,
  icon,
}: {
  title: string;
  description?: string;
  icon?: React.ReactNode;
}) {
  return (
    <div className="flex flex-col items-center justify-center py-16 text-center">
      {icon && <div className="mb-4 text-gray-600">{icon}</div>}
      <p className="text-gray-400 font-medium">{title}</p>
      {description && (
        <p className="text-gray-600 text-sm mt-1 max-w-sm">{description}</p>
      )}
    </div>
  );
}

// ── Error Banner ──────────────────────────────────────────────────────────────

export function ErrorBanner({ message }: { message: string }) {
  return (
    <div className="rounded-lg bg-red-500/10 border border-red-500/30 px-4 py-3">
      <p className="text-sm text-red-400">{message}</p>
    </div>
  );
}
