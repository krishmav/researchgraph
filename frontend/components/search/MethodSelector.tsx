// components/search/MethodSelector.tsx
"use client";
import type { RetrievalMethod } from "@/lib/types";
import clsx from "clsx";

const METHODS: { key: RetrievalMethod; label: string; description: string }[] = [
  { key: "keyword", label: "Keyword",  description: "PostgreSQL full-text (BM25)" },
  { key: "tfidf",   label: "TF-IDF",   description: "Sparse vector retrieval" },
  { key: "miniml",  label: "MiniLM",   description: "384-dim dense embeddings" },
  { key: "mpnet",   label: "MPNet",    description: "768-dim dense embeddings" },
  { key: "bge",     label: "BGE",      description: "1024-dim SOTA embeddings" },
  { key: "graph",   label: "Graph+",   description: "BGE + KG re-ranking" },
];

interface Props {
  selected: RetrievalMethod;
  onChange: (m: RetrievalMethod) => void;
}

export default function MethodSelector({ selected, onChange }: Props) {
  return (
    <div className="flex flex-wrap gap-2">
      {METHODS.map(({ key, label, description }) => (
        <button
          key={key}
          onClick={() => onChange(key)}
          title={description}
          className={clsx(
            "px-3 py-1.5 rounded-lg text-xs font-medium border transition-colors",
            selected === key
              ? "bg-brand-500/20 text-brand-300 border-brand-500/40"
              : "bg-gray-900 text-gray-400 border-gray-700 hover:border-gray-600 hover:text-gray-200"
          )}
        >
          {label}
        </button>
      ))}
    </div>
  );
}
