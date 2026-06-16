// app/evaluate/page.tsx
"use client";
import { useEffect, useState } from "react";
import { getCachedEvaluation, runEvaluation } from "@/lib/api";
import type { EvaluationResponse, MetricRow } from "@/lib/types";
import {
  RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
  ResponsiveContainer, BarChart, Bar, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend,
} from "recharts";
import { LoadingSpinner, ErrorBanner } from "@/components/shared/PaperCard";
import { FlaskConical, Play, Download } from "lucide-react";

const METHOD_LABELS: Record<string, string> = {
  keyword: "Keyword",
  tfidf:   "TF-IDF",
  miniml:  "MiniLM",
  mpnet:   "MPNet",
  bge:     "BGE",
  graph:   "Graph+",
};

const METHOD_COLORS: Record<string, string> = {
  keyword: "#6b7280",
  tfidf:   "#3b82f6",
  miniml:  "#10b981",
  mpnet:   "#f59e0b",
  bge:     "#6366f1",
  graph:   "#ec4899",
};

export default function EvaluatePage() {
  const [data, setData]       = useState<EvaluationResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [running, setRunning] = useState(false);
  const [error, setError]     = useState<string | null>(null);

  useEffect(() => {
    getCachedEvaluation()
      .then(setData)
      .catch(() => {})
      .finally(() => setLoading(false));
  }, []);

  async function handleRun() {
    setRunning(true);
    setError(null);
    try {
      const res = await runEvaluation(100);
      setData(res);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Evaluation failed.");
    } finally {
      setRunning(false);
    }
  }

  function downloadCSV() {
    if (!data) return;
    const headers = Object.keys(data.benchmark[0] ?? {}).join(",");
    const rows = data.benchmark.map((r) => Object.values(r).join(","));
    const csv = [headers, ...rows].join("\n");
    const blob = new Blob([csv], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "researchgraph_benchmark.csv";
    a.click();
  }

  return (
    <div className="max-w-6xl mx-auto px-6 py-10">
      <div className="flex items-start justify-between mb-6">
        <div>
          <h1 className="text-2xl font-bold text-white mb-1">Evaluation Framework</h1>
          <p className="text-gray-400 text-sm">
            Rigorous benchmark across all retrieval methods.
            Pseudo-relevance labels from arXiv category overlap.
          </p>
        </div>
        <div className="flex gap-2">
          {data && (
            <button onClick={downloadCSV} className="btn-ghost text-xs flex items-center gap-1.5">
              <Download className="w-3.5 h-3.5" /> CSV
            </button>
          )}
          <button
            onClick={handleRun}
            disabled={running}
            className="btn-primary text-xs flex items-center gap-1.5"
          >
            {running ? (
              <><LoadingSpinner size="sm" /> Running…</>
            ) : (
              <><Play className="w-3.5 h-3.5" /> Run benchmark</>
            )}
          </button>
        </div>
      </div>

      {error && <div className="mb-4"><ErrorBanner message={error} /></div>}

      {loading && (
        <div className="flex items-center justify-center py-32">
          <LoadingSpinner size="lg" />
        </div>
      )}

      {!loading && !data && (
        <div className="card p-12 text-center">
          <FlaskConical className="w-12 h-12 text-gray-700 mx-auto mb-4" />
          <p className="text-gray-400 text-sm mb-2">No evaluation results yet.</p>
          <p className="text-gray-600 text-xs">
            Click "Run benchmark" to evaluate all retrieval methods.
            Requires ML models to be loaded.
          </p>
        </div>
      )}

      {data && (
        <>
          <MetaBar data={data} />
          <BenchmarkTable rows={data.benchmark} />
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mt-6">
            <NDCGBarChart rows={data.benchmark} />
            <RadarCompare rows={data.benchmark} />
          </div>
          <LatencyChart rows={data.benchmark} />
          {data.significance.length > 0 && (
            <SignificanceTable rows={data.significance as Array<{method_a:string;method_b:string;p_value:number;significant:boolean}>} />
          )}
          <MethodologyNote note={data.methodology_note} />
        </>
      )}
    </div>
  );
}

// ── Sub-components ─────────────────────────────────────────────────────────────

function MetaBar({ data }: { data: EvaluationResponse }) {
  return (
    <div className="grid grid-cols-3 gap-3 mb-6">
      {[
        { label: "Queries evaluated", value: data.num_queries.toLocaleString() },
        { label: "Corpus size",       value: data.corpus_size.toLocaleString() },
        { label: "Methods compared",  value: data.benchmark.length },
      ].map(({ label, value }) => (
        <div key={label} className="card p-3 text-center">
          <p className="text-lg font-bold text-white font-mono">{value}</p>
          <p className="text-xs text-gray-500 mt-0.5">{label}</p>
        </div>
      ))}
    </div>
  );
}

function BenchmarkTable({ rows }: { rows: MetricRow[] }) {
  const best = (key: keyof MetricRow) =>
    Math.max(...rows.map((r) => Number(r[key])));

  return (
    <div className="card overflow-hidden mb-4">
      <div className="overflow-x-auto">
        <table className="w-full text-xs">
          <thead>
            <tr className="bg-gray-800/50">
              {["Method","P@5","P@10","Recall@10","MRR","NDCG@10","p50 (ms)","p95 (ms)"].map((h) => (
                <th key={h} className="text-left px-4 py-2.5 text-gray-400 font-medium">
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr
                key={row.method}
                className="border-t border-gray-800 hover:bg-gray-800/30 transition-colors"
              >
                <td className="px-4 py-2.5">
                  <span
                    className="font-medium"
                    style={{ color: METHOD_COLORS[row.method] ?? "#fff" }}
                  >
                    {METHOD_LABELS[row.method] ?? row.method}
                  </span>
                </td>
                {(["precision_at_5","precision_at_10","recall_at_10","mrr","ndcg_at_10"] as const).map((k) => {
                  const val = row[k];
                  const isTop = Number(val) >= best(k) - 0.001;
                  return (
                    <td
                      key={k}
                      className={`px-4 py-2.5 font-mono ${
                        isTop ? "text-emerald-400 font-bold" : "text-gray-300"
                      }`}
                    >
                      {Number(val).toFixed(3)}
                    </td>
                  );
                })}
                <td className="px-4 py-2.5 font-mono text-gray-400">
                  {row.latency_p50_ms.toFixed(1)}
                </td>
                <td className="px-4 py-2.5 font-mono text-gray-400">
                  {row.latency_p95_ms.toFixed(1)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="text-xs text-gray-600 px-4 py-2 border-t border-gray-800">
        Bold = best in column. Metrics averaged over queries.
      </p>
    </div>
  );
}

function NDCGBarChart({ rows }: { rows: MetricRow[] }) {
  const data = rows.map((r) => ({
    name: METHOD_LABELS[r.method] ?? r.method,
    "NDCG@10": parseFloat(r.ndcg_at_10.toFixed(3)),
    "P@10":    parseFloat(r.precision_at_10.toFixed(3)),
    color: METHOD_COLORS[r.method],
  }));

  return (
    <div className="card p-4">
      <p className="text-xs font-medium text-gray-300 mb-3">NDCG@10 & Precision@10</p>
      <ResponsiveContainer width="100%" height={220}>
        <BarChart data={data} margin={{ top: 5, right: 10, left: 0, bottom: 5 }}>
          <CartesianGrid stroke="#1f2937" strokeDasharray="3 3" />
          <XAxis dataKey="name" tick={{ fill: "#6b7280", fontSize: 10 }} />
          <YAxis domain={[0, 1]} tick={{ fill: "#6b7280", fontSize: 10 }} />
          <Tooltip
            contentStyle={{ backgroundColor: "#111827", border: "1px solid #374151",
                            borderRadius: "8px", fontSize: "12px" }}
          />
          <Legend wrapperStyle={{ fontSize: "11px" }} />
          <Bar dataKey="NDCG@10" fill="#6366f1" radius={[4,4,0,0]} />
          <Bar dataKey="P@10"    fill="#10b981" radius={[4,4,0,0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

function RadarCompare({ rows }: { rows: MetricRow[] }) {
  const metrics = ["P@5", "P@10", "Recall", "MRR", "NDCG"];
  const data = metrics.map((m, i) => {
    const keys: (keyof MetricRow)[] = [
      "precision_at_5","precision_at_10","recall_at_10","mrr","ndcg_at_10"
    ];
    const entry: Record<string, string | number> = { metric: m };
    rows.forEach((r) => {
      entry[METHOD_LABELS[r.method] ?? r.method] = parseFloat(Number(r[keys[i]]).toFixed(3));
    });
    return entry;
  });

  const colors = Object.values(METHOD_COLORS);

  return (
    <div className="card p-4">
      <p className="text-xs font-medium text-gray-300 mb-3">Multi-metric Radar</p>
      <ResponsiveContainer width="100%" height={220}>
        <RadarChart data={data}>
          <PolarGrid stroke="#1f2937" />
          <PolarAngleAxis dataKey="metric" tick={{ fill: "#6b7280", fontSize: 10 }} />
          <PolarRadiusAxis domain={[0, 1]} tick={{ fill: "#6b7280", fontSize: 9 }} />
          {rows.map((r, i) => (
            <Radar
              key={r.method}
              name={METHOD_LABELS[r.method] ?? r.method}
              dataKey={METHOD_LABELS[r.method] ?? r.method}
              stroke={colors[i % colors.length]}
              fill={colors[i % colors.length]}
              fillOpacity={0.1}
            />
          ))}
          <Legend wrapperStyle={{ fontSize: "11px" }} />
        </RadarChart>
      </ResponsiveContainer>
    </div>
  );
}

function LatencyChart({ rows }: { rows: MetricRow[] }) {
  const data = rows.map((r) => ({
    name: METHOD_LABELS[r.method] ?? r.method,
    "p50 ms": parseFloat(r.latency_p50_ms.toFixed(1)),
    "p95 ms": parseFloat(r.latency_p95_ms.toFixed(1)),
  }));

  return (
    <div className="card p-4 mt-6">
      <p className="text-xs font-medium text-gray-300 mb-3">Query Latency (ms)</p>
      <ResponsiveContainer width="100%" height={180}>
        <BarChart data={data} margin={{ top: 5, right: 10, left: 0, bottom: 5 }}>
          <CartesianGrid stroke="#1f2937" strokeDasharray="3 3" />
          <XAxis dataKey="name" tick={{ fill: "#6b7280", fontSize: 10 }} />
          <YAxis tick={{ fill: "#6b7280", fontSize: 10 }} />
          <Tooltip
            contentStyle={{ backgroundColor: "#111827", border: "1px solid #374151",
                            borderRadius: "8px", fontSize: "12px" }}
          />
          <Legend wrapperStyle={{ fontSize: "11px" }} />
          <Bar dataKey="p50 ms" fill="#6366f1" radius={[4,4,0,0]} />
          <Bar dataKey="p95 ms" fill="#f59e0b" radius={[4,4,0,0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

function SignificanceTable({ rows }: { rows: Array<{method_a:string;method_b:string;p_value:number;significant:boolean}> }) {
  return (
    <div className="card overflow-hidden mt-6">
      <div className="px-4 py-3 border-b border-gray-800">
        <p className="text-xs font-medium text-gray-300">
          Statistical Significance (paired t-test vs keyword baseline, p &lt; 0.05)
        </p>
      </div>
      <table className="w-full text-xs">
        <thead>
          <tr className="bg-gray-800/40">
            <th className="text-left px-4 py-2 text-gray-400">Method</th>
            <th className="text-left px-4 py-2 text-gray-400">vs Baseline</th>
            <th className="text-left px-4 py-2 text-gray-400">p-value</th>
            <th className="text-left px-4 py-2 text-gray-400">Significant?</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.method_a} className="border-t border-gray-800">
              <td className="px-4 py-2 text-white font-medium">
                {METHOD_LABELS[r.method_a] ?? r.method_a}
              </td>
              <td className="px-4 py-2 text-gray-400">
                {METHOD_LABELS[r.method_b] ?? r.method_b}
              </td>
              <td className="px-4 py-2 font-mono text-gray-300">
                {r.p_value.toFixed(4)}
              </td>
              <td className="px-4 py-2">
                {r.significant ? (
                  <span className="badge bg-emerald-500/20 text-emerald-300 border border-emerald-500/30">
                    Yes (p&lt;0.05)
                  </span>
                ) : (
                  <span className="badge bg-gray-700 text-gray-400">No</span>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function MethodologyNote({ note }: { note: string }) {
  return (
    <div className="card p-4 mt-6 bg-blue-500/5 border-blue-500/20">
      <p className="text-xs text-gray-400 leading-relaxed">
        <span className="text-blue-300 font-medium">Evaluation methodology: </span>
        {note}
      </p>
    </div>
  );
}
