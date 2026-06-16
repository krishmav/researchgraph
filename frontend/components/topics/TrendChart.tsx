// components/topics/TrendChart.tsx
"use client";
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
} from "recharts";
import type { TopicTrendPoint } from "@/lib/types";

interface Props {
  trend: TopicTrendPoint[];
  label: string;
}

export default function TrendChart({ trend, label }: Props) {
  if (!trend || trend.length === 0) {
    return (
      <p className="text-xs text-gray-600 py-8 text-center">
        No trend data available for this topic.
      </p>
    );
  }

  const data = trend.map((t) => ({
    month: t.year_month.slice(0, 7),
    papers: t.paper_count,
  }));

  // Compute trend line (simple moving average)
  const windowSize = 3;
  const dataWithMA = data.map((d, i) => {
    const window = data.slice(Math.max(0, i - windowSize + 1), i + 1);
    const ma = window.reduce((s, x) => s + x.papers, 0) / window.length;
    return { ...d, ma: parseFloat(ma.toFixed(1)) };
  });

  const maxPapers = Math.max(...data.map((d) => d.papers));

  return (
    <ResponsiveContainer width="100%" height={200}>
      <LineChart data={dataWithMA} margin={{ top: 5, right: 10, left: 0, bottom: 5 }}>
        <CartesianGrid stroke="#1f2937" strokeDasharray="3 3" />
        <XAxis
          dataKey="month"
          tick={{ fill: "#6b7280", fontSize: 10 }}
          tickLine={false}
          interval="preserveStartEnd"
        />
        <YAxis
          tick={{ fill: "#6b7280", fontSize: 10 }}
          tickLine={false}
          axisLine={false}
          width={30}
        />
        <Tooltip
          contentStyle={{
            backgroundColor: "#111827",
            border: "1px solid #374151",
            borderRadius: "8px",
            fontSize: "12px",
            color: "#f3f4f6",
          }}
          labelStyle={{ color: "#9ca3af" }}
        />
        {/* Raw counts */}
        <Line
          type="monotone"
          dataKey="papers"
          stroke="#6366f1"
          strokeWidth={1.5}
          dot={false}
          opacity={0.5}
          name="Papers"
        />
        {/* Moving average */}
        <Line
          type="monotone"
          dataKey="ma"
          stroke="#6366f1"
          strokeWidth={2.5}
          dot={false}
          name="3-mo avg"
        />
      </LineChart>
    </ResponsiveContainer>
  );
}
