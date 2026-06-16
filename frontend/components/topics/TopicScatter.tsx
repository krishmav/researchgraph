// components/topics/TopicScatter.tsx
"use client";
import { useEffect, useRef, useState } from "react";
import * as d3 from "d3";
import type { TopicMapPoint, Topic } from "@/lib/types";

interface Props {
  points: TopicMapPoint[];
  topics: Topic[];
  onTopicSelect?: (topicId: number) => void;
}

const PALETTE = [
  "#6366f1","#10b981","#f59e0b","#ef4444","#3b82f6",
  "#ec4899","#14b8a6","#f97316","#a855f7","#06b6d4",
  "#84cc16","#f43f5e","#0ea5e9","#d946ef","#22c55e",
];

export default function TopicScatter({ points, topics, onTopicSelect }: Props) {
  const svgRef = useRef<SVGSVGElement>(null);
  const [tooltip, setTooltip] = useState<{
    x: number; y: number; title: string; topic: string;
  } | null>(null);

  // Assign stable colour per topic id
  const topicColor = (tid: number) =>
    PALETTE[Math.abs(tid) % PALETTE.length];

  useEffect(() => {
    if (!svgRef.current || points.length === 0) return;

    const el = svgRef.current;
    const W = el.clientWidth || 600;
    const H = 400;

    d3.select(el).selectAll("*").remove();

    const svg = d3.select(el)
      .attr("viewBox", `0 0 ${W} ${H}`)
      .attr("preserveAspectRatio", "xMidYMid meet");

    const xExtent = d3.extent(points, (p) => p.x) as [number, number];
    const yExtent = d3.extent(points, (p) => p.y) as [number, number];

    const xScale = d3.scaleLinear().domain(xExtent).range([20, W - 20]);
    const yScale = d3.scaleLinear().domain(yExtent).range([H - 20, 20]);

    // Zoom
    const zoomG = svg.append("g");
    svg.call(
      d3.zoom<SVGSVGElement, unknown>()
        .scaleExtent([0.5, 8])
        .on("zoom", (event) => zoomG.attr("transform", event.transform))
    );

    // Points
    zoomG.selectAll("circle")
      .data(points)
      .join("circle")
      .attr("cx", (d) => xScale(d.x))
      .attr("cy", (d) => yScale(d.y))
      .attr("r", 3)
      .attr("fill", (d) => topicColor(d.topic_id))
      .attr("opacity", 0.65)
      .attr("stroke", "none")
      .on("mouseenter", function (event, d) {
        d3.select(this).attr("r", 6).attr("opacity", 1);
        const rect = el.getBoundingClientRect();
        setTooltip({
          x: event.clientX - rect.left,
          y: event.clientY - rect.top - 10,
          title: d.title.substring(0, 80),
          topic: d.topic_label,
        });
      })
      .on("mouseleave", function () {
        d3.select(this).attr("r", 3).attr("opacity", 0.65);
        setTooltip(null);
      })
      .on("click", (_event, d) => {
        onTopicSelect?.(d.topic_id);
      });

    // Topic centroids (text labels for biggest clusters)
    const topicCentroids = d3.rollup(
      points,
      (v) => ({
        x: d3.mean(v, (d) => xScale(d.x))!,
        y: d3.mean(v, (d) => yScale(d.y))!,
        label: v[0].topic_label,
        tid: v[0].topic_id,
        count: v.length,
      }),
      (d) => d.topic_id
    );

    const centroidsArr = [...topicCentroids.values()]
      .sort((a, b) => b.count - a.count)
      .slice(0, 12);

    zoomG.selectAll("text.centroid")
      .data(centroidsArr)
      .join("text")
      .attr("class", "centroid")
      .attr("x", (d) => d.x)
      .attr("y", (d) => d.y - 6)
      .attr("text-anchor", "middle")
      .attr("font-size", "9px")
      .attr("fill", (d) => topicColor(d.tid))
      .attr("pointer-events", "none")
      .text((d) => d.label.split(":")[1]?.trim().substring(0, 18) ?? "");

  }, [points]);

  return (
    <div className="relative w-full">
      <svg ref={svgRef} className="w-full" style={{ height: 400 }} />
      {tooltip && (
        <div
          className="absolute pointer-events-none bg-gray-800 border border-gray-700
                     text-xs text-white rounded-lg px-3 py-2 max-w-xs shadow-xl z-10"
          style={{ left: tooltip.x + 8, top: tooltip.y }}
        >
          <p className="font-medium leading-snug">{tooltip.title}</p>
          <p className="text-brand-300 mt-1">{tooltip.topic}</p>
        </div>
      )}
      <p className="text-xs text-gray-600 mt-2 text-right">
        Scroll to zoom · drag to pan · click to select topic
      </p>
    </div>
  );
}
