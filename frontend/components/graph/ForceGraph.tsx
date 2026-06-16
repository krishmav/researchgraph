// components/graph/ForceGraph.tsx
"use client";
import { useEffect, useRef, useState } from "react";
import * as d3 from "d3";
import type { GraphNode, GraphEdge } from "@/lib/types";

interface Props {
  nodes: GraphNode[];
  edges: GraphEdge[];
  height?: number;
}

interface SimNode extends d3.SimulationNodeDatum, GraphNode {}
interface SimEdge extends d3.SimulationLinkDatum<SimNode> {
  edge_type: string;
  weight: number;
}

export default function ForceGraph({ nodes, edges, height = 500 }: Props) {
  const svgRef = useRef<SVGSVGElement>(null);
  const [tooltip, setTooltip] = useState<{
    x: number; y: number; content: string;
  } | null>(null);

  useEffect(() => {
    if (!svgRef.current || nodes.length === 0) return;

    const el = svgRef.current;
    const W = el.clientWidth || 700;
    const H = height;

    d3.select(el).selectAll("*").remove();

    const svg = d3.select(el)
      .attr("viewBox", `0 0 ${W} ${H}`)
      .attr("preserveAspectRatio", "xMidYMid meet");

    const zoomG = svg.append("g");
    svg.call(
      d3.zoom<SVGSVGElement, unknown>()
        .scaleExtent([0.3, 5])
        .on("zoom", (e) => zoomG.attr("transform", e.transform))
    );

    // Prep node and link data
    const nodeById = new Map(nodes.map((n) => [n.id, { ...n } as SimNode]));
    const simNodes: SimNode[] = [...nodeById.values()];

    const simEdges: SimEdge[] = edges
      .map((e) => ({
        source: nodeById.get(e.source)!,
        target: nodeById.get(e.target)!,
        edge_type: e.edge_type,
        weight: e.weight,
      }))
      .filter((e) => e.source && e.target);

    // Simulation
    const sim = d3.forceSimulation<SimNode>(simNodes)
      .force("link", d3.forceLink<SimNode, SimEdge>(simEdges)
        .id((d) => d.id)
        .distance((e) => 60 / (e.weight + 0.1))
        .strength(0.5)
      )
      .force("charge", d3.forceManyBody().strength(-120))
      .force("center", d3.forceCenter(W / 2, H / 2))
      .force("collision", d3.forceCollide<SimNode>((d) => d.size + 4));

    // Links
    const link = zoomG.append("g")
      .selectAll("line")
      .data(simEdges)
      .join("line")
      .attr("stroke", "#374151")
      .attr("stroke-width", (d) => Math.max(0.5, d.weight * 1.5))
      .attr("stroke-opacity", 0.6);

    // Nodes
    const node = zoomG.append("g")
      .selectAll("circle")
      .data(simNodes)
      .join("circle")
      .attr("r", (d) => Math.max(5, d.size / 2))
      .attr("fill", (d) => d.color)
      .attr("stroke", "#111827")
      .attr("stroke-width", 1.5)
      .attr("cursor", "pointer")
      .on("mouseenter", function (event, d) {
        d3.select(this).attr("stroke", "#ffffff").attr("stroke-width", 2);
        const rect = el.getBoundingClientRect();
        const meta = d.metadata as Record<string, unknown>;
        const content = d.node_type === "paper"
          ? `${d.label}\n${meta.primary_category ?? ""}`
          : d.node_type === "author"
          ? `Author: ${d.label}`
          : d.node_type === "topic"
          ? `Topic: ${d.label}`
          : `Area: ${d.label}`;
        setTooltip({
          x: event.clientX - rect.left,
          y: event.clientY - rect.top - 8,
          content,
        });
      })
      .on("mouseleave", function () {
        d3.select(this).attr("stroke", "#111827").attr("stroke-width", 1.5);
        setTooltip(null);
      })
      .call(
        // === FIX: Bypassing strict types for d3 drag behavior ===
        d3.drag<any, any>()
          .on("start", (event, d) => {
            if (!event.active) sim.alphaTarget(0.3).restart();
            d.fx = d.x; d.fy = d.y;
          })
          .on("drag", (event, d) => {
            d.fx = event.x; d.fy = event.y;
          })
          .on("end", (event, d) => {
            if (!event.active) sim.alphaTarget(0);
            d.fx = null; d.fy = null;
          }) as any
      );

    // Labels for large nodes
    const label = zoomG.append("g")
      .selectAll("text")
      .data(simNodes.filter((d) => d.size >= 15 || d.node_type !== "paper"))
      .join("text")
      .attr("font-size", "9px")
      .attr("fill", "#9ca3af")
      .attr("pointer-events", "none")
      .attr("text-anchor", "middle")
      .text((d) => d.label.substring(0, 20));

    // Tick
    sim.on("tick", () => {
      link
        .attr("x1", (d) => (d.source as SimNode).x!)
        .attr("y1", (d) => (d.source as SimNode).y!)
        .attr("x2", (d) => (d.target as SimNode).x!)
        .attr("y2", (d) => (d.target as SimNode).y!);

      node.attr("cx", (d) => d.x!).attr("cy", (d) => d.y!);
      label.attr("x", (d) => d.x!).attr("y", (d) => (d.y ?? 0) - (d.size / 2) - 4);
    });

    // === FIX: Wrapping stop in curly braces to return clean void ===
    return () => {
      sim.stop();
    };
  }, [nodes, edges, height]);

  return (
    <div className="relative w-full" style={{ height }}>
      <svg ref={svgRef} className="w-full h-full" />
      {tooltip && (
        <div
          className="absolute pointer-events-none bg-gray-800 border border-gray-700
                     text-xs text-white rounded-lg px-3 py-2 max-w-xs shadow-xl z-10
                     whitespace-pre-line"
          style={{ left: tooltip.x + 10, top: tooltip.y }}
        >
          {tooltip.content}
        </div>
      )}
    </div>
  );
}