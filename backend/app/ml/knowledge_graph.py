"""
Module 4: Knowledge Graph System
==================================
Constructs a multi-relational graph:
  Nodes : Papers, Authors, Topics, Research Areas
  Edges : authored_by, similar_to, belongs_to_topic, co_authored

Analytics:
  - PageRank
  - Louvain community detection
  - Betweenness centrality
  - Eigenvector centrality

Serialised as NetworkX MultiDiGraph pickle.
PyVis renders subgraphs to interactive HTML for the frontend.
"""
from __future__ import annotations

import json
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np

try:
    from community import best_partition as louvain_partition   # python-louvain
    _LOUVAIN_AVAILABLE = True
except ImportError:
    _LOUVAIN_AVAILABLE = False

from pyvis.network import Network

from app.core.config import get_settings
from app.core.logging import logger

settings = get_settings()

# ── Colour palette per node type ─────────────────────────────────────────────

NODE_COLORS: Dict[str, str] = {
    "paper":  "#6366f1",   # indigo
    "author": "#10b981",   # emerald
    "topic":  "#f59e0b",   # amber
    "area":   "#ef4444",   # red
}


# ── Graph Builder ─────────────────────────────────────────────────────────────

class KnowledgeGraphBuilder:
    """
    Builds the ResearchGraph knowledge graph from paper/author/topic data.
    Call build() with lists of dicts, then save().
    """

    def __init__(self) -> None:
        self.G: nx.MultiDiGraph = nx.MultiDiGraph()
        self._paper_graph: nx.Graph = nx.Graph()   # undirected similarity graph

    # ── Node adders ───────────────────────────────────────────

    def add_papers(self, papers: List[Dict]) -> None:
        """papers: list of {arxiv_id, title, abstract, categories, topic_id, embedding_id}"""
        for p in papers:
            nid = f"paper:{p['arxiv_id']}"
            self.G.add_node(
                nid,
                node_type="paper",
                label=p["title"][:60],
                arxiv_id=p["arxiv_id"],
                title=p["title"],
                primary_category=p.get("primary_category", ""),
                topic_id=p.get("topic_id"),
                color=NODE_COLORS["paper"],
                size=8,
            )
            self._paper_graph.add_node(nid, arxiv_id=p["arxiv_id"])

    def add_authors(self, paper_author_pairs: List[Tuple[str, str]]) -> None:
        """paper_author_pairs: [(arxiv_id, author_name), ...]"""
        co_author_tracker: Dict[str, List[str]] = {}

        for arxiv_id, author_name in paper_author_pairs:
            anid = f"author:{author_name.lower().replace(' ', '_')}"
            pnid = f"paper:{arxiv_id}"

            if not self.G.has_node(anid):
                self.G.add_node(
                    anid,
                    node_type="author",
                    label=author_name,
                    name=author_name,
                    color=NODE_COLORS["author"],
                    size=6,
                )

            self.G.add_edge(pnid, anid, edge_type="authored_by", weight=1.0)

            # Track for co-authorship
            if arxiv_id not in co_author_tracker:
                co_author_tracker[arxiv_id] = []
            co_author_tracker[arxiv_id].append(anid)

        # Add co-authorship edges between authors on same paper
        for arxiv_id, author_nodes in co_author_tracker.items():
            for i, a1 in enumerate(author_nodes):
                for a2 in author_nodes[i + 1:]:
                    if self.G.has_edge(a1, a2):
                        # Increment weight for repeated collaborations
                        for k, data in self.G[a1][a2].items():
                            data["weight"] = data.get("weight", 1) + 1
                    else:
                        self.G.add_edge(
                            a1, a2,
                            edge_type="co_authored",
                            weight=1.0,
                        )

    def add_topics(self, topics: List[Dict]) -> None:
        """topics: list of {id, label, top_words}"""
        for t in topics:
            tnid = f"topic:{t['id']}"
            self.G.add_node(
                tnid,
                node_type="topic",
                label=t["label"],
                top_words=t.get("top_words", []),
                color=NODE_COLORS["topic"],
                size=15,
            )

    def add_topic_memberships(self, paper_topic_pairs: List[Tuple[str, int]]) -> None:
        """paper_topic_pairs: [(arxiv_id, topic_id), ...]"""
        for arxiv_id, topic_id in paper_topic_pairs:
            pnid = f"paper:{arxiv_id}"
            tnid = f"topic:{topic_id}"
            if self.G.has_node(pnid) and self.G.has_node(tnid):
                self.G.add_edge(
                    pnid, tnid,
                    edge_type="belongs_to_topic",
                    weight=1.0,
                )

    def add_research_areas(self, area_map: Dict[str, List[str]]) -> None:
        """area_map: {area_code: [arxiv_id, ...]}"""
        for area_code, arxiv_ids in area_map.items():
            anid = f"area:{area_code}"
            if not self.G.has_node(anid):
                self.G.add_node(
                    anid,
                    node_type="area",
                    label=area_code,
                    color=NODE_COLORS["area"],
                    size=20,
                )
            for arxiv_id in arxiv_ids:
                pnid = f"paper:{arxiv_id}"
                if self.G.has_node(pnid):
                    self.G.add_edge(
                        pnid, anid,
                        edge_type="belongs_to_area",
                        weight=1.0,
                    )

    def add_similarity_edges(
        self, edges: List[Tuple[str, str, float]]
    ) -> None:
        """edges: [(arxiv_id_a, arxiv_id_b, cosine_sim), ...]"""
        for a, b, sim in edges:
            na, nb = f"paper:{a}", f"paper:{b}"
            if self.G.has_node(na) and self.G.has_node(nb):
                self.G.add_edge(na, nb, edge_type="similar_to", weight=sim)
                self._paper_graph.add_edge(na, nb, weight=sim)

    # ── Save / Load ───────────────────────────────────────────

    def save(self) -> None:
        graph_dir = settings.graph_dir
        graph_dir.mkdir(parents=True, exist_ok=True)

        kg_path = graph_dir / "full_kg.pkl"
        sim_path = graph_dir / "paper_similarity.pkl"

        with open(kg_path, "wb") as f:
            pickle.dump(self.G, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(sim_path, "wb") as f:
            pickle.dump(self._paper_graph, f, protocol=pickle.HIGHEST_PROTOCOL)

        logger.info(
            f"Knowledge graph saved: "
            f"nodes={self.G.number_of_nodes()}, "
            f"edges={self.G.number_of_edges()}"
        )


# ── Analytics ─────────────────────────────────────────────────────────────────

class GraphAnalytics:
    """
    Computes PageRank, community detection, betweenness, eigenvector centrality.
    Works on the full KG (MultiDiGraph → converted to DiGraph for analytics).
    """

    def __init__(self, G: nx.MultiDiGraph) -> None:
        # Convert to DiGraph (collapse multi-edges by summing weights)
        self.G = G
        self._simple: nx.DiGraph = nx.DiGraph()
        for u, v, data in G.edges(data=True):
            w = data.get("weight", 1.0)
            if self._simple.has_edge(u, v):
                self._simple[u][v]["weight"] += w
            else:
                self._simple.add_edge(u, v, weight=w)
        for n, data in G.nodes(data=True):
            self._simple.add_node(n, **data)

    def compute_pagerank(
        self, alpha: float = 0.85, max_iter: int = 100
    ) -> Dict[str, float]:
        pr = nx.pagerank(self._simple, alpha=alpha, max_iter=max_iter, weight="weight")
        return pr

    def compute_communities(self) -> Dict[str, int]:
        """Louvain community detection on undirected paper subgraph."""
        if not _LOUVAIN_AVAILABLE:
            logger.warning("python-louvain not available; skipping community detection")
            return {}
        undirected = self._simple.to_undirected()
        # Run on full graph
        try:
            partition = louvain_partition(undirected, weight="weight", random_state=42)
            return partition
        except Exception as e:
            logger.error(f"Louvain community detection failed: {e}")
            return {}

    def compute_betweenness(
        self, k: Optional[int] = 500
    ) -> Dict[str, float]:
        """
        Betweenness centrality. Uses approximate method (k samples)
        for large graphs to keep runtime manageable on CPU.
        """
        n = self._simple.number_of_nodes()
        sample_k = min(k, n) if k else None
        return nx.betweenness_centrality(
            self._simple, k=sample_k, weight="weight", normalized=True
        )

    def compute_degree_centrality(self) -> Dict[str, float]:
        return nx.degree_centrality(self._simple)

    def top_papers_by_pagerank(
        self, pagerank: Dict[str, float], n: int = 50
    ) -> List[Tuple[str, float]]:
        paper_pr = [
            (nid, score)
            for nid, score in pagerank.items()
            if nid.startswith("paper:")
        ]
        return sorted(paper_pr, key=lambda x: x[1], reverse=True)[:n]


# ── PyVis renderer ────────────────────────────────────────────────────────────

class GraphVisualizer:
    """Renders a NetworkX subgraph to an interactive PyVis HTML string."""

    @staticmethod
    def render_subgraph(
        G: nx.MultiDiGraph,
        center_node: str,
        radius: int = 2,
        max_nodes: int = 80,
        height: str = "600px",
        width: str = "100%",
    ) -> str:
        """
        Extract ego graph around center_node, then render with PyVis.
        Returns an HTML string to be served directly by the API.
        """
        if not G.has_node(center_node):
            return "<p>Node not found in graph.</p>"

        subG = nx.ego_graph(G, center_node, radius=radius, undirected=True)

        # Trim if too large
        if subG.number_of_nodes() > max_nodes:
            # Keep nodes closest to center first (BFS order)
            bfs_order = list(nx.bfs_tree(subG.to_undirected(), center_node).nodes())
            keep = set(bfs_order[:max_nodes])
            subG = subG.subgraph(keep).copy()

        net = Network(height=height, width=width, directed=True, notebook=False)
        net.set_options("""
        {
          "physics": {
            "enabled": true,
            "barnesHut": {
              "gravitationalConstant": -8000,
              "centralGravity": 0.3,
              "springLength": 95
            },
            "stabilization": {"iterations": 150}
          },
          "interaction": {"hover": true, "tooltipDelay": 200}
        }
        """)

        for node, data in subG.nodes(data=True):
            size = 20 if node == center_node else data.get("size", 8)
            net.add_node(
                node,
                label=data.get("label", node[:30]),
                color=data.get("color", "#6366f1"),
                size=size,
                title=_build_node_tooltip(data),
            )

        for u, v, data in subG.edges(data=True):
            net.add_edge(
                u, v,
                title=data.get("edge_type", "related"),
                width=max(1, data.get("weight", 1.0) * 3),
                color={"color": "#9ca3af", "opacity": 0.6},
            )

        return net.generate_html(notebook=False)


def _build_node_tooltip(data: Dict[str, Any]) -> str:
    ntype = data.get("node_type", "unknown")
    if ntype == "paper":
        return (
            f"<b>{data.get('title', '')[:80]}</b><br>"
            f"Category: {data.get('primary_category', '')}"
        )
    if ntype == "author":
        return f"<b>Author:</b> {data.get('name', '')}"
    if ntype == "topic":
        words = ", ".join(data.get("top_words", [])[:5])
        return f"<b>Topic:</b> {data.get('label', '')}<br>Keywords: {words}"
    if ntype == "area":
        return f"<b>Research Area:</b> {data.get('label', '')}"
    return str(data.get("label", ""))


# ── Knowledge Graph Service (singleton) ──────────────────────────────────────

class KnowledgeGraphService:
    _instance: Optional["KnowledgeGraphService"] = None

    def __init__(self) -> None:
        self._G: Optional[nx.MultiDiGraph] = None
        self._paper_sim_G: Optional[nx.Graph] = None
        self._pagerank: Dict[str, float] = {}
        self._communities: Dict[str, int] = {}
        self._betweenness: Dict[str, float] = {}
        self._loaded = False

    @classmethod
    def get_instance(cls) -> "KnowledgeGraphService":
        if cls._instance is None:
            cls._instance = KnowledgeGraphService()
        return cls._instance

    def load(self) -> bool:
        kg_path  = settings.graph_dir / "full_kg.pkl"
        sim_path = settings.graph_dir / "paper_similarity.pkl"
        pr_path  = settings.graph_dir / "pagerank.json"
        comm_path = settings.graph_dir / "communities.json"

        if not kg_path.exists():
            logger.warning("Knowledge graph not found. Run scripts/build_knowledge_graph.py")
            return False

        try:
            with open(kg_path, "rb") as f:
                self._G = pickle.load(f)

            if sim_path.exists():
                with open(sim_path, "rb") as f:
                    self._paper_sim_G = pickle.load(f)

            # Load pre-computed analytics if available
            if pr_path.exists():
                with open(pr_path) as f:
                    self._pagerank = json.load(f)
            else:
                logger.info("Computing PageRank (first load)…")
                analytics = GraphAnalytics(self._G)
                self._pagerank = analytics.compute_pagerank()
                with open(pr_path, "w") as f:
                    json.dump(self._pagerank, f)

            if comm_path.exists():
                with open(comm_path) as f:
                    self._communities = json.load(f)
            else:
                logger.info("Computing Louvain communities (first load)…")
                analytics = GraphAnalytics(self._G)
                self._communities = analytics.compute_communities()
                with open(comm_path, "w") as f:
                    json.dump(self._communities, f)

            self._loaded = True
            logger.info(
                f"KG loaded: nodes={self._G.number_of_nodes()}, "
                f"edges={self._G.number_of_edges()}, "
                f"pagerank_entries={len(self._pagerank)}"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to load knowledge graph: {e}")
            return False

    # ── Query API ─────────────────────────────────────────────

    def get_paper_neighbors(
        self, arxiv_id: str, radius: int = 1
    ) -> Set[str]:
        """Return set of arxiv_ids reachable within `radius` hops from paper node."""
        if not self._loaded:
            return set()
        nid = f"paper:{arxiv_id}"
        if not self._G.has_node(nid):
            return set()
        ego = nx.ego_graph(self._G, nid, radius=radius, undirected=True)
        return {
            n.replace("paper:", "")
            for n in ego.nodes()
            if n.startswith("paper:") and n != nid
        }

    def render_paper_subgraph(self, arxiv_id: str, radius: int = 2) -> str:
        if not self._loaded:
            return "<p>Knowledge graph not loaded.</p>"
        visualizer = GraphVisualizer()
        return visualizer.render_subgraph(
            self._G, center_node=f"paper:{arxiv_id}", radius=radius
        )

    def get_top_papers_by_pagerank(self, n: int = 20) -> List[Dict]:
        analytics = GraphAnalytics(self._G)
        top = analytics.top_papers_by_pagerank(self._pagerank, n=n)
        return [
            {
                "arxiv_id": nid.replace("paper:", ""),
                "pagerank": score,
                "community": self._communities.get(nid, -1),
                "betweenness": self._betweenness.get(nid, 0.0),
            }
            for nid, score in top
        ]

    def get_graph_stats(self) -> Dict[str, Any]:
        if not self._loaded:
            return {"loaded": False}
        return {
            "loaded": True,
            "num_nodes": self._G.number_of_nodes(),
            "num_edges": self._G.number_of_edges(),
            "num_communities": len(set(self._communities.values())) if self._communities else 0,
            "num_paper_nodes": sum(
                1 for _, d in self._G.nodes(data=True) if d.get("node_type") == "paper"
            ),
            "num_author_nodes": sum(
                1 for _, d in self._G.nodes(data=True) if d.get("node_type") == "author"
            ),
            "num_topic_nodes": sum(
                1 for _, d in self._G.nodes(data=True) if d.get("node_type") == "topic"
            ),
        }

    def get_nodes_and_edges_for_api(
        self, arxiv_id: str, radius: int = 2, max_nodes: int = 80
    ) -> Tuple[List[Dict], List[Dict]]:
        """Return nodes+edges as plain dicts for JSON API response."""
        if not self._loaded:
            return [], []

        nid = f"paper:{arxiv_id}"
        if not self._G.has_node(nid):
            return [], []

        subG = nx.ego_graph(self._G, nid, radius=radius, undirected=True)
        if subG.number_of_nodes() > max_nodes:
            bfs_order = list(nx.bfs_tree(subG.to_undirected(), nid).nodes())
            keep = set(bfs_order[:max_nodes])
            subG = subG.subgraph(keep).copy()

        nodes = []
        for node, data in subG.nodes(data=True):
            nodes.append({
                "id": node,
                "label": data.get("label", node[:40]),
                "node_type": data.get("node_type", "paper"),
                "size": 20.0 if node == nid else float(data.get("size", 8)),
                "color": data.get("color", NODE_COLORS["paper"]),
                "metadata": {
                    k: v for k, v in data.items()
                    if k not in {"label", "node_type", "size", "color"}
                },
            })

        edges = []
        for u, v, data in subG.edges(data=True):
            edges.append({
                "source": u,
                "target": v,
                "weight": float(data.get("weight", 1.0)),
                "edge_type": data.get("edge_type", "related"),
            })

        return nodes, edges

    @property
    def is_ready(self) -> bool:
        return self._loaded

    @property
    def pagerank_scores(self) -> Dict[str, float]:
        # Strip "paper:" prefix so callers can look up by arxiv_id
        return {
            k.replace("paper:", ""): v
            for k, v in self._pagerank.items()
            if k.startswith("paper:")
        }
