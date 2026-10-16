from __future__ import annotations

from typing import Iterable

import networkx as nx
from opendgo_q1.topologies import FIXED_EDGES, normalize_edges


PROPERTY_NAMES = (
    "density",
    "diameter",
    "clustering",
    "degree_centralization",
    "edge_connectivity",
    "cycle_redundancy",
    "modularity",
)


def as_graph(edges: Iterable[Iterable[int]]) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(range(6))
    graph.add_edges_from(normalize_edges(edges))
    return graph


def freeman_degree_centralization(graph: nx.Graph) -> float:
    degrees = dict(graph.degree())
    maximum = max(degrees.values())
    return sum(maximum - value for value in degrees.values()) / ((6 - 1) * (6 - 2))


def greedy_partition(graph: nx.Graph) -> tuple[list[list[int]], float]:
    communities = list(nx.community.greedy_modularity_communities(graph))
    modularity = float(nx.community.modularity(graph, communities)) if communities else 0.0
    partition = [sorted(int(node) for node in group) for group in communities]
    partition.sort(key=lambda group: (group[0], len(group), group))
    return partition, max(0.0, modularity)


def graph_properties(edges: Iterable[Iterable[int]]) -> dict[str, float]:
    graph = as_graph(edges)
    if not nx.is_connected(graph):
        raise ValueError("Q3 only accepts connected six-node graphs")
    _, modularity = greedy_partition(graph)
    cycle_rank = graph.number_of_edges() - graph.number_of_nodes() + 1
    return {
        "density": float(nx.density(graph)),
        "diameter": float(nx.diameter(graph)),
        "clustering": float(nx.average_clustering(graph)),
        "degree_centralization": float(freeman_degree_centralization(graph)),
        "edge_connectivity": float(nx.edge_connectivity(graph) / (graph.number_of_nodes() - 1)),
        "cycle_redundancy": float(cycle_rank / 10),
        "modularity": float(modularity),
    }


def fixed_records() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for name, raw_edges in FIXED_EDGES.items():
        edges = normalize_edges(raw_edges)
        graph = as_graph(edges)
        partition, _ = greedy_partition(graph)
        records.append(
            {
                "graph_id": name,
                "graph_category": "fixed",
                "nodes": 6,
                "edges": [list(edge) for edge in edges],
                "properties": graph_properties(edges),
                "community_partition": partition if len(partition) > 1 else None,
            }
        )
    return records
