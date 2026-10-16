from __future__ import annotations

from collections import deque
from typing import Iterable

from .schema import N_AGENTS


FIXED_EDGES: dict[str, tuple[tuple[int, int], ...]] = {
    "star": ((0, 1), (0, 2), (0, 3), (0, 4), (0, 5)),
    "chain": ((0, 1), (1, 2), (2, 3), (3, 4), (4, 5)),
    "tree": ((0, 1), (0, 2), (1, 3), (1, 4), (2, 5)),
    "ring": ((0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)),
    "mesh": tuple(
        (left, right) for left in range(N_AGENTS) for right in range(left + 1, N_AGENTS)
    ),
    "community": ((0, 1), (0, 2), (1, 2), (3, 4), (3, 5), (4, 5), (2, 3)),
}


def normalize_edges(edges: Iterable[Iterable[int]]) -> tuple[tuple[int, int], ...]:
    normalized: list[tuple[int, int]] = []
    for raw in edges:
        pair = tuple(int(node) for node in raw)
        if len(pair) != 2:
            raise ValueError(f"edge must contain two nodes: {pair}")
        left, right = pair
        if left == right:
            raise ValueError(f"self-loop is not allowed: {(left, right)}")
        if left not in range(N_AGENTS) or right not in range(N_AGENTS):
            raise ValueError(f"edge node must be in 0..{N_AGENTS - 1}: {(left, right)}")
        normalized.append((min(left, right), max(left, right)))
    if len(normalized) != len(set(normalized)):
        raise ValueError("duplicate undirected edge")
    result = tuple(sorted(normalized))
    if not result:
        raise ValueError("organization graph must contain at least one edge")
    if len(reachable_nodes(result, 0)) != N_AGENTS:
        raise ValueError("organization graph must connect all six nodes")
    return result


def adjacency(edges: Iterable[Iterable[int]]) -> dict[int, list[int]]:
    graph = {node: [] for node in range(N_AGENTS)}
    for left, right in normalize_edges(edges):
        graph[left].append(right)
        graph[right].append(left)
    for values in graph.values():
        values.sort()
    return graph


def reachable_nodes(edges: Iterable[Iterable[int]], source: int) -> set[int]:
    raw_edges = [tuple(int(node) for node in edge) for edge in edges]
    graph = {node: [] for node in range(N_AGENTS)}
    for left, right in raw_edges:
        graph[left].append(right)
        graph[right].append(left)
    seen = {source}
    queue = deque([source])
    while queue:
        node = queue.popleft()
        for neighbor in graph[node]:
            if neighbor not in seen:
                seen.add(neighbor)
                queue.append(neighbor)
    return seen


def distances_to_sink(edges: Iterable[Iterable[int]], sink: int = 0) -> dict[int, int]:
    graph = adjacency(edges)
    distance = {sink: 0}
    queue = deque([sink])
    while queue:
        node = queue.popleft()
        for neighbor in graph[node]:
            if neighbor not in distance:
                distance[neighbor] = distance[node] + 1
                queue.append(neighbor)
    return distance


def sequential_layers(edges: Iterable[Iterable[int]], sink: int = 0) -> list[list[int]]:
    distance = distances_to_sink(edges, sink)
    return [
        sorted(node for node, depth in distance.items() if depth == level)
        for level in range(max(distance.values()), 0, -1)
    ]


def sequential_inputs(edges: Iterable[Iterable[int]], sink: int = 0) -> dict[int, list[int]]:
    graph = adjacency(edges)
    distance = distances_to_sink(edges, sink)
    return {
        node: sorted(neighbor for neighbor in graph[node] if distance[neighbor] > distance[node])
        for node in graph
    }


def diameter(edges: Iterable[Iterable[int]]) -> int:
    graph = adjacency(edges)
    longest = 0
    for source in graph:
        distances = {source: 0}
        queue = deque([source])
        while queue:
            node = queue.popleft()
            for neighbor in graph[node]:
                if neighbor not in distances:
                    distances[neighbor] = distances[node] + 1
                    queue.append(neighbor)
        longest = max(longest, max(distances.values()))
    return longest


def topology_record(name: str, edges: Iterable[Iterable[int]]) -> dict[str, object]:
    normalized = normalize_edges(edges)
    return {
        "name": name,
        "nodes": N_AGENTS,
        "edges": [list(edge) for edge in normalized],
        "edge_count": len(normalized),
        "diameter": diameter(normalized),
        "adjacency": adjacency(normalized),
    }

