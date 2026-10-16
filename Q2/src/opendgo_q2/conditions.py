from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from typing import Iterable

from .schema import N_AGENTS


FIXED_EDGES: dict[str, tuple[tuple[int, int], ...]] = {
    "star": ((0, 1), (0, 2), (0, 3), (0, 4), (0, 5)),
    "chain": ((0, 1), (1, 2), (2, 3), (3, 4), (4, 5)),
    "tree": ((0, 1), (0, 2), (1, 3), (1, 4), (2, 5)),
    "ring": ((0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)),
    "mesh": tuple((i, j) for i in range(N_AGENTS) for j in range(i + 1, N_AGENTS)),
    "community": ((0, 1), (0, 2), (1, 2), (3, 4), (3, 5), (4, 5), (2, 3)),
}


@dataclass(frozen=True)
class PositionCondition:
    topology: str
    label: str
    target_vertex: int
    equivalent_control: bool = False
    rotation: int = 0

    @property
    def slug(self) -> str:
        return self.label.lower().replace(" ", "_")


POSITIONS: dict[str, tuple[PositionCondition, ...]] = {
    "tree": (
        PositionCondition("tree", "Center", 0),
        PositionCondition("tree", "Bridge", 1),
        PositionCondition("tree", "Edge", 3),
    ),
    "star": (
        PositionCondition("star", "Hub", 0),
        PositionCondition("star", "Leaf", 1),
    ),
    "chain": (
        PositionCondition("chain", "Center", 2),
        PositionCondition("chain", "Relay", 1),
        PositionCondition("chain", "End", 0),
    ),
    "ring": tuple(
        PositionCondition("ring", f"Rotation {offset + 1}", offset, True, offset)
        for offset in range(3)
    ),
    "mesh": tuple(
        PositionCondition("mesh", f"Node set {offset + 1}", offset, True, offset)
        for offset in range(3)
    ),
    "community": (
        PositionCondition("community", "Core", 0),
        PositionCondition("community", "Bridge", 2),
        PositionCondition("community", "Peripheral", 4),
    ),
}


@dataclass(frozen=True)
class Assignment:
    condition: PositionCondition
    node_to_role_index: tuple[int, ...]
    expert_node: int
    synthesizer_node: int

    def to_dict(self) -> dict[str, object]:
        return {
            "condition": asdict(self.condition),
            "node_to_role_index": list(self.node_to_role_index),
            "expert_node": self.expert_node,
            "synthesizer_node": self.synthesizer_node,
        }


def adjacency(topology: str) -> dict[int, list[int]]:
    graph = {node: [] for node in range(N_AGENTS)}
    for left, right in FIXED_EDGES[topology]:
        graph[left].append(right)
        graph[right].append(left)
    return {node: sorted(neighbors) for node, neighbors in graph.items()}


def distances(topology: str, source: int) -> dict[int, int]:
    graph = adjacency(topology)
    result = {source: 0}
    queue = deque([source])
    while queue:
        node = queue.popleft()
        for neighbor in graph[node]:
            if neighbor not in result:
                result[neighbor] = result[node] + 1
                queue.append(neighbor)
    return result


def oriented_layers(topology: str, sink: int) -> tuple[list[list[int]], dict[int, list[int]]]:
    graph = adjacency(topology)
    depth = distances(topology, sink)
    layers = [
        sorted(node for node, value in depth.items() if value == level)
        for level in range(max(depth.values()), 0, -1)
    ]
    upstream = {
        node: sorted(neighbor for neighbor in graph[node] if depth[neighbor] > depth[node])
        for node in graph
    }
    return layers, upstream


def assignment_for(
    condition: PositionCondition,
    expert_role_index: int,
    base_synthesizer_node: int = 0,
) -> Assignment:
    if expert_role_index not in range(N_AGENTS):
        raise ValueError("expert role index must be in 0..5")
    node_to_role = list(range(N_AGENTS))
    if condition.equivalent_control:
        node_to_role[0], node_to_role[expert_role_index] = (
            node_to_role[expert_role_index],
            node_to_role[0],
        )
        rotated = [-1] * N_AGENTS
        for old_node, role_index in enumerate(node_to_role):
            rotated[(old_node + condition.rotation) % N_AGENTS] = role_index
        node_to_role = rotated
        expert_node = condition.rotation
        synthesizer = (base_synthesizer_node + condition.rotation) % N_AGENTS
    else:
        target = condition.target_vertex
        node_to_role[target], node_to_role[expert_role_index] = (
            node_to_role[expert_role_index],
            node_to_role[target],
        )
        expert_node = target
        synthesizer = base_synthesizer_node
    if sorted(node_to_role) != list(range(N_AGENTS)):
        raise AssertionError("role assignment is not a permutation")
    if node_to_role[expert_node] != expert_role_index:
        raise AssertionError("expert role was not placed at the target node")
    return Assignment(condition, tuple(node_to_role), expert_node, synthesizer)


def find_condition(topology: str, label_or_slug: str) -> PositionCondition:
    normalized = label_or_slug.strip().lower().replace(" ", "_")
    for condition in POSITIONS[topology]:
        if normalized in {condition.slug, condition.label.lower().replace(" ", "_")}:
            return condition
    raise KeyError(f"unknown position {label_or_slug!r} for {topology}")


def all_conditions(topologies: Iterable[str]) -> list[PositionCondition]:
    return [condition for topology in topologies for condition in POSITIONS[topology]]

