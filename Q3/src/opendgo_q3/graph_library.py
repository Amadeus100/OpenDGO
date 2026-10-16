from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np

from .config import Q3Config
from .graph_properties import PROPERTY_NAMES, fixed_records, graph_properties, greedy_partition
from .util import sha256_json, write_json, write_jsonl


def _graph6(graph: nx.Graph) -> str:
    return nx.to_graph6_bytes(graph, header=False).decode("ascii").strip()


def _record_key(row: dict[str, Any]) -> str:
    return str(row["graph6"] if "graph6" in row else row["graph_id"])


def _candidate_records() -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    fixed_graphs = [nx.Graph(record["edges"]) for record in fixed_records()]
    atlas_graphs = [
        graph.copy()
        for graph in nx.graph_atlas_g()
        if graph.number_of_nodes() == 6 and nx.is_connected(graph)
    ]
    for graph in sorted(atlas_graphs, key=_graph6):
        graph = nx.convert_node_labels_to_integers(graph, ordering="sorted")
        if any(nx.is_isomorphic(graph, classic) for classic in fixed_graphs):
            continue
        edges = sorted((min(left, right), max(left, right)) for left, right in graph.edges())
        partition, _ = greedy_partition(graph)
        candidates.append(
            {
                "graph6": _graph6(graph),
                "nodes": 6,
                "edges": [list(edge) for edge in edges],
                "properties": graph_properties(edges),
                "community_partition": partition if len(partition) > 1 else None,
            }
        )
    return candidates


def _z_maps(rows: list[dict[str, Any]]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    matrix = np.asarray(
        [[float(row["properties"][name]) for name in PROPERTY_NAMES] for row in rows],
        dtype=float,
    )
    mean = matrix.mean(axis=0)
    scale = matrix.std(axis=0)
    scale[scale == 0] = 1.0
    z = (matrix - mean) / scale
    return (
        {_record_key(row): z[index] for index, row in enumerate(rows)},
        {
            "property_order": list(PROPERTY_NAMES),
            "mean": mean.tolist(),
            "population_standard_deviation": scale.tolist(),
        },
    )


def _select_variants(candidates: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    all_rows = [*fixed_records(), *candidates]
    z_map, _ = _z_maps(all_rows)
    selected_vectors = [z_map[str(row["graph_id"])] for row in fixed_records()]
    remaining = list(candidates)
    selected: list[dict[str, Any]] = []
    while remaining and len(selected) < count:
        scored: list[tuple[float, str, dict[str, Any]]] = []
        for row in remaining:
            vector = z_map[str(row["graph6"])]
            distance = min(float(np.linalg.norm(vector - earlier)) for earlier in selected_vectors)
            scored.append((distance, str(row["graph6"]), row))
        _, _, winner = sorted(scored, key=lambda item: (-item[0], item[1]))[0]
        selected.append(winner)
        selected_vectors.append(z_map[str(winner["graph6"])])
        remaining.remove(winner)
    result: list[dict[str, Any]] = []
    for index, row in enumerate(selected, 1):
        record = {**row, "graph_id": f"variant_{index:02d}", "graph_category": "variant"}
        record["graph_sha256"] = sha256_json(record["edges"])
        result.append(record)
    return result


def _matched_pairs(records: list[dict[str, Any]], count: int) -> dict[str, list[dict[str, Any]]]:
    z_map, _ = _z_maps(records)
    result: dict[str, list[dict[str, Any]]] = {}
    for property_index, target in enumerate(PROPERTY_NAMES):
        candidates: list[tuple[float, float, str, str, dict[str, Any]]] = []
        for left_index, left in enumerate(records):
            for right in records[left_index + 1 :]:
                left_z = z_map[_record_key(left)]
                right_z = z_map[_record_key(right)]
                if left_z[property_index] >= right_z[property_index]:
                    high, low, high_z, low_z = left, right, left_z, right_z
                else:
                    high, low, high_z, low_z = right, left, right_z, left_z
                target_change = float(high_z[property_index] - low_z[property_index])
                mask = [index for index in range(len(PROPERTY_NAMES)) if index != property_index]
                collateral = float(np.mean(np.abs(high_z[mask] - low_z[mask])))
                row = {
                    "high_graph_id": high["graph_id"],
                    "low_graph_id": low["graph_id"],
                    "target_property": target,
                    "target_change_z": target_change,
                    "mean_other_property_change_z": collateral,
                }
                candidates.append(
                    (-target_change, collateral, str(high["graph_id"]), str(low["graph_id"]), row)
                )
        result[target] = [item[-1] for item in sorted(candidates)[:count]]
    return result


def build_graph_library(config: Q3Config) -> dict[str, Any]:
    candidates = _candidate_records()
    variants = _select_variants(candidates, int(config.experiment["variant_graph_count"]))
    records = [*fixed_records(), *variants]
    for record in records:
        record.setdefault("graph_sha256", sha256_json(record["edges"]))
    _, standardization = _z_maps(records)
    pairs = _matched_pairs(records, int(config.experiment["matched_pairs_per_property"]))
    root = config.graph_library_root
    write_jsonl(root / "graph_attribute_table.jsonl", records)
    write_json(root / "graph_attribute_table.json", records)
    write_json(root / "matched_pairs.json", pairs)
    manifest = {
        "protocol": "opendgo_q3_outcome_blind_graph_library_v1",
        "networkx_version": nx.__version__,
        "candidate_universe": "connected non-isomorphic six-node undirected graphs from graph_atlas_g",
        "selection": "deterministic max-min standardized property distance from six fixed graphs",
        "outcomes_read": False,
        "fixed_graph_count": 6,
        "variant_graph_count": len(variants),
        "record_count": len(records),
        "standardization": standardization,
        "records_sha256": sha256_json(records),
        "matched_pairs_sha256": sha256_json(pairs),
    }
    write_json(root / "graph_library_manifest.json", manifest)
    return {**manifest, "output_root": str(root)}


def load_graph_records(root: Path) -> dict[str, dict[str, Any]]:
    path = root / "graph_attribute_table.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    records = {str(row["graph_id"]): row for row in rows}
    if len(records) != len(rows):
        raise ValueError("duplicate graph_id in frozen graph library")
    return records
