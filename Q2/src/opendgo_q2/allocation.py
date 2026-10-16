from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .conditions import Assignment, distances
from .schema import ALLOCATIONS, EvidenceUnit, InterventionAnnotation, N_AGENTS, Q2Task
from .util import sha256_json, sha256_text


@dataclass(frozen=True)
class AllocationResult:
    views: tuple[tuple[EvidenceUnit, ...], ...]
    manifest: dict[str, Any]


def _round_robin(
    unit_ids: list[str],
    candidates: list[int],
    destination: dict[str, int],
) -> None:
    if unit_ids and not candidates:
        raise ValueError("no candidate node available for evidence allocation")
    for index, evidence_id in enumerate(sorted(unit_ids)):
        destination[evidence_id] = candidates[index % len(candidates)]


def _conflict_holder(topology: str, expert_node: int, synthesizer_node: int) -> int:
    depth = distances(topology, expert_node)
    candidates = [node for node in range(N_AGENTS) if node != expert_node]
    non_synth = [node for node in candidates if node != synthesizer_node]
    pool = non_synth or candidates
    return sorted(pool, key=lambda node: (-depth[node], node))[0]


def allocate(
    task: Q2Task,
    annotation: InterventionAnnotation,
    assignment: Assignment,
    mode: str,
) -> AllocationResult:
    if mode not in ALLOCATIONS:
        raise KeyError(mode)
    annotation.validate_for(task)
    units = {unit.evidence_id: unit for unit in task.evidence_units}
    all_ids = set(units)
    decisive = set(annotation.decisive_evidence_ids)
    supporting = set(annotation.supporting_evidence_ids)
    conflicting = set(annotation.conflicting_evidence_ids)
    other = all_ids - supporting - conflicting
    destination: dict[str, int] = {}
    expert = assignment.expert_node
    # Ring/Mesh controls must rotate the complete organization, including every
    # non-decisive packet. Build the canonical node-0 allocation first and apply
    # the same permutation used for roles and the synthesizer afterwards.
    working_expert = 0 if assignment.condition.equivalent_control else expert
    working_synthesizer = (
        0 if assignment.condition.equivalent_control else assignment.synthesizer_node
    )

    if mode == "centralized":
        destination = {evidence_id: working_expert for evidence_id in all_ids}
        conflict_node = working_expert
    elif mode == "distributed":
        for evidence_id in decisive:
            destination[evidence_id] = working_expert
        candidates = [node for node in range(N_AGENTS) if node != working_expert]
        _round_robin(sorted(all_ids - decisive), candidates, destination)
        conflict_node = None
    else:
        for evidence_id in decisive:
            destination[evidence_id] = working_expert
        conflict_node = _conflict_holder(
            assignment.condition.topology, working_expert, working_synthesizer
        )
        for evidence_id in conflicting:
            destination[evidence_id] = conflict_node
        candidates = [
            node for node in range(N_AGENTS) if node not in {working_expert, conflict_node}
        ]
        _round_robin(sorted((supporting - decisive) | other), candidates, destination)

    if assignment.condition.equivalent_control:
        rotation = assignment.condition.rotation
        destination = {
            evidence_id: (node + rotation) % N_AGENTS
            for evidence_id, node in destination.items()
        }
        if conflict_node is not None:
            conflict_node = (conflict_node + rotation) % N_AGENTS

    if set(destination) != all_ids:
        missing = all_ids - set(destination)
        extra = set(destination) - all_ids
        raise AssertionError(f"evidence conservation failed: missing={missing}, extra={extra}")
    views: list[list[EvidenceUnit]] = [[] for _ in range(N_AGENTS)]
    records: list[dict[str, Any]] = []
    for evidence_id in sorted(all_ids):
        unit = units[evidence_id]
        node = destination[evidence_id]
        views[node].append(unit)
        truth_role = (
            "decisive"
            if evidence_id in decisive
            else "supporting"
            if evidence_id in supporting
            else "conflicting"
            if evidence_id in conflicting
            else "other"
        )
        records.append(
            {
                "evidence_id": evidence_id,
                "source": unit.source,
                "content_sha256": sha256_text(unit.text),
                "initial_node": node,
                "construction_role": truth_role,
            }
        )

    inventory = [units[evidence_id].to_dict() for evidence_id in sorted(all_ids)]
    manifest = {
        "protocol_version": "opendgo_q2_allocation_v1",
        "task_id": task.task_id,
        "domain": task.domain,
        "task_structure": task.task_structure,
        "topology": assignment.condition.topology,
        "position": assignment.condition.label,
        "information_allocation": mode,
        "expert_role_index": annotation.expert_role_index,
        "expert_node": expert,
        "synthesizer_node": assignment.synthesizer_node,
        "node_to_role_index": list(assignment.node_to_role_index),
        "conflict_holder": conflict_node,
        "evidence_inventory_sha256": sha256_json(inventory),
        "evidence_count": len(all_ids),
        "assignments": records,
        "annotation_provenance": {
            "annotation_source": annotation.annotation_source,
            "reviewed_by": annotation.reviewed_by,
        },
    }
    manifest["manifest_sha256"] = sha256_json(manifest)
    return AllocationResult(tuple(tuple(view) for view in views), manifest)


def validate_mode_family(results: dict[str, AllocationResult]) -> None:
    if set(results) != set(ALLOCATIONS):
        raise ValueError("all three allocation modes are required")
    inventories = {
        result.manifest["evidence_inventory_sha256"] for result in results.values()
    }
    if len(inventories) != 1:
        raise AssertionError("allocation modes changed the evidence inventory")
    counts = {result.manifest["evidence_count"] for result in results.values()}
    if len(counts) != 1:
        raise AssertionError("allocation modes changed the evidence count")

