# OpenDGO Q2: Role Position and Information Allocation

This directory contains only the experiment code for Q2 and does not include plotting code. Q2 holds the six agents, role set, total evidence, tasks, model, tools, budget, and seeds constant, and changes only:

1. The position of the designated expert in the graph;
2. The initial allocation of the same evidence inventory.

The main experiment uses a `3 × 3` design on Tree:

- Expert position: `Center / Bridge / Edge`;
- Information allocation: `Centralized / Distributed / Conflicting`.

The appendix extends the experiment to Star, Chain, Ring, Mesh, and Community. The three horizontal-axis points for Ring and Mesh are equivalent permutation controls of the complete organization configuration and are not interpreted as centrality differences.

## Directory Structure

```text
Q2_Role_Position_and_Information_Allocation/
  configs/                    # Main Tree experiment, six-topology appendix, and small-scale configurations
  data/                       # Q2 audit annotations, task packages, private gold, and allocation manifests
  docs/                       # Runbook and provenance audit
  scripts/                    # Individual-cell execution and server submission scripts
  src/opendgo_q2/
    conditions.py             # Six topologies, 17 position conditions, and role permutations
    allocation.py             # Three information allocations and invariant checks
    prepare.py                # Build task packages from frozen Q1 tasks and manual Q2 annotations
    runner.py                 # Unified execution, budgets, logging, watchdog, and resume support
    evaluation.py             # Scoring with evaluator-only gold
    aggregate.py              # Data and interaction contrasts for Fig. 2/8/9
```

## Data Boundaries

Each domain requires:

```text
data/<domain>/
  annotations/interventions.jsonl  # Expert-role and evidence labels, hidden from the model
  processed/tasks.jsonl            # Frozen Q2 tasks without answers
  private/gold.jsonl               # Read only by the evaluator
  manifests/                       # Evidence destinations and hashes for each allocation
```

`interventions.jsonl` must explicitly specify `expert_role_index`, `decisive_evidence_ids`, `supporting_evidence_ids`, and `conflicting_evidence_ids`. The code does not automatically invent decisive evidence or conflicting facts. All three allocations use exactly the same set of evidence IDs and change only the initial holder nodes.

## Running

```bash
python -m pip install -e .

# First build Q2 data from frozen Q1 tasks and manual audit annotations
opendgo-q2 prepare --config configs/q2_tree.yaml --domain finance

# Check role permutations, evidence conservation, gold isolation, and experiment size
opendgo-q2 validate --config configs/q2_tree.yaml

# Main Tree experiment
opendgo-q2 run --config configs/q2_tree.yaml
opendgo-q2 evaluate --config configs/q2_tree.yaml
opendgo-q2 aggregate --config configs/q2_tree.yaml

# Six-topology appendix experiment
opendgo-q2 run --config configs/q2_appendix.yaml
```

Run filters include `--domain`, `--structure`, `--topology`, `--position`, `--allocation`, `--seed`, and `--limit`. Run keys with an existing `result.json` are skipped.

## Key Implementation Constraints

- Model prompts do not contain Q2, topology names, position names, or allocation condition names; the model sees only its role, task, private evidence, and messages actually delivered by neighbors.
- Expert-position experiments permute the roles themselves, rather than merely moving evidence.
- Tree, Star, Chain, and Community use compensating swaps to preserve the role multiset; Ring and Mesh jointly permute roles, evidence allocations, and the final synthesizer node.
- The Peripheral position in Community uses vertex 4, which is far from the fixed synthesizer node. The server prototype used vertex 3, but vertex 3 is an endpoint of the inter-community bridge and therefore does not meet the structural definition of Peripheral.
- Gold is not written to tasks, prompts, event logs, or run results; it is read only during `evaluate`.
- Coding must run through an external repository-aware executor, with the primary metric provided by the SWE-bench official harness.


