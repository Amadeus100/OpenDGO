# Q2 Data Layout

Each domain uses the following directories:

```text
data/<domain>/
  annotations/interventions.jsonl
  processed/tasks.jsonl
  private/gold.jsonl
  manifests/<task_id>/<topology>/<position>/<allocation>.json
```

## interventions.jsonl

Each line corresponds to one frozen Q1 task:

```json
{
  "task_id": "finance/example-001",
  "expert_role_index": 3,
  "decisive_evidence_ids": ["table_12", "table_13"],
  "supporting_evidence_ids": ["table_12", "table_13", "text_4"],
  "conflicting_evidence_ids": ["text_9"],
  "annotation_source": "FinQA program and evidence indices",
  "reviewed_by": "human-or-rule-version"
}
```

Requirements:

- `expert_role_index` is the original index among the six Q1 roles;
- decisive must be nonempty and a subset of supporting;
- conflicting and supporting must not overlap;
- All IDs must exist in the Q1 model-visible evidence and retain their original sources;
- Conflicting evidence must come from traceable original context and must not be invented by a model;
- Annotations must not contain the gold answer.

`prepare` takes the union of the Q1 model-visible evidence, copies the evaluator-only gold, and freezes the Q2 input hashes. `validate` constructs allocation manifests for every condition and checks evidence conservation, unique assignment, expert-role position, and equivalent permutations for Ring/Mesh.

