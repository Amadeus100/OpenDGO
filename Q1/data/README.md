# Q1 Data Format

Public tasks and private evaluation records must be stored separately for each domain to prevent gold leakage.

## normalized source

Each line in `raw/normalized.jsonl` represents one source question:

```json
{
  "source_id": "dataset-original-id",
  "question": "model-visible question",
  "answer": "evaluator-only gold",
  "dataset": "FinQA",
  "dataset_version": "pinned revision",
  "split": "test",
  "source_document_id": "report-or-repository-id",
  "difficulty": "medium",
  "source_category": "numerical_reasoning",
  "shared_context": {},
  "tools": [],
  "answer_schema": {"type": "object", "required": ["answer"]},
  "evidence_units": [
    {
      "evidence_id": "e1",
      "text": "model-visible evidence",
      "source": "table:3/row:7",
      "kind": "table",
      "required": true,
      "stage": 1,
      "role_hint": 2,
      "truth_status": "supporting"
    }
  ],
  "structure_views": {
    "sequential_dependency": [["e6"], ["e1"], ["e2"], ["e3"], ["e4"], ["e5"]],
    "parallel_decomposition": [["e6"], ["e1"], ["e2"], ["e3"], ["e4"], ["e5"]],
    "evidence_integration": [["e6"], ["e1"], ["e2"], ["e3"], ["e4"], ["e5"]]
  },
  "gold_metadata": {}
}
```

`structure_views` is preferred: it explicitly specifies which evidence IDs each of the six nodes can initially access under each structure. If omitted, `prepare` assigns evidence deterministically using only `stage`, `role_hint`, `required`, and `truth_status`; it does not infer evidence semantics. Evidence Integration requires at least two required evidence units assigned to different roles; otherwise, construction fails.

## processed task

Each line in `processed/tasks.jsonl` contains `task_id`, `source_id`, `domain`, `task_structure`, `question`, six `roles`, six `evidence_views`, `shared_context`, `tools`, `answer_schema`, `provenance`, and `visible_input_sha256`.

## private gold

Each line in `private/gold.jsonl` contains only `task_id`, `domain`, `answer`, and metadata needed for evaluation. The runner does not read this file; only the `evaluate` subcommand reads it.

## Primary Metrics by Domain

- Coding: SWE-bench official `resolved`.
- Finance: numerical/text Exact Match.
- Biomedical: Exact answer accuracy, with F1 also recorded for list questions.
- Legal: Label accuracy.
- Scientific Research: Claim label accuracy, with evidence F1 also recorded.
- Recommendation: Hit@1, with Hit@5, NDCG@5, and target rank also recorded.

