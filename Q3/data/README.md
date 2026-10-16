# Q3 evaluator-only trace annotations

Each domain may provide:

```text
data/<domain>/trace_annotations.jsonl
```

Each line has the following format:

```json
{
  "task_id": "domain/example",
  "gold_supporting_evidence_ids": ["e1", "e2"],
  "conflicts": [
    {
      "evidence_id": "e9",
      "resolution_regex": "(?i)(reject|contradict).*e9"
    }
  ],
  "annotation_source": "dataset evidence labels",
  "reviewed_by": "rule-or-human-version"
}
```

These labels are not included in model prompts or run results and are read only during trace-metric extraction. Use an empty list for tasks without applicable conflicts. When conflicts exist but no auditable resolution rule is available, contradiction resolution is recorded as NA rather than zero.

