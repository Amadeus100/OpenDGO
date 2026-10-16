# Q1 Server Runbook

## 1. Environment

```bash
cd /path/to/Q1_Organization_Benchmark
python -m pip install -e .
export OPENDGO_LLM_BASE_URL=http://127.0.0.1:8993/v1
```


## 2. Building Task Packages

```bash
opendgo-q1 prepare --config configs/q1.yaml --domain finance
opendgo-q1 prepare --config configs/q1.yaml --domain biomedical
opendgo-q1 prepare --config configs/q1.yaml --domain legal
opendgo-q1 prepare --config configs/q1.yaml --domain scientific_research
opendgo-q1 prepare --config configs/q1.yaml --domain recommendation
```

An external SWE-bench executor must prepare the repository snapshot, tool policy, and official tests for Coding tasks.

## 3. Small-Scale Runs

```bash
bash scripts/run_q1_cell.sh finance parallel_decomposition star 101 configs/smoke.yaml 2
python -m opendgo_q1.cli status --config configs/smoke.yaml
```

## 4. Submitting Individual Cells in the Background

```bash
bash scripts/submit_q1_cell.sh finance sequential_dependency chain 101
bash scripts/submit_q1_cell.sh finance parallel_decomposition community 101
bash scripts/submit_q1_cell.sh finance evidence_integration mesh 101
```

Each cell specifies `domain × structure × organization × seed`, with tasks executed in frozen task ID order. Run keys with an existing `result.json` are skipped, so the same command can be used to resume interrupted runs.

## 5. Evaluation and Main Table

```bash
opendgo-q1 evaluate --config configs/q1.yaml
opendgo-q1 aggregate --config configs/q1.yaml
```

Main outputs:

- `results/q1/aggregate/seed_summary.csv`: mean task score and completion rate for each seed.
- `results/q1/aggregate/q1_table_long.csv`: mean and sample SD for each domain × structure × organization combination.
- `results/q1/aggregate/q1_table_wide.csv`: wide-format data corresponding directly to Table 1 in the paper.
- `results/q1/aggregate/aggregation_report.json`: missing runs, Coding records without official evaluation, and incomplete cells.

Aggregation sets `ready=true` only when all expected runs are present and have valid primary metrics; partial results are not silently treated as a complete main table.

## 6. Coding Executor stdout Protocol

The program specified by `executors.coding.command` reads a JSON request from stdin and returns a JSON object on stdout:

```json
{
  "status": "completed",
  "patch": "unified diff",
  "parsed_answer": {"patch": "unified diff"},
  "budget": {"provider_usage": {"total_tokens": 1234}},
  "official_evaluation": {
    "harness": "SWE-bench official",
    "resolved": true
  },
  "metadata": {
    "repository_snapshot": "immutable-id",
    "tool_events_path": "path/to/audit.jsonl"
  }
}
```

A patch may be saved without `official_evaluation`, but that run cannot be included in the Q1 primary-metric table.

