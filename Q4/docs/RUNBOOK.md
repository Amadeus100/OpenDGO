# Q4 Server Runbook

```bash
cd /path/to/Q1_Organization_Benchmark && python -m pip install -e .
cd /path/to/Q4_Quality_Cost_Robustness && python -m pip install -e .
```

Run individual cells:

```bash
bash scripts/run_q4_budget_cell.sh finance sequential_dependency star 101 12000 configs/q4.yaml
bash scripts/run_q4_fault_cell.sh finance sequential_dependency star 101 20 configs/q4.yaml
```

After completing all runs:

```bash
opendgo-q4 evaluate --config configs/q4.yaml
opendgo-q4 aggregate --config configs/q4.yaml
opendgo-q4 status --config configs/q4.yaml
```

`run` supports result-level resume: runs with an existing `result.json` are not repeated. Fault plans are frozen before model calls and written to the results for review.
