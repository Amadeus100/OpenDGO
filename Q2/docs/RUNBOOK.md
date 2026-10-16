# Q2 Server Runbook

## 1. Installation

```bash
cd /path/to/Q2_Role_Position_and_Information_Allocation
python -m pip install -e .
```

Place frozen Q1 tasks under `paths.q1_data_root` and manual/rule-based Q2 audit annotations under `data/<domain>/annotations/interventions.jsonl`.

## 2. Building and Validating Data

```bash
for domain in coding finance biomedical legal scientific_research recommendation; do
  opendgo-q2 prepare --config configs/q2_tree.yaml --domain "$domain"
done
opendgo-q2 validate --config configs/q2_tree.yaml
```

Do not start full experiments until validation passes. In particular, manually inspect samples from `data/<domain>/manifests/` to confirm that decisive evidence is initially held only by the designated nodes and that conflicting evidence comes from actual original sources.

## 3. Running an Individual Cell

```bash
bash scripts/run_q2_cell.sh \
  finance sequential_dependency tree Center centralized 101 \
  configs/q2_tree.yaml
```

Each cell is uniquely identified by `domain × structure × topology × position × allocation × seed`. Tasks run in frozen task ID order, with support for resuming interrupted runs.

## 4. Main Experiment and Appendix

```bash
# Main Tree 3×3 experiment
opendgo-q2 run --config configs/q2_tree.yaml

# Six-topology extension for Fig. 8/9
opendgo-q2 run --config configs/q2_appendix.yaml
```

## 5. Scoring and Aggregation

```bash
opendgo-q2 evaluate --config configs/q2_tree.yaml
opendgo-q2 aggregate --config configs/q2_tree.yaml
```

Aggregated outputs include the condition-level long table, Tree main effects, the best pairing for each domain, Tree interaction contrasts, and appendix curve data. Only data is produced; no figures are generated.

## 6. Coding executor

Coding uses an external repository-aware executor. The stdin request contains the frozen task, role permutation, evidence allocation, graph edges, final synthesizer node, and seed. stdout must return a JSON object containing:

```json
{
  "status": "completed",
  "patch": "unified diff",
  "parsed_answer": {"patch": "unified diff"},
  "official_evaluation": {
    "harness": "SWE-bench official",
    "resolved": true
  },
  "budget": {"provider_usage": {"total_tokens": 1234}}
}
```

Coding runs without official evaluation records are excluded from the primary metric.

