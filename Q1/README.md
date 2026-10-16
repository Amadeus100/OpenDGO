# OpenDGO Q1: Six-Domain Organization Benchmark

This directory contains the experiment code for Q1 in the paper and does not include plotting code. Q1 corresponds to Table 1: it compares Single Agent, six fixed graphs, and three learned organizations across six domains and three task structures.

## 1. Experimental Matrix

- Domains: Coding / SWE-bench Verified, Finance / FinQA, Biomedical / BioASQ Task B, Legal / LegalBench, Scientific Research / SciFact, and Recommendation / Amazon Reviews 2023.
- Task structures: `sequential_dependency`, `parallel_decomposition`, and `evidence_integration`.
- Organizations: `single_agent`, Star, Chain, Tree, Ring, Mesh, Community, G-Designer, ARG-Designer, and GoAgent.
- Full configuration: 500 frozen tasks per domain × structure combination and 5 seeds; the full matrix consists of `6 × 3 × 10 × 500 × 5 = 450,000` runs.

Tasks, seeds, models, tools, role sets, the evidence union, nominal budgets, stopping rules, and evaluators are held constant across methods. Single Agent receives the union of the six model-visible evidence views.

## 2. Code Structure

```text
Q1_Organization_Benchmark/
  configs/                 # Full and smoke configurations
  data/                    # Layout of public task packages and private gold
  learned_graphs/          # Interface for frozen graph artifacts from learned methods
  src/opendgo_q1/
    domains/               # Roles, output protocols, and scoring rules for six domains
    backend.py             # OpenAI-compatible model service
    prepare.py             # Build three task packages from annotated normalized JSONL
    organizations.py       # Resolve Single/fixed/learned organizations
    topologies.py          # Exact six-node edge sets and Sequential DAG
    runner.py              # Unified execution, budgets, resume support, logging, and watchdog
    evaluation.py          # Scoring with evaluator-only gold
    aggregate.py           # Main-table data: seed-level mean ± sample SD
  tests/                   # Protocol tests without real model calls
```

## 3. Data Boundaries

Each domain uses the same layout:

```text
data/<domain>/
  raw/normalized.jsonl        # Frozen source questions and manually/rule-annotated evidence units
  processed/tasks.jsonl       # Model-visible tasks without answers
  private/gold.jsonl          # Read only by the evaluator
```

`tasks.jsonl` stores the source dataset, version, split, source ID, six roles, six private evidence views, tool permissions, answer schema, and a hash of the model-visible input. Gold answers, gold evidence labels, and official test results are never included in prompts.

Source datasets use different formats, so this project does not guess fields when the original data is unavailable. `prepare` accepts only explicitly annotated normalized JSONL; it does not select questions using model outputs or experimental scores.

## 4. Running

```bash
python -m venv .venv
python -m pip install -e .[dev]

# Check data, gold isolation, cell counts, and learned graph artifacts
opendgo-q1 validate --config configs/q1.yaml

# Start with a small-scale Finance engineering check
opendgo-q1 run --config configs/smoke.yaml
opendgo-q1 evaluate --config configs/smoke.yaml
opendgo-q1 aggregate --config configs/smoke.yaml

# Resume full runs; run keys with an existing result.json do not call the model again
opendgo-q1 run --config configs/q1.yaml
opendgo-q1 evaluate --config configs/q1.yaml
opendgo-q1 aggregate --config configs/q1.yaml
```

`run` supports filtering by `--domain`, `--structure`, `--organization`, `--seed`, and `--limit` for submission across GPUs or machines. Each run key is uniquely determined by the task, structure, organization, and seed.
