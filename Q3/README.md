# OpenDGO Q3: Structural Properties and Interaction Mechanisms

This directory contains experiment and statistical analysis code for Q3 and does not include plotting code. Q3 examines:

```text
graph property -> interaction behavior -> task quality
```

## Experimental Components

1. Generate a frozen graph library from all connected, non-isomorphic six-node undirected graphs using an outcome-blind rule;
2. Compute seven structural properties: density, diameter, clustering, degree centralization, normalized edge connectivity, cycle redundancy, and greedy modularity;
3. Select matched graph pairs that change the target property while changing other properties as little as possible;
4. Run graph variants using the frozen Q1 tasks across six domains, with the same roles, evidence, model, budgets, and 5 seeds;
5. Compute six interaction-behavior metrics from message logs;
6. Estimate standardized partial effects using domain fixed effects and matched-graph controls;
7. Compute asymmetric 95% intervals using a bootstrap grouped by task.

Prespecified properties:

- Sequential: diameter, edge connectivity, density;
- Parallel: modularity, clustering;
- Evidence Integration: cycle redundancy, edge connectivity.

## Code Structure

```text
src/opendgo_q3/
  graph_properties.py   # Seven properties and fixed topologies
  graph_library.py      # Outcome-blind graph library and matched-pair generation
  runner.py             # Run custom graphs using the Q1 execution protocol
  evaluation.py         # Reuse Q1 scoring with evaluator-only gold
  trace_metrics.py      # Six interaction-behavior metrics for Table 4
  analysis.py           # Partial effects and mechanism diagnostics
```

## Running

```bash
# Q3 reuses the Q1 task protocol; install both packages first
python -m pip install -e ../Q1_Organization_Benchmark
python -m pip install -e .

# Generate and freeze the outcome-blind graph library
opendgo-q3 graph-library --config configs/q3.yaml

# Run, evaluate, extract interaction metrics, and analyze
opendgo-q3 run --config configs/q3.yaml
opendgo-q3 evaluate --config configs/q3.yaml
opendgo-q3 metrics --config configs/q3.yaml
opendgo-q3 analyze --config configs/q3.yaml
```

`run` supports filtering by domain, structure, graph, seed, and task count. Run results contain no gold; trace annotations are read only during `metrics`.

## Table 4 Metric Boundaries

- Evidence coverage and propagation latency use evaluator-only gold-supporting evidence IDs;
- Duplicate-message rate uses a fixed token-set Jaccard threshold of `>= 0.90`;
- Contradiction resolution accepts only source-grounded conflicts and resolution regexes from audit annotations; missing values are recorded as NA;
- Cross-group flow is computed only when the graph library records a fixed community partition; otherwise, it is NA;
- Synthesizer load uses provider-reported final-synthesis prompt tokens when available; estimates are reconstructed from frozen prompt components only when the provider does not return this value.

