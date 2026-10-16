# Q3 Server Runbook

```bash
cd /path/to/Q1_Organization_Benchmark
python -m pip install -e .
cd /path/to/Q3_Structural_Mechanism_Analysis
python -m pip install -e .
```

First generate the graph library and save its manifest:

```bash
opendgo-q3 graph-library --config configs/q3.yaml
```

Run an individual cell:

```bash
bash scripts/run_q3_cell.sh finance sequential_dependency variant_01 101 configs/q3.yaml
```

Full workflow:

```bash
opendgo-q3 run --config configs/q3.yaml
opendgo-q3 evaluate --config configs/q3.yaml
opendgo-q3 metrics --config configs/q3.yaml
opendgo-q3 analyze --config configs/q3.yaml
```

Main outputs:

- `aggregate/graph_profiles.csv`
- `aggregate/q3_property_effects.csv`
- `aggregate/q3_mechanism_diagnostics.csv`
- `aggregate/analysis_report.json`

These files provide the data for Fig. 3/4 and the mechanism table; no figures are generated.

