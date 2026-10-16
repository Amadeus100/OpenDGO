# OpenDGO Q4: Quality–Cost–Robustness

This directory contains the experiment code for Q4 and does not include plotting code. It uses the frozen Q1 tasks across six domains, three task structures, six fixed organizations, and 5 seeds to produce:

- Quality–cost Pareto data with full token accounting;
- Quality-curve data under shared token caps of `[6000, 9000, 12000, 15000, 18000, 24000]`;
- Quality-retention data under mixed node/directed-message-channel fault pressures of `[0, 10, 20, 30, 40, 50]%`.

## Final Protocol

- Token cost = provider-reported input/output tokens for all agent/final calls + estimated tokens for inter-agent messages actually transmitted;
- Caps retain the same final reserve and suppress subsequent transmissions only at message boundaries; final synthesis uses the state already received;
- Each failure event selects an agent or an active directed message channel with probabilities of 0.5/0.5, using stable SHA-256 sampling frozen before execution;
- Failed agents produce no output; failed channels drop messages without retries;
- Fault pressure is calculated using the initial number of failure-prone components, `6 + 2|E|`, and both requested and realized values are saved in the results.

## Running

```bash
python -m pip install -e ../Q1_Organization_Benchmark
python -m pip install -e .

opendgo-q4 run-budget --config configs/q4.yaml
opendgo-q4 run-fault --config configs/q4.yaml
opendgo-q4 evaluate --config configs/q4.yaml
opendgo-q4 aggregate --config configs/q4.yaml
```

Main outputs are `pareto_frontier.csv`, `budget_curve.csv`, and `fault_retained_quality.csv`; no figures are generated.

The reference implementation used an earlier protocol that separately evaluated agent failure, communication drop, and critical-edge removal, covering only Finance, 3 seeds, and 2 structures. This organized version follows the final paper by using six domains, 5 seeds, 6 fixed organizations, and mixed fault pressure.
