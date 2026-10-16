# Q3 Graph Library

`opendgo-q3 graph-library` writes the following files under `generated/`:

- `graph_attribute_table.jsonl`: six fixed graphs and graph variants selected without observing outcomes;
- `matched_pairs.json`: matched high/low graph pairs for each structure and prespecified property;
- `graph_library_manifest.json`: generation algorithm, dependency versions, candidate count, and all hashes.

The graph library must be frozen before model execution. The generator uses only graph structure properties and does not read task scores or model outputs.

