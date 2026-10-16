from __future__ import annotations

import json
from typing import Any

from opendgo_q1.data import domain_gold_path, load_gold
from opendgo_q1.evaluation import evaluate_result

from .config import Q3Config
from .util import write_json


def evaluate_tree(config: Q3Config) -> dict[str, Any]:
    gold_cache: dict[str, dict[str, Any]] = {}
    evaluated = 0
    invalid = 0
    errors: list[str] = []
    for path in sorted((config.results_root / "runs").rglob("result.json")):
        try:
            result = json.loads(path.read_text(encoding="utf-8"))
            domain = str(result["domain"])
            if domain not in gold_cache:
                gold_cache[domain] = load_gold(
                    domain_gold_path(config.q1_data_root, domain), domain
                )
            gold = gold_cache[domain][str(result["task_id"])]
            evaluation = evaluate_result(result, gold.answer, gold.metadata)
            evaluation.update(
                {
                    "graph_id": result["graph_id"],
                    "graph_category": result["organization_category"],
                    "graph_properties": result["graph_properties"],
                    "run_key": result["run_key"],
                }
            )
            write_json(path.parent / "evaluation.json", evaluation)
            evaluated += 1
            invalid += int(not evaluation.get("valid_for_primary_metric", False))
        except Exception as exc:
            errors.append(f"{path}: {type(exc).__name__}: {exc}")
    summary = {
        "evaluated": evaluated,
        "invalid_for_primary_metric": invalid,
        "errors": errors,
        "ready": evaluated > 0 and invalid == 0 and not errors,
    }
    write_json(config.results_root / "evaluation_summary.json", summary)
    return summary
