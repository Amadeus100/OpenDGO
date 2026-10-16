from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Iterable

from .config import Q2Config
from .data import gold_path, load_gold
from .domains import get_domain
from .util import write_json


_SPACE = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w.%+-]+", re.UNICODE)
_FINANCE_UNITS = re.compile(
    r"\b(?:million|millions|billion|billions|thousand|thousands|year|years|usd)\b",
    re.I,
)


def normalize_text(value: Any) -> str:
    text = str(value if value is not None else "").strip().lower()
    return _SPACE.sub(" ", _PUNCT.sub(" ", text)).strip()


def normalize_finance_number(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    text = str(value if value is not None else "").strip()
    negative = bool(re.fullmatch(r"\(\s*\$?\s*[\d,.]+\s*\)", text))
    text = _FINANCE_UNITS.sub("", text.replace(",", "").replace("$", "").replace("%", "")).strip()
    if negative:
        text = "-" + text.strip("() ")
    return float(text) if re.fullmatch(r"-?\d+(?:\.\d+)?", text) else None


def exact_match(predicted: Any, gold: Any) -> float:
    return float(normalize_text(predicted) == normalize_text(gold))


def finance_exact_match(predicted: Any, gold: Any) -> float:
    left, right = normalize_finance_number(predicted), normalize_finance_number(gold)
    if left is not None and right is not None:
        return float(abs(left - right) <= 1e-6 * max(1.0, abs(right)))
    return exact_match(predicted, gold)


def set_f1(predicted: Iterable[Any], gold: Iterable[Any]) -> float:
    left = {normalize_text(x) for x in predicted if normalize_text(x)}
    right = {normalize_text(x) for x in gold if normalize_text(x)}
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    overlap = len(left & right)
    precision, recall = overlap / len(left), overlap / len(right)
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def _prediction(result: dict[str, Any], key: str) -> Any:
    parsed = result.get("parsed_answer")
    return parsed.get(key) if isinstance(parsed, dict) else parsed


def evaluate_result(
    result: dict[str, Any], gold_answer: Any, gold_metadata: dict[str, Any]
) -> dict[str, Any]:
    domain = str(result["domain"])
    common = {
        key: result.get(key)
        for key in (
            "task_id",
            "source_id",
            "domain",
            "task_structure",
            "topology",
            "position",
            "position_slug",
            "information_allocation",
            "expert_role_index",
            "expert_node",
            "synthesizer_node",
            "seed",
        )
    }
    common["run_status"] = result.get("status", "")
    metric = get_domain(domain).primary_metric
    if result.get("status") != "completed":
        return {
            **common,
            "primary_metric": metric,
            "primary_score": 0.0,
            "valid_for_primary_metric": True,
            "evaluation_status": "incomplete_run_scored_zero",
        }
    if domain == "coding":
        official = result.get("official_evaluation")
        valid = (
            isinstance(official, dict)
            and "swe-bench" in str(official.get("harness", "")).lower()
            and "official" in str(official.get("harness", "")).lower()
            and official.get("resolved") is not None
        )
        return {
            **common,
            "primary_metric": metric,
            "primary_score": float(bool(official.get("resolved"))) if valid else None,
            "valid_for_primary_metric": valid,
            "evaluation_status": "official_harness" if valid else "pending_official_harness",
        }
    if domain == "finance":
        predicted = _prediction(result, "answer")
        recovered = False
        if predicted is None:
            match = re.search(
                r"(?:FINAL\s+ANSWER|ANSWER)\s*[:：-]\s*([^\n\r]+)",
                str(result.get("raw_answer", "")),
                re.I,
            )
            predicted = match.group(1).strip() if match else None
            recovered = match is not None
        score = finance_exact_match(predicted, gold_answer) if predicted is not None else 0.0
        return {
            **common,
            "primary_metric": metric,
            "primary_score": score,
            "exact_match": score,
            "valid_for_primary_metric": True,
            "evaluation_status": "deterministic_parse_recovery" if recovered else "evaluated",
        }
    if result.get("parse_failed"):
        return {
            **common,
            "primary_metric": metric,
            "primary_score": 0.0,
            "valid_for_primary_metric": True,
            "evaluation_status": "unresolved_parse_failure",
        }
    if domain == "biomedical":
        predicted = _prediction(result, "exact_answer")
        if str(gold_metadata.get("question_type", "factoid")).lower() == "list":
            left = predicted if isinstance(predicted, list) else [predicted]
            right = gold_answer if isinstance(gold_answer, list) else [gold_answer]
            score = float({normalize_text(x) for x in left} == {normalize_text(x) for x in right})
            f1 = set_f1(left, right)
        else:
            score = exact_match(predicted, gold_answer)
            f1 = score
        return {
            **common,
            "primary_metric": metric,
            "primary_score": score,
            "answer_f1": f1,
            "valid_for_primary_metric": True,
            "evaluation_status": "evaluated",
        }
    if domain == "legal":
        score = exact_match(_prediction(result, "label"), gold_answer)
        return {**common, "primary_metric": metric, "primary_score": score, "valid_for_primary_metric": True, "evaluation_status": "evaluated"}
    if domain == "scientific_research":
        score = exact_match(_prediction(result, "label"), gold_answer)
        evidence = _prediction(result, "evidence_sentence_ids") or []
        evidence = evidence if isinstance(evidence, list) else [evidence]
        return {
            **common,
            "primary_metric": metric,
            "primary_score": score,
            "evidence_f1": set_f1(evidence, gold_metadata.get("evidence_sentence_ids", [])),
            "valid_for_primary_metric": True,
            "evaluation_status": "evaluated",
        }
    ranking = _prediction(result, "ranked_item_ids") or []
    ranking = ranking if isinstance(ranking, list) else [ranking]
    ranking = [str(item) for item in ranking]
    target = str(gold_answer)
    rank = ranking.index(target) + 1 if target in ranking else None
    score = float(rank == 1)
    return {
        **common,
        "primary_metric": metric,
        "primary_score": score,
        "hit_at_5": float(rank is not None and rank <= 5),
        "ndcg_at_5": 0.0 if rank is None or rank > 5 else 1.0 / math.log2(rank + 1),
        "target_rank": rank,
        "valid_for_primary_metric": True,
        "evaluation_status": "evaluated",
    }


def evaluate_tree(config: Q2Config) -> dict[str, Any]:
    gold_cache: dict[str, dict[str, Any]] = {}
    evaluated = invalid = 0
    errors: list[str] = []
    for result_path in sorted((config.results_root / "runs").rglob("result.json")):
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
            domain = str(result["domain"])
            if domain not in gold_cache:
                gold_cache[domain] = load_gold(gold_path(config.data_root, domain), domain)
            gold = gold_cache[domain][str(result["task_id"])]
            evaluation = evaluate_result(result, gold.answer, gold.metadata)
            write_json(result_path.parent / "evaluation.json", evaluation)
            evaluated += 1
            invalid += int(not evaluation.get("valid_for_primary_metric", False))
        except Exception as exc:
            errors.append(f"{result_path}: {type(exc).__name__}: {exc}")
    summary = {
        "evaluated": evaluated,
        "invalid_for_primary_metric": invalid,
        "errors": errors,
        "ready": evaluated > 0 and invalid == 0 and not errors,
    }
    write_json(config.results_root / "evaluation_summary.json", summary)
    return summary

