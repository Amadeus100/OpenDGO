from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Iterable

from .config import Q1Config
from .data import domain_gold_path, load_gold
from .domains import get_domain


_SPACE = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w.%+-]+", re.UNICODE)
_FINANCE_UNITS = re.compile(
    r"\b(?:million|millions|billion|billions|thousand|thousands|year|years|usd)\b",
    flags=re.IGNORECASE,
)


def normalize_text(value: Any) -> str:
    text = str(value if value is not None else "").strip().lower()
    text = _PUNCT.sub(" ", text)
    return _SPACE.sub(" ", text).strip()


def normalize_finance_number(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    text = str(value if value is not None else "").strip()
    parenthesized_negative = bool(re.fullmatch(r"\(\s*\$?\s*[\d,.]+\s*\)", text))
    text = text.replace(",", "").replace("$", "").replace("%", "")
    text = _FINANCE_UNITS.sub("", text).strip()
    if parenthesized_negative:
        text = "-" + text.strip("() ")
    if not re.fullmatch(r"-?\d+(?:\.\d+)?", text):
        return None
    return float(text)


def exact_match(predicted: Any, gold: Any) -> float:
    return float(normalize_text(predicted) == normalize_text(gold))


def finance_exact_match(predicted: Any, gold: Any) -> float:
    left = normalize_finance_number(predicted)
    right = normalize_finance_number(gold)
    if left is not None and right is not None:
        return float(abs(left - right) <= 1e-6 * max(1.0, abs(right)))
    return exact_match(predicted, gold)


def set_f1(predicted: Iterable[Any], gold: Iterable[Any]) -> float:
    predicted_set = {normalize_text(item) for item in predicted if normalize_text(item)}
    gold_set = {normalize_text(item) for item in gold if normalize_text(item)}
    if not predicted_set and not gold_set:
        return 1.0
    if not predicted_set or not gold_set:
        return 0.0
    overlap = len(predicted_set & gold_set)
    precision = overlap / len(predicted_set)
    recall = overlap / len(gold_set)
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def _prediction(result: dict[str, Any], key: str) -> Any:
    parsed = result.get("parsed_answer")
    if isinstance(parsed, dict):
        return parsed.get(key)
    return parsed


def _finance_prediction(result: dict[str, Any]) -> tuple[Any, bool]:
    predicted = _prediction(result, "answer")
    if predicted is not None:
        return predicted, False
    raw = str(result.get("raw_answer", ""))
    match = re.search(r"(?:FINAL\s+ANSWER|ANSWER)\s*[:：-]\s*([^\n\r]+)", raw, re.I)
    return (match.group(1).strip(), True) if match else (None, False)


def evaluate_result(result: dict[str, Any], gold_answer: Any, gold_metadata: dict[str, Any]) -> dict[str, Any]:
    domain = str(result["domain"])
    common = {
        "task_id": result["task_id"],
        "source_id": result.get("source_id", ""),
        "domain": domain,
        "task_structure": result["task_structure"],
        "organization": result["organization"],
        "organization_category": result.get("organization_category", ""),
        "seed": int(result["seed"]),
        "run_status": result.get("status", ""),
    }
    if result.get("status") != "completed":
        return {
            **common,
            "primary_metric": get_domain(domain).primary_metric,
            "primary_score": 0.0,
            "valid_for_primary_metric": True,
            "evaluation_status": "incomplete_run_scored_zero",
        }

    if domain == "coding":
        official = result.get("official_evaluation")
        if not isinstance(official, dict):
            return {
                **common,
                "primary_metric": "official_resolve_rate",
                "primary_score": None,
                "valid_for_primary_metric": False,
                "evaluation_status": "pending_official_swebench_harness",
            }
        harness = str(official.get("harness", "")).lower()
        resolved = official.get("resolved")
        valid = "swe-bench" in harness and "official" in harness and resolved is not None
        return {
            **common,
            "primary_metric": "official_resolve_rate",
            "primary_score": float(bool(resolved)) if valid else None,
            "valid_for_primary_metric": valid,
            "evaluation_status": "official_harness" if valid else "invalid_harness_record",
        }

    if domain == "finance":
        predicted, recovered = _finance_prediction(result)
        if predicted is None:
            score = 0.0
            status = "unresolved_parse_failure"
        else:
            score = finance_exact_match(predicted, gold_answer)
            status = "deterministic_parse_recovery" if recovered else "evaluated"
        return {
            **common,
            "primary_metric": "exact_match",
            "primary_score": score,
            "exact_match": score,
            "valid_for_primary_metric": True,
            "evaluation_status": status,
        }

    if result.get("parse_failed"):
        return {
            **common,
            "primary_metric": get_domain(domain).primary_metric,
            "primary_score": 0.0,
            "valid_for_primary_metric": True,
            "evaluation_status": "unresolved_parse_failure",
        }

    if domain == "biomedical":
        predicted = _prediction(result, "exact_answer")
        question_type = str(gold_metadata.get("question_type", "factoid")).lower()
        if question_type == "list":
            predicted_list = predicted if isinstance(predicted, list) else [predicted]
            gold_list = gold_answer if isinstance(gold_answer, list) else [gold_answer]
            exact = float(
                {normalize_text(value) for value in predicted_list}
                == {normalize_text(value) for value in gold_list}
            )
            answer_f1 = set_f1(predicted_list, gold_list)
        else:
            exact = exact_match(predicted, gold_answer)
            answer_f1 = exact
        return {
            **common,
            "primary_metric": "exact_answer_accuracy",
            "primary_score": exact,
            "exact_answer_accuracy": exact,
            "answer_f1": answer_f1,
            "valid_for_primary_metric": True,
            "evaluation_status": "evaluated",
        }

    if domain == "legal":
        score = exact_match(_prediction(result, "label"), gold_answer)
        return {
            **common,
            "primary_metric": "label_accuracy",
            "primary_score": score,
            "label_accuracy": score,
            "valid_for_primary_metric": True,
            "evaluation_status": "evaluated",
        }

    if domain == "scientific_research":
        label = exact_match(_prediction(result, "label"), gold_answer)
        predicted_evidence = _prediction(result, "evidence_sentence_ids") or []
        if not isinstance(predicted_evidence, list):
            predicted_evidence = [predicted_evidence]
        gold_evidence = gold_metadata.get("evidence_sentence_ids", [])
        return {
            **common,
            "primary_metric": "label_accuracy",
            "primary_score": label,
            "label_accuracy": label,
            "evidence_f1": set_f1(predicted_evidence, gold_evidence),
            "valid_for_primary_metric": True,
            "evaluation_status": "evaluated",
        }

    ranking = _prediction(result, "ranked_item_ids") or []
    if not isinstance(ranking, list):
        ranking = [ranking]
    normalized = [str(item) for item in ranking]
    target = str(gold_answer)
    rank = normalized.index(target) + 1 if target in normalized else None
    hit_at_1 = float(rank == 1)
    hit_at_5 = float(rank is not None and rank <= 5)
    ndcg_at_5 = 0.0 if rank is None or rank > 5 else 1.0 / math.log2(rank + 1)
    return {
        **common,
        "primary_metric": "hit_at_1",
        "primary_score": hit_at_1,
        "hit_at_1": hit_at_1,
        "hit_at_5": hit_at_5,
        "ndcg_at_5": ndcg_at_5,
        "target_rank": rank,
        "valid_for_primary_metric": True,
        "evaluation_status": "evaluated",
    }


def evaluate_tree(config: Q1Config) -> dict[str, Any]:
    gold_cache: dict[str, dict[str, Any]] = {}
    evaluated = 0
    invalid = 0
    errors: list[str] = []
    for result_path in sorted((config.results_root / "runs").rglob("result.json")):
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
            domain = str(result["domain"])
            if domain not in gold_cache:
                gold_cache[domain] = load_gold(domain_gold_path(config.data_root, domain), domain)
            gold = gold_cache[domain][str(result["task_id"])]
            evaluation = evaluate_result(result, gold.answer, gold.metadata)
            (result_path.parent / "evaluation.json").write_text(
                json.dumps(evaluation, ensure_ascii=False, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            evaluated += 1
            if not evaluation.get("valid_for_primary_metric", False):
                invalid += 1
        except Exception as exc:
            errors.append(f"{result_path}: {type(exc).__name__}: {exc}")
    summary = {
        "evaluated": evaluated,
        "invalid_for_primary_metric": invalid,
        "errors": errors,
        "ready": evaluated > 0 and invalid == 0 and not errors,
    }
    output = config.results_root / "evaluation_summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary

