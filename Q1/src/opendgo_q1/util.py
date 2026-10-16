from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def safe_component(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return cleaned[:180] or "unnamed"


def extract_json(text: str) -> Any | None:
    stripped = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", stripped, flags=re.DOTALL | re.IGNORECASE)
    candidate = fenced.group(1).strip() if fenced else stripped
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass
    starts = [index for index in (candidate.find("{"), candidate.find("[")) if index >= 0]
    if not starts:
        return None
    start = min(starts)
    for end in range(len(candidate), start, -1):
        try:
            return json.loads(candidate[start:end])
        except json.JSONDecodeError:
            continue
    return None


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} must contain one JSON object")
            rows.append(value)
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def normalize_for_loop_detection(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def estimate_tokens(text: str) -> int:
    # Used only for message accounting when the provider does not expose a
    # tokenizer. Provider-reported model usage remains the authoritative total.
    return max(1, (len(text) + 3) // 4) if text else 0

