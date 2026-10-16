#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 || $# -gt 6 ]]; then
  echo "usage: $0 DOMAIN STRUCTURE ORGANIZATION SEED [CONFIG] [LIMIT]" >&2
  exit 2
fi

DOMAIN="$1"
STRUCTURE="$2"
ORGANIZATION="$3"
SEED="$4"
CONFIG="${5:-configs/q1.yaml}"
LIMIT="${6:-}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

ARGS=(
  -m opendgo_q1.cli run
  --config "${CONFIG}"
  --domain "${DOMAIN}"
  --structure "${STRUCTURE}"
  --organization "${ORGANIZATION}"
  --seed "${SEED}"
)
if [[ -n "${LIMIT}" ]]; then
  ARGS+=(--limit "${LIMIT}")
fi

python "${ARGS[@]}"

