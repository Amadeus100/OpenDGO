#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 || $# -gt 6 ]]; then
  echo "usage: $0 DOMAIN STRUCTURE ORGANIZATION SEED [CONFIG] [LIMIT]" >&2
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
LOG_ROOT="${PROJECT_ROOT}/results/logs"
mkdir -p "${LOG_ROOT}"

DOMAIN="$1"
STRUCTURE="$2"
ORGANIZATION="$3"
SEED="$4"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_PATH="${LOG_ROOT}/${DOMAIN}__${STRUCTURE}__${ORGANIZATION}__${SEED}__${STAMP}.log"

nohup bash "${SCRIPT_DIR}/run_q1_cell.sh" "$@" >"${LOG_PATH}" 2>&1 &
PID="$!"
echo "pid=${PID}"
echo "log=${LOG_PATH}"

