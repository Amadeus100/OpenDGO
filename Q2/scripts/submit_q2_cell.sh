#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 6 || $# -gt 7 ]]; then
  echo "usage: $0 DOMAIN STRUCTURE TOPOLOGY POSITION ALLOCATION SEED [CONFIG]" >&2
  exit 2
fi

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DOMAIN=$1
STRUCTURE=$2
TOPOLOGY=$3
POSITION=$4
ALLOCATION=$5
SEED=$6
CONFIG=${7:-configs/q2_tree.yaml}
POSITION_SLUG=$(printf '%s' "$POSITION" | tr '[:upper:] ' '[:lower:]_')
LOG_DIR="$ROOT/logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/${DOMAIN}__${STRUCTURE}__${TOPOLOGY}__${POSITION_SLUG}__${ALLOCATION}__s${SEED}.log"

cd "$ROOT"
nohup bash scripts/run_q2_cell.sh \
  "$DOMAIN" "$STRUCTURE" "$TOPOLOGY" "$POSITION" "$ALLOCATION" "$SEED" "$CONFIG" \
  >"$LOG" 2>&1 &
echo "pid=$! log=$LOG"

