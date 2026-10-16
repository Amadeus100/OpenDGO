#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 6 || $# -gt 8 ]]; then
  echo "usage: $0 DOMAIN STRUCTURE TOPOLOGY POSITION ALLOCATION SEED [CONFIG] [LIMIT]" >&2
  exit 2
fi

DOMAIN=$1
STRUCTURE=$2
TOPOLOGY=$3
POSITION=$4
ALLOCATION=$5
SEED=$6
CONFIG=${7:-configs/q2_tree.yaml}
LIMIT=${8:-}

ARGS=(
  -m opendgo_q2 run
  --config "$CONFIG"
  --domain "$DOMAIN"
  --structure "$STRUCTURE"
  --topology "$TOPOLOGY"
  --position "$POSITION"
  --allocation "$ALLOCATION"
  --seed "$SEED"
)
if [[ -n "$LIMIT" ]]; then
  ARGS+=(--limit "$LIMIT")
fi
python "${ARGS[@]}"

