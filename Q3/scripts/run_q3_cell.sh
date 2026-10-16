#!/usr/bin/env bash
set -euo pipefail

DOMAIN="${1:?domain}"
STRUCTURE="${2:?structure}"
GRAPH="${3:?graph_id}"
SEED="${4:?seed}"
CONFIG="${5:-configs/q3.yaml}"

exec opendgo-q3 run \
  --config "$CONFIG" \
  --domain "$DOMAIN" \
  --structure "$STRUCTURE" \
  --graph "$GRAPH" \
  --seed "$SEED"
