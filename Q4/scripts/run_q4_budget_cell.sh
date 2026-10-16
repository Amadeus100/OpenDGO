#!/usr/bin/env bash
set -euo pipefail

DOMAIN="${1:?domain}"
STRUCTURE="${2:?structure}"
ORGANIZATION="${3:?organization}"
SEED="${4:?seed}"
CAP="${5:?token_cap}"
CONFIG="${6:-configs/q4.yaml}"

exec opendgo-q4 run-budget \
  --config "$CONFIG" \
  --domain "$DOMAIN" \
  --structure "$STRUCTURE" \
  --organization "$ORGANIZATION" \
  --seed "$SEED" \
  --cap "$CAP"
