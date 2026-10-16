#!/usr/bin/env bash
set -euo pipefail

DOMAIN="${1:?domain}"
STRUCTURE="${2:?structure}"
ORGANIZATION="${3:?organization}"
SEED="${4:?seed}"
PRESSURE="${5:?fault_pressure_percent}"
CONFIG="${6:-configs/q4.yaml}"

exec opendgo-q4 run-fault \
  --config "$CONFIG" \
  --domain "$DOMAIN" \
  --structure "$STRUCTURE" \
  --organization "$ORGANIZATION" \
  --seed "$SEED" \
  --pressure "$PRESSURE"
