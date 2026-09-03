#!/usr/bin/env bash
# Run neural deep phases that don't need backbone holdout scores.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
PY="${PY:-$ROOT/.venv/bin/python}"
export PYTHONUNBUFFERED=1
OUT="$ROOT/experiments/desres_v2"
mkdir -p "$OUT/logs"
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

log "Phase1 calibration deep"
PYTHONPATH="$ROOT/src/model" "$PY" "$OUT/run_calibration_deep.py" 2>&1 | tee "$OUT/logs/calibration_deep.log"

log "Phase2 SSL deep"
SSL_MAX_BASES="${SSL_MAX_BASES:-25000000}" SSL_SAMPLES="${SSL_SAMPLES:-64}" SSL_EPOCHS="${SSL_EPOCHS:-2}" \
  PYTHONPATH="$ROOT/src/model" "$PY" "$OUT/run_ssl_deep.py" 2>&1 | tee "$OUT/logs/ssl_deep.log"

log "Phase3 segment CRF deep"
PYTHONPATH="$ROOT/src/model" "$PY" "$OUT/run_segment_crf_deep.py" 2>&1 | tee "$OUT/logs/segment_crf_deep.log"

log "Phase4 CORAL adapt"
PYTHONPATH="$ROOT/src/model" "$PY" "$OUT/run_coral_deep.py" 2>&1 | tee "$OUT/logs/coral_deep.log"

log "NEURAL_PHASES_DONE"
