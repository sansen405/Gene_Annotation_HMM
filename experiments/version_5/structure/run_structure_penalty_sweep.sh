#!/usr/bin/env bash
# Structure-objective gene-start penalty sweep (diagnostic on eval subset).
# Prefer train-fit operating points for reported numbers; this is for portfolio
# sensitivity analysis.
#
# Usage:
#   bash experiments/version_5/structure/run_structure_penalty_sweep.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
PROFILE="${PROFILE:-$ROOT/src/genome_profiles/fission_yeasts/fission_yeasts.json}"
BIN="${BIN:-$ROOT/build/full_genome_validation}"
OUT="${OUT:-$ROOT/experiments/version_5/structure/penalty_sweep}"

mkdir -p "$OUT"

"$BIN" \
  --profile "$PROFILE" \
  --structure-objective \
  --tune-gene-start-penalty \
  --tune-only \
  --tune-subset-ranges "${TUNE_SUBSET:-64}" \
  | tee "$OUT/console.log"

echo "Wrote $OUT/console.log"
