#!/usr/bin/env bash
# Duration ablation: geometric (none) vs histogram vs negative-binomial HSMM
# on the fission_yeasts profile. Writes per-kind reports under
# experiments/version_5/duration/results/.
#
# Usage (from repo root):
#   bash experiments/version_5/duration/run_duration_ablation.sh
#
# Requires: build/full_genome_validation (make validation)

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
PROFILE="${PROFILE:-$ROOT/src/genome_profiles/fission_yeasts/fission_yeasts.json}"
BIN="${BIN:-$ROOT/build/full_genome_validation}"
OUT_ROOT="${OUT_ROOT:-$ROOT/experiments/version_5/duration/results}"

if [[ ! -x "$BIN" ]]; then
  echo "Missing $BIN — run: make validation" >&2
  exit 1
fi

mkdir -p "$OUT_ROOT"

for kind in none histogram nb; do
  dest="$OUT_ROOT/$kind"
  mkdir -p "$dest"
  echo "=== duration-model=$kind -> $dest ==="
  "$BIN" \
    --profile "$PROFILE" \
    --duration-model "$kind" \
    --results-dir "$dest" \
    | tee "$dest/console.log"
done

python3 - <<'PY'
from pathlib import Path
import re

root = Path("experiments/version_5/duration/results")
rows = []
for kind in ("none", "histogram", "nb"):
    path = root / kind / "combined.txt"
    if not path.exists():
        continue
    text = path.read_text()
    def grab(label):
        m = re.search(rf"{re.escape(label)}.*?([0-9]+\.[0-9]+)", text)
        return m.group(1) if m else ""
    # Fallback: keep raw pointer to report
    rows.append((kind, str(path)))

summary = root / "ablation_index.tsv"
summary.write_text("duration_model\treport\n" + "\n".join(f"{k}\t{p}" for k, p in rows) + "\n")
print(f"Wrote {summary}")
PY

echo "Duration ablation finished. See $OUT_ROOT"
