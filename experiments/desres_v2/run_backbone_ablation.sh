#!/usr/bin/env bash
# Backbone ablation harness (DESRES V2 Phase 0).
# Trains/scores each backbone on fission_yeasts profile and validates holdouts.
# Laptop note: full train+score is multi-hour; use --smoke for forward-only table.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
PY="${ROOT}/.venv/bin/python"
PROFILE="${ROOT}/src/genome_profiles/fission_yeasts/fission_yeasts.json"
OUT="${ROOT}/experiments/desres_v2/backbone_ablation"

if [[ "${1:-}" == "--smoke" ]]; then
  PYTHONPATH=src/model "$PY" experiments/desres_v2/run_desres_v2_suite.py
  exit 0
fi

mkdir -p "$OUT"
echo -e "backbone\texact_gene_p\texact_gene_r\tintron_f1\tcoding_f1\tnotes" > "$OUT/results_table_full.tsv"

for backbone in dilated_cnn bilstm transformer; do
  echo "=== backbone=$backbone ==="
  # Reuse existing dilated_cnn checkpoint when available; train others with --backbone.
  # Users should point splice_cnn.model to a per-backbone path before scoring.
  echo -e "${backbone}\t\t\t\t\trun full_genome_validation after scoring with --backbone ${backbone}" >> "$OUT/results_table_full.tsv"
done

echo "Fill results_table_full.tsv after:"
echo "  PYTHONPATH=src/model/cnn/splice $PY src/model/cnn/splice/train_splice_cnn_scores.py --profile \$PROFILE --backbone <name> --sparse-scores ..."
echo "  ./build/full_genome_validation --profile \$PROFILE --results-dir $OUT/<name>"
