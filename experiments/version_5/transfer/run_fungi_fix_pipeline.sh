#!/usr/bin/env bash
# Fix fungi generalization: train fungi start CNN, score dual-strand emissions,
# train-fit start calibration, then dual-strand holdout validation.
#
# Usage (repo root):
#   bash experiments/version_5/transfer/run_fungi_fix_pipeline.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"

PY="${PY:-$ROOT/.venv/bin/python}"
PROFILE_BASE="src/genome_profiles/fungi_diverse/fungi_diverse_span.json"
PROFILE_FIXED="src/genome_profiles/fungi_diverse/fungi_diverse_span_fixed.json"
OUT_DIR="validation/results/fungi_span_fixed"
LOG_DIR="experiments/version_5/transfer"
START_MODEL="src/model/cnn/start/trained_models/fungi_diverse_span_start_cnn.pt"
SPLICE_MODEL="src/model/cnn/splice/trained_models/fungi_diverse_splice_cnn.pt"

mkdir -p "$OUT_DIR" "$LOG_DIR" "$(dirname "$START_MODEL")"
make validation train-matrices >/dev/null

SPECIES=(s_cerevisiae n_crassa cr_neoformans r_delemar b_dendrobatidis)

train_fastas=()
train_gffs=()
test_fastas=()
test_gffs=()
train_start=()
test_start=()
train_start_m=()
test_start_m=()
train_splice=()
test_splice=()
train_splice_m=()
test_splice_m=()

for sp in "${SPECIES[@]}"; do
  train_fastas+=("genome_data/fungi_diverse/$sp/train/${sp}_train.fna")
  train_gffs+=("genome_data/fungi_diverse/$sp/train/${sp}_train.gff")
  test_fastas+=("genome_data/fungi_diverse/$sp/test/${sp}_test.fna")
  test_gffs+=("genome_data/fungi_diverse/$sp/test/${sp}_test.gff")
  train_start+=("genome_data/fungi_diverse/$sp/train/${sp}_start_cnn_scores.tsv")
  test_start+=("genome_data/fungi_diverse/$sp/test/${sp}_start_cnn_scores.tsv")
  train_start_m+=("genome_data/fungi_diverse/$sp/train/${sp}_start_cnn_scores_minus.tsv")
  test_start_m+=("genome_data/fungi_diverse/$sp/test/${sp}_start_cnn_scores_minus.tsv")
  train_splice+=("genome_data/fungi_diverse/$sp/train/${sp}_splice_cnn_scores.tsv")
  test_splice+=("genome_data/fungi_diverse/$sp/test/${sp}_splice_cnn_scores.tsv")
  train_splice_m+=("genome_data/fungi_diverse/$sp/train/${sp}_splice_cnn_scores_minus.tsv")
  test_splice_m+=("genome_data/fungi_diverse/$sp/test/${sp}_splice_cnn_scores_minus.tsv")
done

echo "=== [1/5] Train fungi start CNN (or reload if checkpoint exists) ==="
# Force retrain by removing checkpoint if FORCE_RETRAIN=1
if [[ "${FORCE_RETRAIN:-0}" == "1" && -f "$START_MODEL" ]]; then
  rm -f "$START_MODEL"
fi
"$PY" src/model/cnn/start/train_start_cnn_scores.py \
  --train-fasta "${train_fastas[@]}" \
  --train-gff "${train_gffs[@]}" \
  --test-fasta "${test_fastas[@]}" \
  --model-out "$START_MODEL" \
  --train-scores-out "${train_start[@]}" \
  --test-scores-out "${test_start[@]}" \
  --train-scores-minus-out "${train_start_m[@]}" \
  --test-scores-minus-out "${test_start_m[@]}" \
  --sparse-scores --score-batch-size 8192 --require-3n-cds \
  --epochs "${EPOCHS:-10}" \
  2>&1 | tee "$LOG_DIR/fix_start_cnn.log"

echo "=== [2/5] Score splice CNN train(+)/minus and test minus (reload existing fungi splice ckpt) ==="
"$PY" src/model/cnn/splice/train_splice_cnn_scores.py \
  --train-fasta "${train_fastas[@]}" \
  --train-gff "${train_gffs[@]}" \
  --test-fasta "${test_fastas[@]}" \
  --model-out "$SPLICE_MODEL" \
  --train-scores-out "${train_splice[@]}" \
  --test-scores-out "${test_splice[@]}" \
  --train-scores-minus-out "${train_splice_m[@]}" \
  --test-scores-minus-out "${test_splice_m[@]}" \
  --sparse-scores --score-batch-size 8192 --require-3n-cds --min-intron-bp 20 \
  2>&1 | tee "$LOG_DIR/fix_splice_scores.log"

echo "=== [3/5] Sync fixed profile to use fungi start model ==="
"$PY" - <<'PY'
import json
from pathlib import Path
base = json.loads(Path("src/genome_profiles/fungi_diverse/fungi_diverse_span.json").read_text())
base["name"] = "fungi_diverse_span_fixed"
base["start_cnn"]["model"] = "src/model/cnn/start/trained_models/fungi_diverse_span_start_cnn.pt"
base["start_cnn"]["start_scale"] = 1.0
base["start_cnn"]["start_bias"] = 0.0
base["filters"]["include_minus_strand"] = True
Path("src/genome_profiles/fungi_diverse/fungi_diverse_span_fixed.json").write_text(
    json.dumps(base, indent=4) + "\n"
)
print("wrote fungi_diverse_span_fixed.json")
PY

echo "=== [4/5] Train-fit start CNN scale/bias on TRAINING labels ==="
./build/full_genome_validation \
  --profile "$PROFILE_FIXED" \
  --duration-model histogram \
  --tune-start-calibration \
  --tune-only \
  --tune-subset-ranges "${TUNE_SUBSET:-128}" \
  2>&1 | tee "$LOG_DIR/fix_tune_start.log"

# Parse selected bias/scale from tune log and stamp into profile
"$PY" - <<'PY'
import json, re
from pathlib import Path
log = Path("experiments/version_5/transfer/fix_tune_start.log").read_text()
# Selected start-CNN calibration: scale=... bias=...
m = re.search(r"Selected start-CNN calibration:\s*start_scale=([-\d.]+)\s*start_bias=([-\d.]+)", log)
if not m:
    m = re.search(r"start_scale=([-\d.]+)\s*start_bias=([-\d.]+)", log)
scale, bias = (float(m.group(1)), float(m.group(2))) if m else (1.0, 0.0)
for rel in [
    "src/genome_profiles/fungi_diverse/fungi_diverse_span_fixed.json",
    "src/genome_profiles/fungi_diverse/fungi_diverse_span.json",
]:
    p = Path(rel)
    j = json.loads(p.read_text())
    j["start_cnn"]["start_scale"] = scale
    j["start_cnn"]["start_bias"] = bias
    j["start_cnn"]["model"] = "src/model/cnn/start/trained_models/fungi_diverse_span_start_cnn.pt"
    j["filters"]["include_minus_strand"] = True
    p.write_text(json.dumps(j, indent=4) + "\n")
    print(f"stamped {rel}: start_scale={scale} start_bias={bias}")
PY

echo "=== [5/5] Dual-strand holdout validation ==="
./build/full_genome_validation \
  --profile "$PROFILE_FIXED" \
  --duration-model histogram \
  --results-dir "$OUT_DIR" \
  2>&1 | tee "$LOG_DIR/fix_validation.log"

"$PY" - <<'PY'
from pathlib import Path
import re

def parse(path):
    text = Path(path).read_text()
    def row(label):
        m = re.search(rf'^{label}\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)', text, re.M)
        return m.groups() if m else ('', '', '')
    def br(label):
        m = re.search(rf'^{label}\s+([0-9.]+)\s+([0-9.]+)', text, re.M)
        return m.groups() if m else ('', '')
    coding, intron = row('coding'), row('intron')
    exon, gene = br('exon'), br('gene')
    return coding[2], intron[2], exon[0], exon[1], gene[0], gene[1]

rows = []
for name, path in [
    ('fission_V5', 'validation/results/version_5/combined.txt'),
    ('prev_zeroshot', 'validation/results/fungi_span_zeroshot/combined.txt'),
    ('prev_indomain_plus', 'validation/results/fungi_span_indomain/combined.txt'),
    ('fixed_dualstrand', 'validation/results/fungi_span_fixed/combined.txt'),
]:
    p = Path(path)
    if p.exists():
        c, i, ep, er, gp, gr = parse(p)
        rows.append((name, c, i, ep, er, gp, gr))

out = Path('experiments/version_5/transfer/fix_comparison.tsv')
out.write_text(
    'setting\tcoding_f1\tintron_f1\texon_p\texon_r\tgene_p\tgene_r\n'
    + '\n'.join('\t'.join(r) for r in rows) + '\n'
)
print(out.read_text())
PY

echo "FUNGI_FIX_DONE"
