#!/usr/bin/env bash
# DESRES V2 full deep run — optimized for laptop.
# Scores TEST (+/−) only (holdout validation path); trains BiLSTM/Transformer splice.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

PY="${PY:-$ROOT/.venv/bin/python}"
export PYTHONUNBUFFERED=1
PROFILE_FISSION="src/genome_profiles/fission_yeasts/fission_yeasts.json"
PROFILE_FUNGI_ZS="src/genome_profiles/fungi_diverse/fungi_span_zeroshot_from_fission.json"
OUT="$ROOT/experiments/desres_v2"
LOG="$OUT/deep_run.log"
VAL_BIN="$ROOT/build/full_genome_validation"
EPOCHS="${EPOCHS:-4}"
NEG_PER_POS="${NEG_PER_POS:-3}"
SSL_MAX_BASES="${SSL_MAX_BASES:-25000000}"
SSL_SAMPLES="${SSL_SAMPLES:-64}"
SSL_EPOCHS="${SSL_EPOCHS:-2}"
# Comma-separated: dilated_cnn,bilstm,transformer — default all
BACKBONES_CSV="${BACKBONES:-dilated_cnn,bilstm,transformer}"
IFS=',' read -r -a BACKBONE_LIST <<< "$BACKBONES_CSV"

SPECIES=(s_pombe s_japonicus s_octosporus s_cryophilus)
FUNGI_SPECIES=(s_cerevisiae n_crassa cr_neoformans r_delemar b_dendrobatidis)

mkdir -p "$OUT"/{backbone_ablation,calibration,ssl,segment_crf,coral,results,profiles,logs}

ts() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(ts)] $*" | tee -a "$LOG"; }

# fresh log for this run
: > "$LOG"
log "===== DESRES V2 DEEP RUN START ====="
log "epochs=$EPOCHS neg=$NEG_PER_POS ssl_max_bases=$SSL_MAX_BASES ssl_samples=$SSL_SAMPLES backbones=${BACKBONE_LIST[*]}"

if [[ ! -x "$VAL_BIN" ]]; then
  log "building full_genome_validation"
  make validation
fi

write_backbone_profile() {
  local backbone="$1"
  local profile_out="$OUT/profiles/fission_${backbone}.json"
  "$PY" - "$PROFILE_FISSION" "$backbone" "$profile_out" "$OUT" <<'PY'
import json, sys
from pathlib import Path
base = json.loads(Path(sys.argv[1]).read_text())
bb, out, root = sys.argv[2], Path(sys.argv[3]), Path(sys.argv[4])
species = ["s_pombe", "s_japonicus", "s_octosporus", "s_cryophilus"]
score_root = root / "backbone_ablation" / bb / "scores"
base["name"] = f"fission_yeasts_{bb}"
base["splice_cnn"]["model"] = str(root / "backbone_ablation" / bb / f"splice_{bb}.pt")
# Train scores unused unless --tune-start-calibration; point at test for path presence.
base["splice_cnn"]["train_scores"] = [str(score_root / f"{sp}_test_splice.tsv") for sp in species]
base["splice_cnn"]["test_scores"] = [str(score_root / f"{sp}_test_splice.tsv") for sp in species]
base["splice_cnn"]["train_scores_minus"] = [str(score_root / f"{sp}_test_splice_minus.tsv") for sp in species]
base["splice_cnn"]["test_scores_minus"] = [str(score_root / f"{sp}_test_splice_minus.tsv") for sp in species]
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(base, indent=2) + "\n")
print(out)
PY
}

link_or_score_dilated() {
  local dir="$OUT/backbone_ablation/dilated_cnn"
  local scores="$dir/scores"
  mkdir -p "$scores" "$dir"
  cp -f src/model/cnn/splice/trained_models/fission_yeasts_splice_cnn.pt "$dir/splice_dilated_cnn.pt"
  local need=0
  for sp in "${SPECIES[@]}"; do
    local src="genome_data/fission_yeasts/$sp/test/${sp}_splice_cnn_scores.tsv"
    local srcm="genome_data/fission_yeasts/$sp/test/${sp}_splice_cnn_scores_minus.tsv"
    local dst="$scores/${sp}_test_splice.tsv"
    local dstm="$scores/${sp}_test_splice_minus.tsv"
    if [[ -s "$src" && -s "$srcm" ]]; then
      ln -sfn "$ROOT/$src" "$dst"
      ln -sfn "$ROOT/$srcm" "$dstm"
    else
      need=1
    fi
  done
  write_backbone_profile dilated_cnn >/dev/null
  if [[ "$need" -eq 1 ]]; then
    log "dilated production test scores missing; will score via trainer"
    return 1
  fi
  log "dilated_cnn: linked production test (+/−) scores"
  return 0
}

train_score_backbone() {
  local backbone="$1"
  local dir="$OUT/backbone_ablation/$backbone"
  local scores="$dir/scores"
  mkdir -p "$scores" "$dir"
  local model="$dir/splice_${backbone}.pt"
  local profile
  profile="$(write_backbone_profile "$backbone")"

  if [[ "$backbone" == "dilated_cnn" ]]; then
    if link_or_score_dilated; then
      :
    else
      cp -f src/model/cnn/splice/trained_models/fission_yeasts_splice_cnn.pt "$model"
    fi
  fi

  local test_fastas=() train_fastas=() train_gffs=()
  local test_out=() test_m=() train_out=() train_m=()
  for sp in "${SPECIES[@]}"; do
    train_fastas+=("genome_data/fission_yeasts/$sp/train/${sp}_train.fna")
    train_gffs+=("genome_data/fission_yeasts/$sp/train/${sp}_train.gff")
    test_fastas+=("genome_data/fission_yeasts/$sp/test/${sp}_test.fna")
    # Dummy train outs (trainer requires them); write tiny then overwrite only if needed.
    train_out+=("$scores/${sp}_train_splice_UNUSED.tsv")
    train_m+=("$scores/${sp}_train_splice_minus_UNUSED.tsv")
    test_out+=("$scores/${sp}_test_splice.tsv")
    test_m+=("$scores/${sp}_test_splice_minus.tsv")
  done

  local need=0
  for f in "${test_out[@]}" "${test_m[@]}"; do
    [[ -s "$f" ]] || need=1
  done
  [[ -f "$model" ]] || need=1

  if [[ "$need" -eq 0 ]]; then
    log "backbone=$backbone test scores present; skip train/score"
  else
    log "=== Phase0 train/score backbone=$backbone (TEST strands only) ==="
    if [[ "$backbone" != "dilated_cnn" ]]; then
      rm -f "$model"
    fi
    # Score only test by passing test fasta as both train and test with matching gffs for train label sampling.
    # For existing dilated checkpoint we only need scoring: provide train fasta/gff so CLI validates,
    # model exists → skip training, then emit train+test outs — to avoid huge train scoring, symlink
    # train unused outputs after a minimal run isn't possible. Instead: use a custom score-only call.
    PYTHONPATH="$ROOT/src/model/cnn/splice:$ROOT/src/model" \
      "$PY" "$OUT/score_backbone_test_only.py" \
      --backbone "$backbone" \
      --model-out "$model" \
      --train-fasta "${train_fastas[@]}" \
      --train-gff "${train_gffs[@]}" \
      --test-fasta "${test_fastas[@]}" \
      --test-scores-out "${test_out[@]}" \
      --test-scores-minus-out "${test_m[@]}" \
      --epochs "$EPOCHS" \
      --negatives-per-positive "$NEG_PER_POS" \
      2>&1 | tee "$OUT/logs/train_score_${backbone}.log"
  fi

  log "=== Phase0 validate backbone=$backbone ==="
  local res="$dir/validation"
  mkdir -p "$res"
  if [[ ! -s "$res/combined.txt" ]]; then
    "$VAL_BIN" --profile "$profile" --results-dir "$res" --duration-model histogram \
      2>&1 | tee "$OUT/logs/validate_${backbone}.log"
  else
    log "validation already present for $backbone"
  fi
}

# ---------------------------------------------------------------------------
for bb in "${BACKBONE_LIST[@]}"; do
  train_score_backbone "$bb"
done

log "=== Phase0 parse backbone metrics ==="
PYTHONPATH="$ROOT/src/model" "$PY" "$OUT/parse_deep_results.py" --phase backbone

log "=== Phase1 deep calibration ==="
PYTHONPATH="$ROOT/src/model" "$PY" "$OUT/run_calibration_deep.py" \
  2>&1 | tee "$OUT/logs/calibration_deep.log"

log "=== Phase2 deep SSL ==="
SSL_MAX_BASES="$SSL_MAX_BASES" SSL_SAMPLES="$SSL_SAMPLES" SSL_EPOCHS="$SSL_EPOCHS" \
  PYTHONPATH="$ROOT/src/model" "$PY" "$OUT/run_ssl_deep.py" \
  2>&1 | tee "$OUT/logs/ssl_deep.log"

log "=== Phase3 deep segment CRF ==="
PYTHONPATH="$ROOT/src/model" "$PY" "$OUT/run_segment_crf_deep.py" \
  2>&1 | tee "$OUT/logs/segment_crf_deep.log"

log "=== Phase4 CORAL adapt ==="
PYTHONPATH="$ROOT/src/model" "$PY" "$OUT/run_coral_deep.py" \
  2>&1 | tee "$OUT/logs/coral_deep.log"

ADAPTED="$OUT/coral/adapted_splice.pt"
if [[ -f "$ADAPTED" ]]; then
  coral_scores="$OUT/coral/scores"
  mkdir -p "$coral_scores"
  train_fastas=() train_gffs=() test_fastas=()
  test_out=() test_m=()
  for sp in "${FUNGI_SPECIES[@]}"; do
    train_fastas+=("genome_data/fungi_diverse/$sp/train/${sp}_train.fna")
    train_gffs+=("genome_data/fungi_diverse/$sp/train/${sp}_train.gff")
    test_fastas+=("genome_data/fungi_diverse/$sp/test/${sp}_test.fna")
    test_out+=("$coral_scores/${sp}_test_splice.tsv")
    test_m+=("$coral_scores/${sp}_test_splice_minus.tsv")
  done
  need=0
  for f in "${test_out[@]}" "${test_m[@]}"; do
    [[ -s "$f" ]] || need=1
  done
  if [[ "$need" -eq 1 ]]; then
    log "scoring fungi TEST with CORAL-adapted splice model"
    PYTHONPATH="$ROOT/src/model/cnn/splice:$ROOT/src/model" \
      "$PY" "$OUT/score_backbone_test_only.py" \
      --backbone dilated_cnn \
      --model-out "$ADAPTED" \
      --train-fasta "${train_fastas[@]}" \
      --train-gff "${train_gffs[@]}" \
      --test-fasta "${test_fastas[@]}" \
      --test-scores-out "${test_out[@]}" \
      --test-scores-minus-out "${test_m[@]}" \
      --epochs 1 \
      2>&1 | tee "$OUT/logs/coral_fungi_score.log"
  fi

  "$PY" - "$PROFILE_FUNGI_ZS" "$OUT/profiles/fungi_coral_adapted.json" "$coral_scores" <<'PY'
import json, sys
from pathlib import Path
zs = json.loads(Path(sys.argv[1]).read_text())
out = Path(sys.argv[2])
scores = Path(sys.argv[3])
species = ["s_cerevisiae", "n_crassa", "cr_neoformans", "r_delemar", "b_dendrobatidis"]
zs["name"] = "fungi_span_coral_adapted"
zs["splice_cnn"]["model"] = "experiments/desres_v2/coral/adapted_splice.pt"
zs["splice_cnn"]["train_scores"] = [str(scores / f"{sp}_test_splice.tsv") for sp in species]
zs["splice_cnn"]["test_scores"] = [str(scores / f"{sp}_test_splice.tsv") for sp in species]
zs["splice_cnn"]["train_scores_minus"] = [str(scores / f"{sp}_test_splice_minus.tsv") for sp in species]
zs["splice_cnn"]["test_scores_minus"] = [str(scores / f"{sp}_test_splice_minus.tsv") for sp in species]
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(zs, indent=2) + "\n")
print(out)
PY

  res="$OUT/coral/validation"
  mkdir -p "$res"
  if [[ ! -s "$res/combined.txt" ]]; then
    "$VAL_BIN" --profile "$OUT/profiles/fungi_coral_adapted.json" --results-dir "$res" \
      --duration-model histogram \
      2>&1 | tee "$OUT/logs/validate_coral.log"
  else
    log "CORAL validation already present"
  fi
else
  log "WARNING: adapted_splice.pt missing"
fi

log "=== Phase5 master pack ==="
( make bench-viterbi 2>&1 | tee "$OUT/results/bench_viterbi.txt" ) || true
PYTHONPATH="$ROOT/src/model" "$PY" "$OUT/parse_deep_results.py" --phase master \
  2>&1 | tee "$OUT/logs/master_pack.log"

log "===== DESRES V2 DEEP RUN DONE ====="
echo "DESRES_V2_DEEP_DONE" | tee -a "$LOG"
