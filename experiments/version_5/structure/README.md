# Structure-first operating point (V5)

Use exact exon / exact gene F1 as the tuning objective instead of nucleotide
intron + splice-boundary F1.

```sh
make validation

# Diagnostic gene-start penalty sweep under structure objective
bash experiments/version_5/structure/run_structure_penalty_sweep.sh

# Eval-label splice calibration under structure objective (do NOT report as
# the operating point — train-fit / profile defaults remain the claim)
./build/full_genome_validation \
  --profile src/genome_profiles/fission_yeasts/fission_yeasts.json \
  --structure-objective \
  --tune-cnn-calibration \
  --tune-only \
  --tune-subset-ranges 64
```

Reported portfolio operating point stays profile defaults (start bias −6,
splice scale/bias 1/0, gene-start penalty 1.0, histogram duration) unless a
**train-fit** sweep is documented separately.
