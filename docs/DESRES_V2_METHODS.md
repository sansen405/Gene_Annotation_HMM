# DESRES V2 methods — structured biological sequence modeling

## Claim

This repository is a **research platform** for machine learning on structured
biological sequences: pluggable neural emission models feed a biology-constrained
C++ hidden semi-Markov decoder. The product is scientific methodology (emissions,
calibration, pretraining, differentiable segment training, domain adaptation),
not a general-purpose gene annotator.

## Architecture

1. **Backbones** (`src/model/emissions_nn/`): DilatedCNN (production), BiLSTM, Transformer.
2. **Heads**: splice (donor/acceptor), start, multi-task shared encoder.
3. **Calibration**: temperature priors + ECE / reliability diagrams + MC dropout.
4. **SSL**: windowed masked nucleotide LM on in-repo fungal FASTAs (laptop scale).
5. **Segment CRF / soft-Viterbi**: differentiable fine-tuning on short labeled segments.
6. **CORAL**: second-order feature alignment for fission → fungi transfer.
7. **Decode**: unchanged sparse HSMM in C++ consuming sparse score TSVs.

## Laptop scale honesty

- Windows ≈ 121 bp; models ≤ a few million parameters; CPU/MPS.
- SSL token counts are measured and reported in `experiments/desres_v2/ssl/token_count.txt`.
- Full-chromosome differentiable Viterbi is explicitly out of scope.
- Foundation DNA models (DNABERT, Nucleotide Transformer, HyenaDNA) are future work.

## In-domain vs transfer

Fission-yeast holdouts remain strong (exact gene P≈0.81, intron F1≈0.84).
Diverse fungi zeroshot collapses structure metrics (~0.39 gene P) — motivating
CORAL adaptation experiments rather than claiming universal fungal gene finding.
See `experiments/version_5/transfer/transfer_summary.tsv` and
`experiments/desres_v2/coral/structure_metrics.tsv`.

## Experiments

```sh
PYTHONPATH=src/model .venv/bin/python experiments/desres_v2/run_desres_v2_suite.py
```

Artifacts live under `experiments/desres_v2/`.
