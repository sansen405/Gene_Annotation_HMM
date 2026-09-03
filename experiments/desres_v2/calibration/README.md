# Calibration (DESRES V2 Phase 1)

- `ece_table.json` — ECE / accuracy / MC-dropout variance summary
- `reliability_diagram.tsv` — binned confidence vs accuracy
- `reliability_diagram.svg` — checked-in reliability plot
- `mc_dropout_diagnostics.tsv` — position-wise mean logit + variance

C++ HSMM continues to consume mean log-odds only; variance is diagnostic.
