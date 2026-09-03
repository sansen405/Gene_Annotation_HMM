# Version 5 — intron duration ablation

Structure-first HSMM contribution: same fission-yeast holdouts, three duration
models sharing one decoder.

| Kind | Flag | Meaning |
| --- | --- | --- |
| Geometric | `--duration-model none` | Empty duration table; intron self-loops keep transition costs |
| Histogram | `--duration-model histogram` | Laplace-smoothed empirical `P(L)` (default / production) |
| NegBin | `--duration-model nb` | Negative-binomial MLE fit on train intron bodies |

## Run

```sh
make validation
bash experiments/version_5/duration/run_duration_ablation.sh
```

Reports land in `experiments/version_5/duration/results/{none,histogram,nb}/`.

## Dwell diagnostic

Compare geometric mean dwell implied by transition self-loop `A` vs empirical /
histogram / NB means:

```sh
# lengths file: one intron-body length integer per line (from training)
python3 experiments/version_5/duration/dwell_diagnostic.py \
  --lengths path/to/lengths.txt \
  --self-loop-log-prob -0.02 \
  --out experiments/version_5/duration/dwell_summary.tsv
```

## Metrics to highlight

Primary: intron F1, donor/acceptor boundary P/R, **exact exon F1**, **exact gene P/R**.  
Secondary: nucleotide coding F1.
