# Version 5 validation results

Regenerate after duration / structure changes:

```sh
make validation
./build/full_genome_validation \
  --profile src/genome_profiles/fission_yeasts/fission_yeasts.json \
  --duration-model histogram \
  --results-dir validation/results/version_5
```

Ablations:

```sh
bash experiments/version_5/duration/run_duration_ablation.sh
```

Until regenerated, V4.2 combined metrics remain the last checked-in holdout
snapshot in `validation/results/version_4/` (same profile; V5 changes generative
consistency of duration across predict/FB/`path_log_prob`, not the default
histogram fit used in V4.2 validation decode).

## Structure-first baseline comparison (V4.2 numbers, V5 headline framing)

Same four holdout FASTAs. Nucleotide coding F1 is secondary.

| Metric | Augustus | Helixer | This HSMM (V4.2 snapshot) |
| --- | ---: | ---: | ---: |
| Exact exon P / R | — | — | 0.8400 / 0.8187 |
| Exact gene P / R | — | — | 0.8080 / 0.8157 |
| Intron F1 | 0.6922 | 0.8744 | 0.8413 |
| Coding F1 (secondary) | 0.9617 | 0.9833 | 0.9757 |
| Start boundary P / R | 0.8272 / 0.7475 | 0.8891 / 0.8912 | 0.8474 / 0.8554 |
| Stop boundary P / R | 0.9435 / 0.8524 | 0.9273 / 0.9293 | 0.9164 / 0.9252 |

Augustus/Helixer combined nucleotide reports:
`validation/baselines/augustus/combined.txt`,
`validation/baselines/helixer/combined.txt`.
