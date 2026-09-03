# HSMM-Gene methods note (V5)

## Claim

A generative-consistent intron hidden semi-Markov model (HSMM) for fission-yeast
gene structure: Viterbi MAP, Forward–Backward posteriors, dual-strand path
scoring, and the prediction CLI share one intron-duration model. Headlines are
**exact exon / exact gene / intron F1**, not nucleotide coding F1.

## Model

- 21-state topology with frame-tracked exons and introns
  (`src/topology/Topology.hpp`).
- Semi-Markov intron duration: drop geometric self-loop cost while dwelling;
  charge `log P(L)` once on intron-body → acceptor
  (`src/decoding/Intron_Duration.hpp`, Viterbi, Forward–Backward).
- Hard min/max duration gates shared by MAP and FB.
- CNN log-odds emissions for donor/acceptor/start (canonical motif gates).
- Dual-strand merge ranks genes by HSMM `path_log_prob` (includes duration).

## Duration variants

| Kind | `--duration-model` |
| --- | --- |
| Empirical histogram (default) | `histogram` |
| Negative binomial | `nb` |
| Geometric (ablation) | `none` |

## Eval protocol

- Profile: `fission_yeasts` (four held-out chromosomes, both strands).
- Primary metrics: exact exon P/R/F1, exact gene P/R, intron nucleotide F1,
  splice-boundary P/R.
- Secondary: coding nucleotide F1, exact 21-state accuracy.
- Baselines: Augustus + Helixer on the same holdout FASTAs
  (`validation/baselines/`).
- Do not present `--tune-cnn-calibration` on evaluation labels as the reported
  operating point. `--structure-objective` retargets diagnostic sweeps only.

## Limitations

- Single-isoform CDS topology (no UTRs / alternative splicing).
- Holdouts are closely related Schizosaccharomyces species.
- Hybrid CNN emissions inside a generative HSMM (not joint CRF training).
