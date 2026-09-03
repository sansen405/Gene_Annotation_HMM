# HSMM-Gene — Structure-First Semi-Markov Gene Finding

A C++ **hidden semi-Markov model (HSMM)** for ab-initio gene prediction in
fission yeast. A 21-state topology (intergenic, start/stop codons, frame-tracked
exons, donor/acceptor splice sites, frame-tracked introns) is trained from a
genome FASTA plus a GFF annotation. Intron duration is explicit (`log P(L)` at
acceptor close), shared by Viterbi, Forward–Backward, dual-strand merge, and the
prediction CLI. Donor/acceptor and translation starts are scored by PyTorch
CNNs; the HSMM consumes their per-position log-odds.

**DESRES V2 research platform:** pluggable neural emissions, calibration, SSL,
segment CRF, and CORAL adaptation — see
[`docs/DESRES_V2_METHODS.md`](docs/DESRES_V2_METHODS.md) and
[`experiments/desres_v2/`](experiments/desres_v2/).

**Current version: 5.0** — generative-consistent intron HSMM + structure-first
metrics. Methods note: [`docs/HSMM_GENE_METHODS.md`](docs/HSMM_GENE_METHODS.md).

## Results (structure-first framing; V4.2 holdout snapshot)

Four held-out chromosomes (one per Schizosaccharomyces species), both strands,
23,565,610 evaluated bases. Full nucleotide tables:
[`validation/results/version_4/`](validation/results/version_4/).
V5 regeneration protocol: [`validation/results/version_5/`](validation/results/version_5/).

| Metric | Value |
| --- | --- |
| Exact exon P / R | 0.8400 / 0.8187 |
| Exact gene P / R (sens/spec) | 0.8080 / 0.8157 |
| Intron F1 | 0.8413 (P 0.9221 / R 0.7735) |
| Start / stop boundary P/R | 0.8474/0.8554 · 0.9164/0.9252 |
| Donor / acceptor boundary P/R | 0.9043/0.8516 · 0.9049/0.8522 |
| Coding F1 (secondary) | 0.9757 |
| Exact 21-state accuracy | 0.9723 |
| Predicted genes | 7514 (gold 7443) |

Nucleotide coding F1 (~0.98) outruns exact gene/exon recovery (~0.81–0.84) —
structure metrics are the harder, more honest claim and the V5 headline.

### External baselines (Augustus + Helixer)

Same four holdout FASTAs. Reports:
[`validation/baselines/augustus/combined.txt`](validation/baselines/augustus/combined.txt),
[`validation/baselines/helixer/combined.txt`](validation/baselines/helixer/combined.txt).

| Metric | Augustus | Helixer | This HSMM |
| --- | ---: | ---: | ---: |
| Exact exon / gene | — | — | 0.84/0.82 · 0.81/0.82 |
| Intron F1 | 0.6922 | 0.8744 | 0.8413 |
| Coding F1 (secondary) | 0.9617 | 0.9833 | 0.9757 |
| Start boundary P / R | 0.8272 / 0.7475 | 0.8891 / 0.8912 | 0.8474 / 0.8554 |
| Stop boundary P / R | 0.9435 / 0.8524 | 0.9273 / 0.9293 | 0.9164 / 0.9252 |

## Model

The HSMM emits one symbol per base. Each state family draws from a different
emission model:

| State family | Emission | Detail |
| --- | --- | --- |
| `INTERGENIC` | Markov order-1 | `log P(base \| previous base)` |
| `INTRON_1/2/3` | Markov order-5 + **semi-Markov duration** | body content + `log P(L)` at close |
| `EXON_FRAME_1/2/3` | Markov order-5, per frame | codon-periodic 1024×4 table per frame |
| `DONOR_1/2/3` | splice CNN log-odds | requires canonical `GT` |
| `ACCEPTOR_1/2/3` | splice CNN log-odds | requires canonical `AG` |
| `START_CODON_1` | start CNN log-odds | requires `ATG` |
| `START_CODON_2/3` | deterministic | second/third start bases `T`, `G` |
| `STOP_CODON_1/2/3` | deterministic | one of `TAA`, `TAG`, `TGA` |

Duration helpers live in [`src/decoding/Intron_Duration.hpp`](src/decoding/Intron_Duration.hpp)
(`histogram` default, `nb` parametric, `none` geometric ablation). CNN scores are
**required** at decode time. The decoder also applies a gene-start penalty.

### Complexity

Viterbi / Forward–Backward iterate only legal emitting edges from
`Topology::Transitions` (`emitting_predecessors` / `emitting_successors`):
**O(T · E)** time with E ≈ 28 vs dense S² ≈ 361 (plus **O(T · D)** duration
rows when the length model is active). Scores stream in 2 rows; backpointers
remain **O(T · S)**.

## Methods and limitations

- **Hybrid model.** CNN log-odds are plugged in as emissions of a generative
  HSMM (not a jointly trained CRF). Scale/bias set the operating point.
- **Generative consistency (V5).** MAP, FB, `path_log_prob` strand merge, and
  `hmm_predict_fna` share the same duration table and min/max gates.
- **Holdout scope.** Headline numbers are four closely related fission yeasts.
  `fungi_diverse` is scaffolded but not the reported eval.
- **Metrics.** Exact gene/exon structure and intron F1 are primary; nucleotide
  coding F1 is secondary. Topology is single-isoform CDS.
- **Calibration hygiene.** `--tune-cnn-calibration` / `--structure-objective` on
  evaluation labels are diagnostics—do not treat them as the reported operating
  point (profile defaults: donor/acceptor scale=1, bias=0).

## Repository layout

```text
.
├── src/
│   ├── topology/Topology.hpp            # State/Nucleotide enums, sparse transitions
│   ├── parsers/                         # FASTA -> sequence, GFF -> per-base labels
│   ├── model/
│   │   ├── transition/                  # bigram transition matrix
│   │   ├── emission/                    # HMM emission tables + CNN bridge
│   │   ├── cnn/{splice,start}/          # PyTorch CNNs + C++ score loaders
│   │   └── training_pipeline/           # cached-model build scripts
│   ├── decoding/                        # sparse Viterbi + FB + Intron_Duration
│   ├── tools/                           # predict_fna, split_genome_data
│   └── genome_profiles/                 # per-profile folders (JSON + dataset scripts)
├── validation/                          # holdout runner, bench_viterbi, baselines/, results/
├── experiments/version_5/               # duration + structure ablations
├── docs/                                # methods note + interview pack
├── Makefile / CMakeLists.txt
└── frontend/                            # local React UI + Node API
```

## Genome profile

Each profile in `src/genome_profiles/` declares input paths, held-out test
chromosomes, gene-quality filters, CNN score paths (plus `*_minus` variants for
dual-strand decode), emission choices, and smoothing. See
[`fission_yeasts.json`](src/genome_profiles/fission_yeasts/fission_yeasts.json).

## Build and run

Requires a C++17 compiler and [nlohmann/json](https://github.com/nlohmann/json)
(`brew install nlohmann-json`, `apt install nlohmann-json3-dev`, or set
`JSON_INCLUDE`).

```sh
make all          # tests, validation, predict, train-matrices -> build/
make run-tests
make run-validation
make bench-viterbi
```

Python CNN training deps: `python3 -m pip install -r requirements.txt`.

## Training pipeline

```sh
python3 src/model/training_pipeline/train_cached_model.py \
  --profile src/genome_profiles/fission_yeasts/fission_yeasts.json
```

## Prediction CLI

```sh
make predict
build/hmm_predict_fna --profile src/genome_profiles/fission_yeasts/fission_yeasts.json \
  --fna INPUT.fna --splice-cnn-scores SPLICE.tsv --start-cnn-scores START.tsv \
  > predictions.json
```

`--splice-cnn-scores` and `--start-cnn-scores` are required. Prediction builds
the same train histogram duration table (p95 cap) used in validation.

## Duration / structure ablations

```sh
bash experiments/version_5/duration/run_duration_ablation.sh
bash experiments/version_5/structure/run_structure_penalty_sweep.sh
```

Validation flags: `--duration-model {histogram,nb,none}`, `--structure-objective`,
`--tune-gene-start-penalty`.

## Frontend

```sh
cd frontend && npm install && npm run dev:all
```

## Version history

- **5.0** — Structure-first HSMM packaging: shared `Intron_Duration` helpers;
  FB min/max gates match Viterbi; `path_log_prob` and `hmm_predict_fna` use the
  duration model; parametric NB duration + dwell diagnostic; structure-objective
  sweeps; methods note. Headlines = exact exon/gene + intron F1.
- **4.2** — intron body emission upgraded from Markov order-1 to Markov order-5;
  intron F1 0.8169 → 0.8413 on dual-strand eval.
- **4.1** — dual-strand decoding with reverse-complement merge by path score.
- **4.0** — translation start CNN (replaces start PSSM).
- **3.1** — semi-Markov intron length model in Viterbi.
- **3.0** — calibrated splice CNN.
- **2.x** — CNN donor/acceptor emissions; cached training pipeline.
- **1.x** — frame-specific exon emissions; Forward-Backward posterior confidence.
