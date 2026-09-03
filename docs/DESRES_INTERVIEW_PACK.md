# DESRES interview pack — structured sequence modeling

Use this as a one-page prep sheet. The goal is transferable computational maturity
(C++/PyTorch scientific software, DP, calibration, domain adaptation)—not
“I annotate genomes.”

**V2 methods:** [`DESRES_V2_METHODS.md`](DESRES_V2_METHODS.md)
**Linear algebra deep dive:** [`LINEAR_ALGEBRA_METHODS.md`](LINEAR_ALGEBRA_METHODS.md)

## Resume bullets (pick 1–2)

- Built a research platform for structured biological sequence modeling: pluggable
  DilatedCNN / BiLSTM / Transformer emission models integrated into a 21-state
  C++ hidden semi-Markov decoder with generative-consistent intron durations,
  dual-strand merge, and structure-first evaluation on fission-yeast holdouts
  (exact gene P≈0.81, intron F1≈0.84).
- Added laptop-scale ML methodology on top of the HSMM: ECE/MC-dropout calibration,
  windowed masked-nucleotide pretraining (measured token counts), segment-level
  CRF/soft-Viterbi fine-tuning, and CORAL domain adaptation targeting the measured
  fission→fungi generalization gap (~0.39 zeroshot gene P).

## 30-second pitch

I work on machine learning for structured biological sequences. Neural emission
models (CNN, BiLSTM, Transformer) score local signals; a constrained HSMM in C++
enforces reading frame and splice grammar via sparse DP. I care about calibration,
honest transfer evaluation, and systems that stay inspectable—hybrid classical +
deep learning under hard scientific constraints.

## 10-minute whiteboard narrative

### 1. Topology (2 min)

Draw the path:

```text
INTERGENIC → ATG → exon frames ↻ → [donor → intron → acceptor → exon]* → stop → INTERGENIC
```

Emphasize frame indices on exons *and* introns so a splice cannot scramble the
codon phase. Point to `src/topology/Topology.hpp` (21 states, sparse
`Transitions` / `emitting_predecessors`).

### 2. Why semi-Markov intron length (2 min)

Geometric self-loops imply `P(length) ∝ (1−p)^{L}`—bad for real intron length
histograms. Instead: drop the intron self-loop cost while the length model is
active; charge `log P(L)` once when the intron closes into an acceptor
(`src/decoding/Intron_Duration.hpp`, `Viterbi.cpp`, `Forward_Backward.cpp`).
V5: MAP, FB posteriors, `path_log_prob` strand merge, and `hmm_predict_fna` share
the same duration table and min/max hard gates (histogram default; NB / none
ablations under `experiments/version_5/duration/`).

### 3. CNN → emission (3 min)

- Sparse candidate scoring at canonical `GT` / `AG` / `ATG` only.
- Network logits → temperature / prior calibration → log-odds.
- Hard motif gates: impossible consensus → `LOG_ZERO` (no silent PSSM fallback).
- Own the hybrid: generative HMM + discriminative scores is deliberate and
  imperfect; start bias (−6 in the fission-yeast profile) is an operating-point
  choice from training-fit calibration, not a likelihood MLE.

Contrast with “why not a CRF / joint end-to-end?”: modularity, hard biological
constraints, and a decoder you can reason about; joint training is a natural
next step you’d discuss tradeoffs for.

### 4. One ablation (2 min)

Order-5 intron body (V4.2) vs order-1 (V4.1): intron F1 0.8169 → 0.8413, intron
recall 0.7304 → 0.7735, donor/acceptor recall ~0.80 → ~0.85, precision essentially
flat. Story: content model lets borderline CNN splice scores flip into accepted
introns without dumping precision.

Optional second ablation: gene-start penalty sweep in `experiments/version_3/gene_penalty/`.

### 5. Complexity / systems closer (1 min)

Sparse DP is `O(T · E)` with `E ≈ 28` legal emitting edges vs dense `S² ≈ 361`.
Memory is still `O(T · S)` for the DP tables (~few hundred MiB per multi-Mbp
chromosome). Know `make bench-viterbi` numbers; next levers: chunking, beam,
streaming backpointers.

**Semiring one-liner (optional if they lean algorithms):** Viterbi is the same
sparse mat-vec as Forward, but in the **max-plus** semiring instead of
**log-sum-exp**. Illegal edges are structural zeros in that operator—not soft
weights. Full whiteboard + matrix inventory:
[`LINEAR_ALGEBRA_METHODS.md`](LINEAR_ALGEBRA_METHODS.md).

## Probes to answer before they ask

| Probe | Honest answer |
| --- | --- |
| External baseline? | Augustus on the same four holdouts; see `validation/baselines/augustus/combined.txt`. Pombe model used for all four Schizosaccharomyces species. HMM coding F1 0.976 vs Augustus 0.962; intron F1 0.841 vs 0.692; Augustus higher stop precision. |
| Generalization? | Holdouts are closely related fission yeasts; `fungi_diverse` is scaffolded, not the headline. |
| Eval leakage? | Never present `--tune-cnn-calibration` on eval labels as the operating point. |
| Transformers? | None. Title was cleaned; CNNs only. |
| Why DESRES? | Scientific software + ML under physical/biological constraints maps to their drug-discovery ML / simulation stack—not gene finding per se. |
| Matrices / LA? | Count→log-prob tables + sparse log-DP (not genome-scale SVD). Name semirings, dwell-time criticism of geometric `A`, Markov-5 backoff, CNN affine calibration. Rehearse Q1–Q3 in `LINEAR_ALGEBRA_METHODS.md`. |
| What would you build next? | Structure-objective train-fit of gene-start penalty; duration ablation tables under `experiments/version_5/duration/`; dwell diagnostic already in-repo. Defer Baum–Welch / fungi_diverse headline until structure metrics move. |

## Cover-letter bridge (2–3 sentences)

I like problems where algorithms and domain constraints meet: hard biology rules
in the state machine, learned scores where sequence context matters, and
evaluation that admits what the model still gets wrong. That is the same shape
of work I want on DESRES’s CS/ML team—building reliable scientific software and
ML systems that have to be right under physical constraints, not just on a
leaderboard.
