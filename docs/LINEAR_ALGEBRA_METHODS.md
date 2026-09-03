# Linear algebra methods for this gene HMM (DESRES prep)

Companion to [`DESRES_INTERVIEW_PACK.md`](DESRES_INTERVIEW_PACK.md). Use this when
interviewers probe matrices, complexity, or “what other LA would you apply?”
The goal is transferable scientific-computing maturity—not a LAPACK laundry list.

**One-line framing:** gene finding here is constrained inference on a sparse
state graph with log-space matrix recurrences, not a black-box sequence model.

---

## Current matrix inventory

| Object | Shape / structure | Representation | Where |
| --- | --- | --- | --- |
| Transition counts `C` | Dense `21×21` bigrams; START/END per chromosome island | `uint64_t` | `src/model/transition/Transition_Model.cpp` (`count_bigrams`) |
| Transition log-matrix `A` | Dense `21×21`; illegal edges `LOG_ZERO` (`-∞`) | `Log_Prob` + Laplace over **allowed** outs only | `Transition_Model::compute_log_probs` |
| Topology adjacency | Hard graph; emitting preds/succs ~avg degree 1.5, `E≈28` | `unordered_map` + `array<vector<State>>` | `src/topology/Topology.hpp` |
| Intergenic Markov-1 | `4×4` context→base | counts → log probs | `Emission_Model::count_markov1_*` / `compute_markov1_*` |
| Intron / exon Markov-5 | `1024×4` (5-mer contexts); 3 exon frames | counts → log probs | `Emission_Model` (`MARKOV5_CONTEXTS`) |
| CNN score vectors | Length `T` donor/acceptor/start log-odds | dense `vector` after sparse motif write | `Splice_CNN_Scores`, `Start_CNN_Scores` |
| Affinity calibration | `scale·score + bias` | 2 scalars per channel | `Emission_Model::set_*_cnn_calibration` |
| Duration PMF | `log P(L)` histogram (+1 smooth) | `vector<Log_Prob>` | `validation/full_genome_validation.cpp` |
| Viterbi | Streaming 2-row scores; `O(T·S)` backpointers | max-plus over sparse preds | `src/decoding/Viterbi.cpp` |
| Forward–Backward | Full `T×S` α/β; posteriors `α+β−ℓ` | log-sum-exp | `src/decoding/Forward_Backward.cpp` |

**Explicitly absent today:** eigendecomposition, SVD, dense SpMV libraries,
BLAS/LAPACK solves. That is fine—name the structure you *do* exploit.

```text
counts ──► log A  ──┐
counts ──► Markov E ─┼──► sparse log-DP ──► Viterbi path / FB posteriors
CNN ──► affine calib ┤
duration PMF ────────┘
```

---

## Whiteboard: same SpMV, two semirings

Draw one sparse adjacency (legal edges only). Each time step is a mat-vec;
only the “addition” changes.

### Max-plus (Viterbi)

\[
V_t(s) = \max_{p \in \mathrm{Pred}(s)}
  \bigl( V_{t-1}(p) + \log A_{ps} + \log e_s(x_t) \bigr)
\]

- “Multiply” = ordinary `+` on log-probs.
- “Add” = `max`.
- Path recovery = argmax backpointers (already streaming in `Viterbi.cpp`).

### Log-sum-exp (Forward)

\[
\alpha_t(s) = \log\sum_{p \in \mathrm{Pred}(s)}
  \exp\bigl( \alpha_{t-1}(p) + \log A_{ps} + \log e_s(x_t) \bigr)
\]

- Same sparse pattern; “add” = `logsumexp`.
- Backward is the transpose SpMV on successors (`emitting_successors`).

### Complexity line (say out loud)

- Dense naive: `O(T · S²)` with `S≈19` emitting → ~361.
- Sparse: `O(T · E)` with `E≈28` (`make bench-viterbi`).
- Memory still `O(T · S)` for backpointers / FB tables unless you chunk or
  stream further.

**Interview phrase:** “Illegal transitions are structural zeros in the linear
operator, not soft penalties I hope the optimizer learns.”

---

## Methods map (what else fits *this* HMM)

Each entry: idea → equation/intuition → where it plugs in → probe you should
own.

### 1. Semiring linear algebra (already implicit—name it)

- **Idea:** Viterbi and Forward are the same sparse recurrence in max-plus vs
  log-sum-exp.
- **Where:** `Viterbi.cpp`, `Forward_Backward.cpp`, `Topology.hpp`.
- **Probe:** “Is Viterbi just DP?” → “Yes, and it is also max-plus SpMV on the
  biology graph.”

### 2. Packed sparse / CSR transition + log-SpMV

- **Idea:** Store only `(from, to, log A)` for legal edges instead of dense
  `21×21` with `-∞`.
- **Where:** `Transition_Model` storage; predecessor loops already sparse.
- **Probe:** “Why not dense BLAS?” → “Operator is ~92% structural zero; sparse
  wins on flops and clarifies constraints.”

### 3. Stationary distribution / spectral diagnostics of `A`

- **Idea:** On an emitting submatrix (or intron self-loop / exon 3-cycle
  blocks), solve \(\pi^\top A = \pi^\top\) (Perron). Implied geometric mean
  dwell \(1/(1-A_{ii})\) vs empirical intron length histogram → justifies
  semi-Markov.
- **Where:** Offline tool next to `compute_log_probs` (not hot path).
- **Probe:** “Did you check the geometric model?” → “Spectrally / via dwell
  times—histogram disagrees in the tail, so we charge \(\log P(L)\) once at
  acceptor close.”

### 4. Interpolated / backoff Markov-5

- **Idea:** Rare 5-mers → high-variance `1024×4` rows. Interpolate order-5 with
  order-1 (Jelinek–Mercer / Witten–Bell) instead of flat Laplace alone.
- **Where:** `compute_markov5_log_probs` / `emission_log_prob`.
- **Probe:** “Order-5 always better?” → “Only with enough counts; backoff is
  the statistically honest version of V4.2.”

### 5. Low-rank factorization of context×base tables

- **Idea:** \(M \approx UV^\top\) (or shared context embedding + frame head) on
  intron/exon tables; fewer free params than `3×1024×4`.
- **Where:** `train_hmm_matrices.cpp` training; decode stays table lookup or
  fast `U[ctx]·V`.
- **Probe:** “Why not just a neural emission?” → “Low-rank keeps
  interpretability and a tiny interface to the same sparse DP.”

### 6. Baum–Welch (posterior outer-product accumulation)

- **Idea:** From FB, accumulate \(\xi_t(i,j)\) and emission counts from
  \(\gamma_t(i)\); re-normalize with existing Laplace.
- **Where:** `Forward_Backward` → existing count APIs.
- **Honesty:** Duration-aware FB matches Viterbi’s generative assumptions
  (shared `Intron_Duration` gates + length table) before unsupervised EM claims.

### 7. Parametric duration (NB / spline MLE)

- **Idea:** Replace +1 histogram with negative-binomial or spline on
  \(\log p(L)\); same `vector<Log_Prob>` interface.
- **Where:** `Intron_Duration.hpp` (`build_negative_binomial_intron_length_log_probs`)
  → Viterbi/FB/predict duration args; ablation
  `experiments/version_5/duration/`.
- **Probe:** “Why parametric?” → “Better tails than a capped histogram; still
  falsifiable against holdout length stats.”
- **Dwell diagnostic:** `experiments/version_5/duration/dwell_diagnostic.py`
  compares geometric mean from `A` self-loop vs empirical / histogram / NB.

### 8. Small linear / logistic CNN↔HMM calibration

- **Idea:** Fit `(scale, bias)` by least squares or logistic regression on
  train motif sites vs hard negatives (vs grid search).
- **Where:** Existing affine form in `Emission_Model::set_*_cnn_calibration`.
- **Probe:** “How do discriminative scores enter a generative HMM?” →
  “Calibrated log-odds at hard-gated motifs; body stays Markov.”

### 9. Block / Kronecker structure of codon phase

- **Idea:** `EXON_1→2→3→1` is a length-3 circulant; frame-tracked introns
  preserve phase. `A` is block-sparse with a 3-cycle factor—vectorized 3-wide
  updates are available if needed.
- **Where:** Narrative now; optional micro-opt in exon/donor loops later.
- **Probe:** “Any structure beyond sparsity?” → “Phase cycle is circulant;
  biology handed us the Kronecker/block pattern.”

### 10. What not to fake

- Do **not** claim genome-scale SVD/PCA as your core method unless you shipped
  it.
- Stronger line: “Dense eigendecomp of full-genome matrices is not the
  bottleneck—**structured sparse log-DP and regularized emission tables** are.”
- Transformers: “They can replace scoring; the frame-constrained sparse state
  machine is still the right place for hard biological constraints.”

---

## Talking points (DESRES AI-biology)

- **Constraints as structure:** topology is a sparse linear operator.
- **Log-space is numerical LA:** underflow is floating-point; `logsumexp` is
  the associative sum.
- **Spectra / dwell times as model criticism:** geometric vs semi-Markov is
  falsifiable.
- **Hybrid systems:** discriminative CNN features + generative Markov bodies +
  calibrated interface.
- **Eval honesty:** holdouts, Augustus/Helixer, ablations, FB↔Viterbi duration
  consistency.

---

## Rehearsal Q&A (practice out loud)

### Q1 — Spectra and dwell times

**They ask:** “Your transition matrix has self-loops on introns. Why add a
semi-Markov length model instead of trusting the geometric dwell from `A`?”

**You answer:**
1. Geometric dwell from \(A_{ii}\) implies \(P(L)\propto (1-p)^L\).
2. Empirical intron lengths (train histogram) are not geometric in the tail.
3. Offline check: stationary / mean dwell on the intron block vs histogram
   (spectral diagnostic)—mismatch is the scientific justification.
4. Implementation: drop self-loop transition cost while duration is active;
   charge \(\log P(L)\) once on intron→acceptor (`Viterbi.cpp`). Own the gap:
   FB duration must stay consistent before posterior claims match Viterbi.

### Q2 — Backoff / interpolated Markov-5

**They ask:** “V4.2 moved intron emissions from order-1 to order-5. When does
that hurt, and what would you do next?”

**You answer:**
1. Order-5 is a `1024×4` count matrix; rare contexts are high-variance even
   with Laplace \(\alpha\).
2. Ablation already showed holdout intron F1 lift when data support it
   (fission yeasts).
3. Next: interpolate \(P_5\) with \(P_1\) (Jelinek–Mercer / Witten–Bell) so
   rare 5-mers shrink toward the Markov-1 table you already train for
   intergenic/intron bodies.
4. Metric: same nucleotide intron F1 ablation with λ sweep—statistical
   hygiene, not a bigger network.

### Q3 — Hybrid CNN calibration as a small linear solve

**They ask:** “How do CNN splice scores interact with the generative HMM
without breaking the probabilistic story?”

**You answer:**
1. Hard gates: non-GT/AG/ATG → `LOG_ZERO` (structural impossibility).
2. At candidates: emission = `scale · cnn_log_odds + bias` (affine interface
   already in `Emission_Model`).
3. Prefer a **tiny train-only solve** (logistic / least squares on true sites
   vs hard negatives) over eval-label grid search—never present
   `--tune-cnn-calibration` on holdout labels as the operating point.
4. Body sequence still uses generative Markov tables; the CNN is a calibrated
   discriminative feature at sparse motif positions—classic scientific-ML
   hybrid, not end-to-end opacity.

---

## Highest-leverage follow-ons (not implemented in this doc pass)

Ship these later for “I implemented the math” signal with small scope:

1. **Offline spectral / stationary dwell diagnostic** for `A` vs empirical
   intron lengths (small Python/C++ tool + plot under `experiments/`).
2. **Interpolated Markov-5** (order-5 ↔ order-1) with a one-line validation
   ablation.

Defer Baum–Welch and low-rank factorization until fungi holdout baselines are
solid—larger surface area and easier to overclaim in an interview.
