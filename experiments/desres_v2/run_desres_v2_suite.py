#!/usr/bin/env python3
"""DESRES V2 master suite — runs all phase scripts and writes master ablation table."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "experiments" / "desres_v2"
PY = REPO / ".venv" / "bin" / "python"
if not PY.exists():
    PY = Path(sys.executable)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run(script: str) -> None:
    path = OUT / script
    log(f"running {script}")
    env = {"PYTHONPATH": str(REPO / "src" / "model"), **dict(**{k: v for k, v in __import__("os").environ.items()})}
    subprocess.check_call([str(PY), str(path)], cwd=str(REPO), env=env)


def phase0_inline() -> dict:
    """Keep backbone smoke in-process for speed."""
    sys.path.insert(0, str(REPO / "src" / "model"))
    from emissions_nn.backbones import BACKBONE_NAMES, one_hot_encode_windows
    from emissions_nn.models import SpliceEmissionModel, StartEmissionModel
    import numpy as np
    import torch

    rng = np.random.default_rng(0)
    bases = np.array(list("ACGT"))
    windows = ["".join(bases[rng.integers(0, 4, size=121)].tolist()) for _ in range(32)]
    x = one_hot_encode_windows(windows)
    rows = []
    for name in BACKBONE_NAMES:
        t0 = time.perf_counter()
        splice = SpliceEmissionModel(backbone=name).eval()
        start = StartEmissionModel(backbone=name).eval()
        with torch.no_grad():
            s_out = splice(x)
            st_out = start(x)
        elapsed = time.perf_counter() - t0
        n_params = sum(p.numel() for p in splice.parameters()) + sum(p.numel() for p in start.parameters())
        rows.append(
            {
                "backbone": name,
                "splice_out_shape": list(s_out.shape),
                "start_out_shape": list(st_out.shape),
                "params": int(n_params),
                "forward_s": round(elapsed, 4),
            }
        )
        log(f"  {name}: params={n_params:,} forward={elapsed:.3f}s")

    sys.path.insert(0, str(REPO / "src" / "model" / "cnn" / "splice"))
    from splice_cnn_network import SpliceCNN, build_splice_model

    assert isinstance(build_splice_model(backbone="dilated_cnn"), SpliceCNN)
    built_tf = build_splice_model(backbone="transformer")
    assert hasattr(built_tf, "backbone_name")

    out_dir = OUT / "backbone_ablation"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "smoke_forward.json").write_text(json.dumps(rows, indent=2) + "\n")
    (out_dir / "README.md").write_text(
        "# Backbone ablation\n\n"
        "Smoke: `smoke_forward.json`. Full holdout: `./run_backbone_ablation.sh` "
        "(after per-backbone score export) then `full_genome_validation`.\n"
    )
    (out_dir / "results_table.tsv").write_text(
        "backbone\tparams\tforward_s_smoke\texact_gene_f1\tintron_f1\tece_donor\tnotes\n"
        + "\n".join(f"{r['backbone']}\t{r['params']}\t{r['forward_s']}\t\t\t\tsmoke_only" for r in rows)
        + "\n"
    )
    return {"n_backbones": len(rows), "rows": rows}


def write_master(summary: dict) -> None:
    results_dir = OUT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    ece = None
    ece_path = OUT / "calibration" / "ece_table.json"
    if ece_path.exists():
        ece = json.loads(ece_path.read_text()).get("ece")

    tokens = None
    ssl_path = OUT / "ssl" / "scratch_vs_pretrained.json"
    if ssl_path.exists():
        tokens = json.loads(ssl_path.read_text()).get("tokens_supervised")

    crf_nll = None
    crf_path = OUT / "segment_crf" / "ablation.json"
    if crf_path.exists():
        crf_nll = json.loads(crf_path.read_text()).get("comparison", {}).get("crf_final_nll")

    coral_red = None
    coral_path = OUT / "coral" / "adaptation.json"
    if coral_path.exists():
        coral_red = json.loads(coral_path.read_text()).get("coral_reduction")

    lines = [
        "component\tmetric\tvalue\tnotes",
        f"backbone_smoke\tn_backbones\t{summary['phase0']['n_backbones']}\tDilatedCNN/BiLSTM/Transformer",
        f"calibration\tece\t{ece}\tholdout-candidate smoke",
        f"ssl\ttokens_supervised\t{int(tokens) if tokens is not None else ''}\twindowed MLM",
        f"segment_crf\tcrf_nll\t{crf_nll}\tsoft-Viterbi path",
        f"coral\tcoral_reduction\t{coral_red}\tfission→fungi feature align",
        "fission_holdout_V5\texact_gene_p\t0.8080\tvalidation/results/version_5",
        "fission_holdout_V5\tintron_f1\t0.8413\tvalidation/results/version_5",
        "fungi_zeroshot\texact_gene_p\t0.3898\ttransfer_summary.tsv",
        "fungi_indomain_plus\texact_gene_p\t0.4557\ttransfer_summary.tsv",
    ]
    (results_dir / "master_ablation.tsv").write_text("\n".join(lines) + "\n")
    (results_dir / "runtime_note.txt").write_text(
        "Neural smoke forwards: backbone_ablation/smoke_forward.json\n"
        "C++ HSMM: make bench-viterbi\n"
        "Laptop: windowed ≤~few M params; no foundation-model fine-tunes.\n"
    )
    summary_out = {
        "phase0_n_backbones": summary["phase0"]["n_backbones"],
        "phase1_ece": ece,
        "phase2_tokens": tokens,
        "phase3_crf_nll": crf_nll,
        "phase4_coral_reduction": coral_red,
    }
    (OUT / "suite_summary.json").write_text(json.dumps(summary_out, indent=2) + "\n")
    log(f"wrote {results_dir / 'master_ablation.tsv'}")


def write_docs() -> None:
    methods = REPO / "docs" / "DESRES_V2_METHODS.md"
    methods.write_text(
        """# DESRES V2 methods — structured biological sequence modeling

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
"""
    )

    pack = REPO / "docs" / "DESRES_INTERVIEW_PACK.md"
    text = pack.read_text() if pack.exists() else ""
    header = """# DESRES interview pack — structured sequence modeling

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

"""
    marker = "## 10-minute whiteboard narrative"
    if marker in text:
        rest = text.split(marker, 1)[1]
        pack.write_text(header + "## 10-minute whiteboard narrative" + rest)
    else:
        pack.write_text(header + text)

    readme = REPO / "README.md"
    readme_text = readme.read_text()
    if "DESRES V2" not in readme_text:
        insert = (
            "\n**DESRES V2 research platform:** pluggable neural emissions, calibration, "
            "SSL, segment CRF, and CORAL adaptation — see "
            "[`docs/DESRES_V2_METHODS.md`](docs/DESRES_V2_METHODS.md) and "
            "[`experiments/desres_v2/`](experiments/desres_v2/).\n"
        )
        parts = readme_text.split("\n\n", 2)
        if len(parts) >= 2:
            readme.write_text(
                parts[0] + "\n\n" + parts[1] + insert + ("\n\n" + parts[2] if len(parts) > 2 else "")
            )
        else:
            readme.write_text(readme_text + insert)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    log("Phase 0: backbone smoke")
    phase0 = phase0_inline()
    run("run_calibration.py")
    run("run_ssl_pretrain.py")
    run("run_segment_crf.py")
    run("run_coral.py")
    write_docs()
    write_master({"phase0": phase0})
    log("DESRES_V2_SUITE_DONE")


if __name__ == "__main__":
    main()
