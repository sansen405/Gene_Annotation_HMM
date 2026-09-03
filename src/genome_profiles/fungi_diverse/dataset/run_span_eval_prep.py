#!/usr/bin/env python3
"""Score CNN emissions for fungi_diverse_span TEST holdouts only, then fit HMM matrices.

Faster than the full cached pipeline (skips multi-100Mbp train-score TSVs).
Uses existing fungi_diverse_splice_cnn.pt and fission start CNN checkpoint.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
PROFILE = REPO / "src/genome_profiles/fungi_diverse/fungi_diverse_span.json"
PY = REPO / ".venv/bin/python"
SPLICE = REPO / "src/model/cnn/splice/train_splice_cnn_scores.py"
START = REPO / "src/model/cnn/start/train_start_cnn_scores.py"
HMM_BIN = REPO / "build/train_hmm_matrices"


def main() -> None:
    profile = json.loads(PROFILE.read_text())
    species = profile["dataset"]["species"]

    # Build a temporary profile that only lists test score outputs so the
    # existing trainers only numerify/score holdout chromosomes.
    tmp = PROFILE.with_name("fungi_diverse_span_eval.json")
    eval_profile = json.loads(PROFILE.read_text())
    # Keep one tiny train fasta entry so trainers' required train fields resolve:
    # point train paths at the same test files (scores unused for HMM fit).
    for block in ("splice_cnn", "start_cnn"):
        eval_profile[block]["train_scores"] = [
            p.replace("/test/", "/train/").replace("_test_", "_train_")
            for p in eval_profile[block]["test_scores"]
        ]
        eval_profile[block]["train_scores_minus"] = [
            p.replace("/test/", "/train/").replace("_test_", "_train_")
            for p in eval_profile[block].get("test_scores_minus", [])
        ]
    # Also need train fasta/gff for HMM; keep real train paths on species entries.
    tmp.write_text(json.dumps(eval_profile, indent=4) + "\n")

    for script, label in ((SPLICE, "splice"), (START, "start")):
        print(f"=== scoring {label} CNN on span holdouts ===", flush=True)
        subprocess.run(
            [
                str(PY),
                str(script),
                "--profile",
                str(tmp.relative_to(REPO)),
                "--score-batch-size",
                "8192",
                "--sparse-scores",
            ],
            cwd=REPO,
            check=True,
        )

    if not HMM_BIN.exists():
        subprocess.run(["make", "train-matrices"], cwd=REPO, check=True)

    out_dir = REPO / "src/model/training_pipeline/trained_models/fungi_diverse_span"
    out_dir.mkdir(parents=True, exist_ok=True)
    print("=== fitting HMM matrices on fungi_diverse_span train splits ===", flush=True)
    subprocess.run(
        [
            str(HMM_BIN),
            "--profile",
            str(PROFILE),
            "--out-dir",
            str(out_dir),
        ],
        cwd=REPO,
        check=True,
    )
    print("done", flush=True)


if __name__ == "__main__":
    main()
