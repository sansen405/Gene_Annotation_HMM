#!/usr/bin/env python3
"""Write per-base donor/acceptor CNN scores for one FASTA (HMM input)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from train_splice_cnn_scores import (  # noqa: E402
    Calibration,
    detect_device,
    log,
    read_fasta,
    write_sparse_scores,
)

DEFAULT_RADIUS = 60
DEFAULT_BATCH_SIZE = 8192


def load_checkpoint(model_path: Path, radius: int):
    import torch
    from splice_cnn_network import SpliceCNN

    checkpoint = torch.load(model_path, map_location="cpu")
    model = SpliceCNN(window_size=radius * 2 + 1)

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
        calibration = Calibration(**checkpoint["calibration"])
        return model, calibration

    model.load_state_dict(checkpoint)
    raise ValueError(
        f"Checkpoint at {model_path} is missing calibration metadata. "
        "Retrain or replace the model with a V3 checkpoint."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Score splice sites for one FASTA.")
    parser.add_argument("--fasta", required=True, type=Path)
    parser.add_argument(
        "--model",
        default=SCRIPT_DIR / "trained_models" / "fission_yeasts_splice_cnn.pt",
        type=Path,
    )
    parser.add_argument("--scores-out", required=True, type=Path)
    parser.add_argument("--radius", default=DEFAULT_RADIUS, type=int)
    parser.add_argument("--batch-size", default=DEFAULT_BATCH_SIZE, type=int)
    parser.add_argument(
        "--dense",
        action="store_true",
        help="Score every base (slow). Default scores GT/AG candidates only.",
    )
    args = parser.parse_args()

    if not args.model.exists():
        raise FileNotFoundError(
            f"CNN checkpoint not found: {args.model}. "
            "Run: python3 src/model/training_pipeline/train_cached_model.py --skip-compile"
        )

    log(f"loading CNN checkpoint: {args.model}")
    model, calibration = load_checkpoint(args.model, args.radius)
    device = detect_device()
    model = model.to(device)
    model.eval()
    log(
        f"loaded calibration: temperature={calibration.temperature:.4f} "
        f"donor_prior_logit={calibration.donor_prior_logit:.4f} "
        f"acceptor_prior_logit={calibration.acceptor_prior_logit:.4f}"
    )

    dataset = read_fasta(args.fasta)
    args.scores_out.parent.mkdir(parents=True, exist_ok=True)
    if args.dense:
        from train_splice_cnn_scores import write_scores

        write_scores(
            model,
            dataset,
            args.radius,
            args.scores_out,
            args.batch_size,
            device,
            calibration,
        )
    else:
        write_sparse_scores(
            model,
            dataset,
            args.radius,
            args.scores_out,
            args.batch_size,
            device,
            calibration,
        )
    log(f"scores ready: {args.scores_out}")


if __name__ == "__main__":
    main()
