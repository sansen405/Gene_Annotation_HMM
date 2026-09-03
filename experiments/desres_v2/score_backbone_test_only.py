#!/usr/bin/env python3
"""Train (if needed) a splice backbone and score TEST (+/−) strands only."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src" / "model" / "cnn" / "splice"))
sys.path.insert(0, str(REPO / "src" / "model"))

# Import trainer helpers from the splice training module.
import train_splice_cnn_scores as train  # noqa: E402


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True, choices=["dilated_cnn", "bilstm", "transformer"])
    ap.add_argument("--model-out", type=Path, required=True)
    ap.add_argument("--train-fasta", type=Path, nargs="+", required=True)
    ap.add_argument("--train-gff", type=Path, nargs="+", required=True)
    ap.add_argument("--test-fasta", type=Path, nargs="+", required=True)
    ap.add_argument("--test-scores-out", type=Path, nargs="+", required=True)
    ap.add_argument("--test-scores-minus-out", type=Path, nargs="+", required=True)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--radius", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--score-batch-size", type=int, default=8192)
    ap.add_argument("--negatives-per-positive", type=int, default=3)
    args = ap.parse_args()

    import torch
    from splice_cnn_network import build_splice_model, one_hot_encode_windows
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    # Bind into trainer module namespace expected by train_model helpers
    train.torch = torch
    train.nn = nn
    train.DataLoader = DataLoader
    train.TensorDataset = TensorDataset
    train.SpliceCNN = None  # type: ignore
    train.one_hot_encode_windows = one_hot_encode_windows
    train.build_splice_model = build_splice_model

    device = train.detect_device()
    log(f"device: {device} backbone={args.backbone} epochs={args.epochs} neg={args.negatives_per_positive}")

    train_datasets = [train.read_fasta(p) for p in args.train_fasta]
    test_datasets = [train.read_fasta(p) for p in args.test_fasta]
    model = build_splice_model(window_size=args.radius * 2 + 1, backbone=args.backbone)

    if args.model_out.exists():
        log(f"loading checkpoint {args.model_out}")
        ckpt = torch.load(args.model_out, map_location="cpu")
        ckpt_bb = ckpt.get("backbone", "dilated_cnn")
        if ckpt_bb != args.backbone:
            log(f"rebuild for checkpoint backbone={ckpt_bb}")
            model = build_splice_model(window_size=args.radius * 2 + 1, backbone=ckpt_bb)
            args.backbone = ckpt_bb
        model.load_state_dict(ckpt["state_dict"])
        model = model.to(device)
        calibration = train.Calibration(**ckpt["calibration"])
    else:
        log("training from scratch on train GFF labels")
        input_batches = []
        label_batches = []
        for dataset, gff_path in zip(train_datasets, args.train_gff):
            donors, acceptors = train.splice_sites_from_gff(
                gff_path, dataset.offsets, min_intron_bp=20, require_3n_cds=True
            )
            inputs, labels = train.sample_training_examples(
                dataset, donors, acceptors, args.radius, args.negatives_per_positive
            )
            if len(labels) > 0:
                input_batches.append(inputs)
                label_batches.append(labels)
        if not input_batches:
            raise SystemExit("No training examples found")
        inputs = torch.cat(input_batches)
        labels = torch.cat(label_batches)
        calibration = train.train_model(model, inputs, labels, args.epochs, args.batch_size, device)
        args.model_out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                "backbone": args.backbone,
                "calibration": {
                    "temperature": calibration.temperature,
                    "donor_prior_logit": calibration.donor_prior_logit,
                    "acceptor_prior_logit": calibration.acceptor_prior_logit,
                },
            },
            args.model_out,
        )
        log(f"saved {args.model_out}")

    assert len(args.test_fasta) == len(args.test_scores_out) == len(args.test_scores_minus_out)
    for dataset, out_plus, out_minus in zip(test_datasets, args.test_scores_out, args.test_scores_minus_out):
        train.write_sparse_scores(
            model, dataset, args.radius, out_plus, args.score_batch_size, device, calibration, reverse=False
        )
        train.write_sparse_scores(
            model, dataset, args.radius, out_minus, args.score_batch_size, device, calibration, reverse=True
        )
    log("finished test-only scoring")


if __name__ == "__main__":
    main()
