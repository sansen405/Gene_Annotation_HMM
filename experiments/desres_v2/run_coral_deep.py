#!/usr/bin/env python3
"""CORAL-adapt fission DilatedCNN splice emissions toward fungi_diverse_span."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src" / "model"))
sys.path.insert(0, str(REPO / "src" / "model" / "cnn" / "splice"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from emissions_nn.backbones import one_hot_encode_windows
from emissions_nn.coral import coral_loss
from emissions_nn.pretrain import load_fasta_sequences
from run_calibration_deep import read_fasta, splice_sites_from_gff, window_at
from splice_cnn_network import SpliceCNN, build_splice_model

OUT = REPO / "experiments" / "desres_v2" / "coral"


def sample_windows(paths: list[Path], n: int, seed: int) -> list[str]:
    seqs = load_fasta_sequences(paths, max_bases=8_000_000) if paths else []
    rng = np.random.default_rng(seed)
    if not seqs:
        bases = np.array(list("ACGT"))
        return ["".join(bases[rng.integers(0, 4, size=121)].tolist()) for _ in range(n)]
    out = []
    while len(out) < n:
        seq = seqs[int(rng.integers(0, len(seqs)))]
        if len(seq) < 121:
            continue
        start = int(rng.integers(0, len(seq) - 121 + 1))
        out.append(seq[start : start + 121])
    return out


def features(model: SpliceCNN, x: torch.Tensor) -> torch.Tensor:
    d_lo, d_hi = model.donor_slice
    a_lo, a_hi = model.acceptor_slice
    d = model.donor_features(x[:, :, d_lo:d_hi]).flatten(1)
    a = model.acceptor_features(x[:, :, a_lo:a_hi]).flatten(1)
    return torch.cat([d, a], dim=1)


def supervised_batch(max_n: int = 4000):
    windows: list[str] = []
    labels: list[list[float]] = []
    rng = np.random.default_rng(5)
    for sp in ("s_pombe", "s_japonicus"):
        fasta = REPO / "genome_data" / "fission_yeasts" / sp / "train" / f"{sp}_train.fna"
        gff = REPO / "genome_data" / "fission_yeasts" / sp / "train" / f"{sp}_train.gff"
        seq = read_fasta(fasta)
        donors, acceptors = splice_sites_from_gff(gff)
        donor0 = {p - 1 for p in donors}
        acc0 = {p - 1 for p in acceptors}
        pos = list(donor0 | acc0)
        if len(pos) > max_n // 4:
            pos = list(rng.choice(pos, size=max_n // 4, replace=False))
        for i in pos:
            windows.append(window_at(seq, i))
            labels.append([1.0 if i in donor0 else 0.0, 1.0 if i in acc0 else 0.0])
        negs = [i for i in range(1, len(seq) - 1) if seq[i : i + 2] == "GT" and i not in donor0]
        if len(negs) > max_n // 4:
            negs = list(rng.choice(negs, size=max_n // 4, replace=False))
        for i in negs:
            windows.append(window_at(seq, i))
            labels.append([0.0, 0.0])
    return windows, torch.tensor(labels, dtype=torch.float32)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ckpt_path = REPO / "src" / "model" / "cnn" / "splice" / "trained_models" / "fission_yeasts_splice_cnn.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = build_splice_model(backbone=ckpt.get("backbone", "dilated_cnn"))
    model.load_state_dict(ckpt["state_dict"])
    assert isinstance(model, SpliceCNN)

    fission_fastas = [
        REPO / "genome_data" / "fission_yeasts" / sp / "train" / f"{sp}_train.fna"
        for sp in ("s_pombe", "s_japonicus", "s_octosporus", "s_cryophilus")
    ]
    fungi_fastas = sorted((REPO / "genome_data" / "fungi_diverse").rglob("*_train.fna"))[:10]
    src_u = sample_windows([p for p in fission_fastas if p.exists()], n=512, seed=1)
    tgt_u = sample_windows(fungi_fastas, n=512, seed=2)
    src_x = one_hot_encode_windows(src_u)
    tgt_x = one_hot_encode_windows(tgt_u)

    sup_w, sup_y = supervised_batch()
    sup_x = one_hot_encode_windows(sup_w)

    with torch.no_grad():
        coral0 = float(coral_loss(features(model, src_x), features(model, tgt_x)).item())

    # Adapt feature extractors; keep heads mostly stable with low LR
    params = list(model.donor_features.parameters()) + list(model.acceptor_features.parameters())
    opt = torch.optim.Adam(
        [
            {"params": params, "lr": 1e-4},
            {"params": list(model.donor_head.parameters()) + list(model.acceptor_head.parameters()), "lr": 3e-5},
        ]
    )
    bce = nn.BCEWithLogitsLoss()
    history = []
    model.train()
    for step in range(120):
        logits = model(sup_x)
        task = bce(logits, sup_y)
        align = coral_loss(features(model, src_x), features(model, tgt_x))
        loss = task + 10.0 * align
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 20 == 0:
            history.append({"step": step, "task": float(task.item()), "coral": float(align.item())})
            print(history[-1], flush=True)

    model.eval()
    with torch.no_grad():
        coral1 = float(coral_loss(features(model, src_x), features(model, tgt_x)).item())

    out_ckpt = {
        "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        "backbone": "dilated_cnn",
        "calibration": ckpt.get(
            "calibration",
            {"temperature": 1.0, "donor_prior_logit": 0.0, "acceptor_prior_logit": 0.0},
        ),
        "coral": {"initial": coral0, "final": coral1, "history": history},
    }
    torch.save(out_ckpt, OUT / "adapted_splice.pt")
    payload = {
        "coral_initial": coral0,
        "coral_final": coral1,
        "coral_reduction": coral0 - coral1,
        "history": history,
        "adapted_checkpoint": str(OUT / "adapted_splice.pt"),
        "protocol": "Score fungi holdouts with adapted_splice.pt; validate vs zeroshot/indomain.",
    }
    (OUT / "adaptation_deep.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
