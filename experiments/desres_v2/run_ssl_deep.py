#!/usr/bin/env python3
"""Deep windowed MLM pretrain + scratch vs pretrained finetune on real splice labels."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src" / "model"))
sys.path.insert(0, str(REPO / "src" / "model" / "cnn" / "splice"))

from emissions_nn.backbones import one_hot_encode_windows
from emissions_nn.models import MultiTaskEmissionModel
from emissions_nn.pretrain import NucleotideMLM, load_fasta_sequences, train_mlm

OUT = REPO / "experiments" / "desres_v2" / "ssl"


def load_supervised_splice(max_pos: int = 4000, max_neg: int = 8000, radius: int = 60):
    """Build donor labels from fission train GFFs (sparse)."""
    # reuse calibration helper logic inline
    from run_calibration_deep import read_fasta, splice_sites_from_gff, window_at

    windows: list[str] = []
    labels: list[list[float]] = []
    rng = np.random.default_rng(1)
    for sp in ("s_pombe", "s_japonicus", "s_octosporus", "s_cryophilus"):
        fasta = REPO / "genome_data" / "fission_yeasts" / sp / "train" / f"{sp}_train.fna"
        gff = REPO / "genome_data" / "fission_yeasts" / sp / "train" / f"{sp}_train.gff"
        if not fasta.exists():
            continue
        seq = read_fasta(fasta)
        donors, acceptors = splice_sites_from_gff(gff)
        donor0 = {p - 1 for p in donors}
        acc0 = {p - 1 for p in acceptors}
        pos_d = [i for i in donor0 if 0 <= i < len(seq)]
        pos_a = [i for i in acc0 if 0 <= i < len(seq)]
        if len(pos_d) > max_pos // 4:
            pos_d = list(rng.choice(pos_d, size=max_pos // 4, replace=False))
        if len(pos_a) > max_pos // 4:
            pos_a = list(rng.choice(pos_a, size=max_pos // 4, replace=False))
        for i in pos_d:
            windows.append(window_at(seq, i, radius))
            labels.append([1.0, 0.0, 0.0])
        for i in pos_a:
            windows.append(window_at(seq, i, radius))
            labels.append([0.0, 1.0, 0.0])
        # negatives at random GT/AG
        negs = []
        for i in range(1, len(seq) - 1):
            if seq[i : i + 2] == "GT" and i not in donor0:
                negs.append(i)
            if seq[i - 1 : i + 1] == "AG" and i not in acc0:
                negs.append(i)
        if len(negs) > max_neg // 4:
            negs = list(rng.choice(negs, size=max_neg // 4, replace=False))
        for i in negs:
            windows.append(window_at(seq, i, radius))
            labels.append([0.0, 0.0, 0.0])
    y = torch.tensor(labels, dtype=torch.float32)
    return windows, y


def finetune(init_sd: dict | None, windows: list[str], y: torch.Tensor, steps: int = 80) -> dict:
    model = MultiTaskEmissionModel(backbone="transformer", hidden=128)
    if init_sd:
        own = model.state_dict()
        mapped = {k: v for k, v in init_sd.items() if k in own and own[k].shape == v.shape}
        own.update(mapped)
        model.load_state_dict(own)
    x = one_hot_encode_windows(windows)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    bce = nn.BCEWithLogitsLoss()
    model.train()
    last = 0.0
    for _ in range(steps):
        out = model(x)
        loss = (
            bce(out["donor"], y[:, 0:1])
            + bce(out["acceptor"], y[:, 1:2])
            + bce(out["start"], y[:, 2:3])
        )
        opt.zero_grad()
        loss.backward()
        opt.step()
        last = float(loss.item())
    model.eval()
    with torch.no_grad():
        out = model(x)
        pred_d = (torch.sigmoid(out["donor"]).view(-1) >= 0.5).float()
        true_d = y[:, 0]
        tp = ((pred_d == 1) & (true_d == 1)).sum().item()
        fp = ((pred_d == 1) & (true_d == 0)).sum().item()
        fn = ((pred_d == 0) & (true_d == 1)).sum().item()
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-8)
    return {"bce": last, "donor_f1": float(f1), "donor_precision": float(prec), "donor_recall": float(rec)}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    max_bases = int(os.environ.get("SSL_MAX_BASES", "40000000"))
    samples = int(os.environ.get("SSL_SAMPLES", "96"))
    epochs = int(os.environ.get("SSL_EPOCHS", "2"))

    fasta_paths = []
    for sp in ("s_pombe", "s_japonicus", "s_octosporus", "s_cryophilus"):
        p = REPO / "genome_data" / "fission_yeasts" / sp / "train" / f"{sp}_train.fna"
        if p.exists():
            fasta_paths.append(p)
    fungi_root = REPO / "genome_data" / "fungi_diverse"
    if fungi_root.exists():
        for p in sorted(fungi_root.rglob("*_train.fna"))[:12]:
            fasta_paths.append(p)

    sequences = load_fasta_sequences(fasta_paths, max_bases=max_bases)
    # chunk for denser sampling
    chunked: list[str] = []
    chunk = 80_000
    for seq in sequences:
        if len(seq) <= chunk:
            chunked.append(seq)
            continue
        for i in range(0, len(seq) - chunk + 1, chunk):
            chunked.append(seq[i : i + chunk])
            if len(chunked) >= 400:
                break
        if len(chunked) >= 400:
            break
    sequences = chunked or sequences
    print(f"ssl sequences={len(sequences)} max_bases={max_bases} samples_per_seq={samples} epochs={epochs}")

    mlm = NucleotideMLM(backbone="transformer", hidden=128)
    stats = train_mlm(
        mlm,
        sequences,
        epochs=epochs,
        batch_size=64,
        window=121,
        samples_per_seq=samples,
        device="cpu",
    )
    torch.save({"state_dict": mlm.state_dict(), "stats": stats, "backbone": "transformer"}, OUT / "mlm_pretrained_deep.pt")

    windows, y = load_supervised_splice()
    print(f"supervised windows={len(windows)}")
    scratch = finetune(None, windows, y)
    pretrained = finetune(mlm.state_dict(), windows, y)

    payload = {
        **stats,
        "max_bases": max_bases,
        "samples_per_seq": samples,
        "epochs": epochs,
        "n_fasta": len(fasta_paths),
        "scratch": scratch,
        "pretrained": pretrained,
        "pretrained_better_donor_f1": pretrained["donor_f1"] >= scratch["donor_f1"],
        "claim": (
            f"pretrained encoder on {int(stats['tokens_supervised'])} masked nucleotides "
            f"(windowed MLM; {int(stats['n_windows'])} windows) before supervised fine-tuning"
        ),
    }
    (OUT / "scratch_vs_pretrained_deep.json").write_text(json.dumps(payload, indent=2) + "\n")
    (OUT / "token_count_deep.txt").write_text(payload["claim"] + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    # allow importing sibling deep calibration helpers
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
