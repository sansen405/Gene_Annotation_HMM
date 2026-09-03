#!/usr/bin/env python3
"""Deep multi-task + segment CRF / soft-Viterbi ablation on real-ish mini-genes."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src" / "model"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from emissions_nn.backbones import one_hot_encode_windows
from emissions_nn.models import MultiTaskEmissionModel
from emissions_nn.segment_crf import LABEL_TO_IDX, NUM_LABELS, EmissionToCRFUnary, SegmentCRF
from run_calibration_deep import read_fasta, splice_sites_from_gff, window_at

OUT = REPO / "experiments" / "desres_v2" / "segment_crf"


def build_dataset(n_pos: int = 3000, n_neg: int = 6000):
    windows: list[str] = []
    y_d: list[float] = []
    y_a: list[float] = []
    y_s: list[float] = []
    rng = np.random.default_rng(3)
    for sp in ("s_pombe", "s_japonicus"):
        fasta = REPO / "genome_data" / "fission_yeasts" / sp / "train" / f"{sp}_train.fna"
        gff = REPO / "genome_data" / "fission_yeasts" / sp / "train" / f"{sp}_train.gff"
        seq = read_fasta(fasta)
        donors, acceptors = splice_sites_from_gff(gff)
        donor0 = {p - 1 for p in donors}
        acc0 = {p - 1 for p in acceptors}
        # starts: ATG not in splice sets (weak)
        starts = [i for i in range(len(seq) - 2) if seq[i : i + 3] == "ATG"]
        pos_d = list(donor0)
        pos_a = list(acc0)
        if len(pos_d) > n_pos // 2:
            pos_d = list(rng.choice(pos_d, size=n_pos // 2, replace=False))
        if len(pos_a) > n_pos // 2:
            pos_a = list(rng.choice(pos_a, size=n_pos // 2, replace=False))
        for i in pos_d:
            windows.append(window_at(seq, i))
            y_d.append(1.0)
            y_a.append(0.0)
            y_s.append(0.0)
        for i in pos_a:
            windows.append(window_at(seq, i))
            y_d.append(0.0)
            y_a.append(1.0)
            y_s.append(0.0)
        if len(starts) > 500:
            starts = list(rng.choice(starts, size=500, replace=False))
        for i in starts:
            windows.append(window_at(seq, i))
            y_d.append(0.0)
            y_a.append(0.0)
            y_s.append(1.0)
        negs = [i for i in range(1, len(seq) - 1) if seq[i : i + 2] == "GT" and i not in donor0]
        if len(negs) > n_neg // 2:
            negs = list(rng.choice(negs, size=n_neg // 2, replace=False))
        for i in negs:
            windows.append(window_at(seq, i))
            y_d.append(0.0)
            y_a.append(0.0)
            y_s.append(0.0)
    return (
        windows,
        torch.tensor(y_d).view(-1, 1),
        torch.tensor(y_a).view(-1, 1),
        torch.tensor(y_s).view(-1, 1),
    )


def donor_f1(logits: torch.Tensor, y: torch.Tensor) -> float:
    pred = (torch.sigmoid(logits).view(-1) >= 0.5).float()
    true = y.view(-1)
    tp = ((pred == 1) & (true == 1)).sum().item()
    fp = ((pred == 1) & (true == 0)).sum().item()
    fn = ((pred == 0) & (true == 1)).sum().item()
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    return float(2 * p * r / max(p + r, 1e-8))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    windows, y_d, y_a, y_s = build_dataset()
    x = one_hot_encode_windows(windows)
    print(f"n={len(windows)}")

    # Emission-only
    model_ce = MultiTaskEmissionModel(backbone="bilstm", hidden=128)
    opt = torch.optim.Adam(model_ce.parameters(), lr=1e-3)
    bce = nn.BCEWithLogitsLoss()
    for step in range(120):
        out = model_ce(x)
        loss = bce(out["donor"], y_d) + bce(out["acceptor"], y_a) + bce(out["start"], y_s)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        out = model_ce(x)
        ce_f1 = donor_f1(out["donor"], y_d)

    # CRF fine-tune
    model = MultiTaskEmissionModel(backbone="bilstm", hidden=128)
    model.load_state_dict(model_ce.state_dict())
    unary = EmissionToCRFUnary(model.encoder.out_dim, NUM_LABELS)
    crf = SegmentCRF()
    T = 16
    pattern = [
        LABEL_TO_IDX["intergenic"],
        LABEL_TO_IDX["start"],
        LABEL_TO_IDX["coding"],
        LABEL_TO_IDX["donor"],
        LABEL_TO_IDX["intron"],
        LABEL_TO_IDX["acceptor"],
        LABEL_TO_IDX["coding"],
        LABEL_TO_IDX["stop"],
    ]
    # Use a subset for CRF (memory)
    n_crf = min(256, len(windows))
    tags = torch.tensor([pattern[t % len(pattern)] for t in range(T)], dtype=torch.long).unsqueeze(0).expand(n_crf, -1)
    # Mark donor timestep with donor labels where y_d==1
    for b in range(n_crf):
        if y_d[b, 0] > 0.5:
            tags[b, 3] = LABEL_TO_IDX["donor"]
        if y_a[b, 0] > 0.5:
            tags[b, 5] = LABEL_TO_IDX["acceptor"]
        if y_s[b, 0] > 0.5:
            tags[b, 1] = LABEL_TO_IDX["start"]

    opt2 = torch.optim.Adam(list(model.parameters()) + list(unary.parameters()) + list(crf.parameters()), lr=5e-4)
    last_nll = 0.0
    for step in range(80):
        out = model(x[:n_crf])
        ce = bce(out["donor"], y_d[:n_crf]) + bce(out["acceptor"], y_a[:n_crf]) + bce(out["start"], y_s[:n_crf])
        em = unary(model.encode(x[:n_crf])).unsqueeze(1).expand(-1, T, -1)
        nll = crf.nll(em, tags)
        loss = ce + 0.25 * nll
        opt2.zero_grad()
        loss.backward()
        opt2.step()
        last_nll = float(nll.item())

    with torch.no_grad():
        out = model(x)
        crf_f1 = donor_f1(out["donor"], y_d)
        post = crf.soft_viterbi(unary(model.encode(x[:n_crf])).unsqueeze(1).expand(-1, T, -1))
        donor_mass = float(post[:, 3, LABEL_TO_IDX["donor"]].mean().item())

    payload = {
        "n_windows": len(windows),
        "emission_only_boundary_f1": ce_f1,
        "crf_boundary_f1": crf_f1,
        "crf_nll": last_nll,
        "soft_viterbi_donor_mass": donor_mass,
        "crf_improves_donor_f1": crf_f1 >= ce_f1,
    }
    torch.save({"model": model.state_dict(), "unary": unary.state_dict(), "crf": crf.state_dict()}, OUT / "segment_crf_deep.pt")
    (OUT / "ablation_deep.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
