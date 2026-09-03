#!/usr/bin/env python3
"""Multi-task heads + segment CRF / soft-Viterbi fine-tune (Phase 3)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src" / "model"))

from emissions_nn.backbones import one_hot_encode_windows
from emissions_nn.models import MultiTaskEmissionModel
from emissions_nn.segment_crf import LABEL_TO_IDX, NUM_LABELS, EmissionToCRFUnary, SegmentCRF

OUT = REPO / "experiments" / "desres_v2" / "segment_crf"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(11)
    bases = np.array(list("ACGT"))
    n = 128
    windows = ["".join(bases[rng.integers(0, 4, size=121)].tolist()) for _ in range(n)]
    x = one_hot_encode_windows(windows)
    y_donor = torch.tensor([[1.0] if w[60:62] == "GT" else [0.0] for w in windows])
    y_acc = torch.tensor([[1.0] if w[59:61] == "AG" else [0.0] for w in windows])
    y_start = torch.tensor([[1.0] if w[60:63] == "ATG" else [0.0] for w in windows])

    model = MultiTaskEmissionModel(backbone="bilstm", hidden=64)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    bce = nn.BCEWithLogitsLoss()

    # Emission-only CE baseline
    ce_losses = []
    for step in range(50):
        out = model(x)
        loss = bce(out["donor"], y_donor) + bce(out["acceptor"], y_acc) + bce(out["start"], y_start)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 10 == 0:
            ce_losses.append(float(loss.item()))

    # Segment CRF fine-tune on short T sequences derived from features
    T = 16
    feats = model.encode(x[:32]).detach()
    unary = EmissionToCRFUnary(feats.size(-1), NUM_LABELS)
    crf = SegmentCRF()
    # Synthetic phase-aware tags: intergenic -> start -> coding -> donor -> intron -> acceptor -> coding -> stop
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
    tags = torch.tensor([pattern[t % len(pattern)] for t in range(T)], dtype=torch.long).unsqueeze(0).expand(32, -1)

    opt2 = torch.optim.Adam(
        list(model.parameters()) + list(unary.parameters()) + list(crf.parameters()),
        lr=5e-4,
    )
    crf_losses = []
    for step in range(40):
        feat = model.encode(x[:32])
        em = unary(feat).unsqueeze(1).expand(-1, T, -1)
        # blend CE + CRF
        out = model(x[:32])
        ce = bce(out["donor"], y_donor[:32]) + bce(out["acceptor"], y_acc[:32]) + bce(out["start"], y_start[:32])
        nll = crf.nll(em, tags)
        loss = ce + 0.5 * nll
        opt2.zero_grad()
        loss.backward()
        opt2.step()
        if step % 10 == 0:
            crf_losses.append({"step": step, "ce": float(ce.item()), "crf_nll": float(nll.item())})

    with torch.no_grad():
        post = crf.soft_viterbi(unary(model.encode(x[:8])).unsqueeze(1).expand(-1, T, -1))
        # Boundary F1 proxy: donor positions correctly peaked
        donor_t = 3  # pattern index
        pred_donor = post[:, donor_t, LABEL_TO_IDX["donor"]]
        boundary_score = float(pred_donor.mean().item())

    payload = {
        "emission_only_ce_curve": ce_losses,
        "crf_finetune_curve": crf_losses,
        "soft_viterbi_donor_mass": boundary_score,
        "labels": list(LABEL_TO_IDX.keys()),
        "comparison": {
            "emission_only_final_ce": ce_losses[-1] if ce_losses else None,
            "crf_final_nll": crf_losses[-1]["crf_nll"] if crf_losses else None,
        },
    }
    torch.save(
        {"multitask": model.state_dict(), "unary": unary.state_dict(), "crf": crf.state_dict()},
        OUT / "segment_crf.pt",
    )
    (OUT / "ablation.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["comparison"], indent=2))


if __name__ == "__main__":
    main()
