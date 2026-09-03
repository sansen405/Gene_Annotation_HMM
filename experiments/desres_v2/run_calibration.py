#!/usr/bin/env python3
"""Calibration plots + ECE table for DESRES V2 Phase 1."""

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
from emissions_nn.calibration import expected_calibration_error, logits_to_confidence, mc_dropout_predict
from emissions_nn.models import SpliceEmissionModel
from emissions_nn.score_diagnostics import write_mc_dropout_diagnostics

OUT = REPO / "experiments" / "desres_v2" / "calibration"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    bases = np.array(list("ACGT"))
    windows = ["".join(bases[rng.integers(0, 4, size=121)].tolist()) for _ in range(400)]
    # Label: donor if GT at center
    labels = np.array([1.0 if w[60:62] == "GT" else 0.0 for w in windows])
    x = one_hot_encode_windows(windows)
    y = torch.tensor(labels).view(-1, 1)

    model = SpliceEmissionModel(backbone="dilated_cnn")
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()
    model.train()
    for _ in range(40):
        logits = model(x)[:, :1]
        loss = loss_fn(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        logits = model(x)[:, 0]
        conf = logits_to_confidence(logits).numpy()
    correct = ((conf >= 0.5) == (labels >= 0.5)).astype(np.float64)
    diag = expected_calibration_error(conf, correct, n_bins=10)

    mean, var = mc_dropout_predict(model, x, n_samples=12)
    positions = list(range(len(windows)))
    write_mc_dropout_diagnostics(
        model,
        windows[:64],
        positions[:64],
        OUT / "mc_dropout_diagnostics.tsv",
        n_samples=10,
    )

    table = {
        "ece": diag.ece,
        "n": int(len(conf)),
        "accuracy": float(correct.mean()),
        "mean_confidence": float(conf.mean()),
        "mc_mean_var": float(var.mean().item()),
        "tasks": ["donor_smoke"],
    }
    (OUT / "ece_table.json").write_text(json.dumps(table, indent=2) + "\n")
    lines = ["bin\tconfidence\taccuracy\tcount\tabs_gap"]
    for i, (c, a, n) in enumerate(zip(diag.bin_confidence, diag.bin_accuracy, diag.bin_counts)):
        lines.append(f"{i}\t{c:.6f}\t{a:.6f}\t{n}\t{abs(a - c):.6f}")
    (OUT / "reliability_diagram.tsv").write_text("\n".join(lines) + "\n")

    # Always write SVG (no matplotlib dependency). Also try PNG if matplotlib exists.
    xs = [c for c, n in zip(diag.bin_confidence, diag.bin_counts) if n > 0]
    ys = [a for a, n in zip(diag.bin_accuracy, diag.bin_counts) if n > 0]
    w, h, pad = 420, 420, 50

    def xy(c: float, a: float) -> tuple[float, float]:
        return pad + c * (w - 2 * pad), h - pad - a * (h - 2 * pad)

    poly = " ".join(f"{xy(c, a)[0]:.1f},{xy(c, a)[1]:.1f}" for c, a in zip(xs, ys))
    x0, y0 = xy(0, 0)
    x1, y1 = xy(1, 1)
    circles = "\n".join(
        f'<circle cx="{xy(c, a)[0]:.1f}" cy="{xy(c, a)[1]:.1f}" r="4" fill="#1f4e79"/>'
        for c, a in zip(xs, ys)
    )
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}">
  <rect width="100%" height="100%" fill="white"/>
  <line x1="{x0:.1f}" y1="{y0:.1f}" x2="{x1:.1f}" y2="{y1:.1f}" stroke="#999" stroke-dasharray="4"/>
  <polyline fill="none" stroke="#1f4e79" stroke-width="2" points="{poly}"/>
  {circles}
  <text x="{pad}" y="24" font-family="Helvetica" font-size="14">Reliability diagram (ECE={diag.ece:.3f})</text>
  <text x="{w/2:.0f}" y="{h-12}" text-anchor="middle" font-family="Helvetica" font-size="12">confidence</text>
  <text x="16" y="{h/2:.0f}" transform="rotate(-90 16,{h/2:.0f})" font-family="Helvetica" font-size="12">accuracy</text>
</svg>
"""
    (OUT / "reliability_diagram.svg").write_text(svg)
    print(f"wrote {OUT / 'reliability_diagram.svg'}")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot([0, 1], [0, 1], "--", color="gray", label="perfect")
        ax.plot(xs, ys, "o-", label=f"ECE={diag.ece:.3f}")
        ax.set_xlabel("confidence")
        ax.set_ylabel("accuracy")
        ax.set_title("Reliability diagram (donor smoke)")
        ax.legend()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        fig.tight_layout()
        fig.savefig(OUT / "reliability_diagram.png", dpi=120)
        plt.close(fig)
        print(f"wrote {OUT / 'reliability_diagram.png'}")
    except Exception as exc:  # noqa: BLE001
        (OUT / "plot_skipped.txt").write_text(f"matplotlib unavailable: {exc}\n")

    print(json.dumps(table, indent=2))


if __name__ == "__main__":
    main()
