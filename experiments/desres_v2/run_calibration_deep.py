#!/usr/bin/env python3
"""Deep calibration: ECE + reliability + MC dropout on real fission holdout candidates."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src" / "model"))
sys.path.insert(0, str(REPO / "src" / "model" / "cnn" / "splice"))

from emissions_nn.backbones import one_hot_encode_windows
from emissions_nn.calibration import expected_calibration_error, logits_to_confidence, mc_dropout_predict
from emissions_nn.score_diagnostics import write_mc_dropout_diagnostics
from splice_cnn_network import build_splice_model

OUT = REPO / "experiments" / "desres_v2" / "calibration"


def read_fasta(path: Path) -> str:
    seqs = []
    with path.open() as f:
        for line in f:
            if line.startswith(">"):
                continue
            seqs.append(line.strip().upper())
    return "".join(seqs)


def splice_sites_from_gff(gff: Path) -> tuple[set[int], set[int]]:
    donors: set[int] = set()
    acceptors: set[int] = set()
    # GFF is 1-based inclusive; collect CDS exon edges as approx splice sites
    cds: dict[tuple[str, str], list[tuple[int, int]]] = {}
    with gff.open() as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 8 or parts[2].lower() != "cds":
                continue
            chrom, start, end, strand = parts[0], int(parts[3]), int(parts[4]), parts[6]
            key = (chrom, strand)
            cds.setdefault(key, []).append((start, end))
    for (chrom, strand), intervals in cds.items():
        intervals = sorted(intervals)
        if strand == "+":
            for i in range(len(intervals) - 1):
                donors.add(intervals[i][1])  # last coding base before intron
                acceptors.add(intervals[i + 1][0])  # first coding base after intron
        else:
            for i in range(len(intervals) - 1):
                # minus strand: order by genomic start still sorted ascending
                donors.add(intervals[i + 1][0])
                acceptors.add(intervals[i][1])
    return donors, acceptors


def window_at(seq: str, pos0: int, radius: int = 60) -> str:
    # pos0 0-based
    lo = pos0 - radius
    hi = pos0 + radius + 1
    if lo < 0 or hi > len(seq):
        pad_left = max(0, -lo)
        pad_right = max(0, hi - len(seq))
        core = seq[max(0, lo) : min(len(seq), hi)]
        return "N" * pad_left + core + "N" * pad_right
    return seq[lo:hi]


def collect_candidates(max_per_species: int = 8000) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Return windows, donor_label, acceptor_label for GT/AG candidates on test sets."""
    windows: list[str] = []
    y_d: list[float] = []
    y_a: list[float] = []
    rng = np.random.default_rng(0)
    for sp in ("s_pombe", "s_japonicus", "s_octosporus", "s_cryophilus"):
        fasta = REPO / "genome_data" / "fission_yeasts" / sp / "test" / f"{sp}_test.fna"
        gff = REPO / "genome_data" / "fission_yeasts" / sp / "test" / f"{sp}_test.gff"
        if not fasta.exists() or not gff.exists():
            continue
        seq = read_fasta(fasta)
        donors, acceptors = splice_sites_from_gff(gff)
        # convert GFF 1-based to 0-based approx for window center at splice junction
        donor0 = {p - 1 for p in donors}
        acc0 = {p - 1 for p in acceptors}
        cands = []
        for i in range(1, len(seq) - 1):
            if seq[i : i + 2] == "GT" or seq[i - 1 : i + 1] == "AG":
                cands.append(i)
        if len(cands) > max_per_species:
            cands = list(rng.choice(cands, size=max_per_species, replace=False))
        for i in cands:
            windows.append(window_at(seq, i))
            y_d.append(1.0 if i in donor0 else 0.0)
            y_a.append(1.0 if i in acc0 else 0.0)
    return windows, np.asarray(y_d), np.asarray(y_a)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ckpt_path = REPO / "src" / "model" / "cnn" / "splice" / "trained_models" / "fission_yeasts_splice_cnn.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu")
    backbone = ckpt.get("backbone", "dilated_cnn")
    model = build_splice_model(backbone=backbone)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    windows, y_d, y_a = collect_candidates()
    print(f"candidates={len(windows)} donor_pos={int(y_d.sum())} acceptor_pos={int(y_a.sum())}")
    x = one_hot_encode_windows(windows)
    with torch.no_grad():
        logits = model(x)
        d_logit = logits[:, 0]
        a_logit = logits[:, 1]
        d_conf = logits_to_confidence(d_logit).numpy()
        a_conf = logits_to_confidence(a_logit).numpy()

    d_correct = ((d_conf >= 0.5) == (y_d >= 0.5)).astype(np.float64)
    a_correct = ((a_conf >= 0.5) == (y_a >= 0.5)).astype(np.float64)
    d_diag = expected_calibration_error(d_conf, d_correct, n_bins=15)
    a_diag = expected_calibration_error(a_conf, a_correct, n_bins=15)

    # MC dropout on a subset
    subset = min(2048, len(windows))
    mean, var = mc_dropout_predict(model, x[:subset], n_samples=16)
    write_mc_dropout_diagnostics(
        model,
        windows[:512],
        list(range(512)),
        OUT / "mc_dropout_diagnostics_deep.tsv",
        n_samples=12,
    )

    payload = {
        "n_candidates": len(windows),
        "donor_positives": int(y_d.sum()),
        "acceptor_positives": int(y_a.sum()),
        "ece_donor": d_diag.ece,
        "ece_acceptor": a_diag.ece,
        "donor_accuracy": float(d_correct.mean()),
        "acceptor_accuracy": float(a_correct.mean()),
        "donor_mean_confidence": float(d_conf.mean()),
        "acceptor_mean_confidence": float(a_conf.mean()),
        "mc_mean_var": float(var.mean().item()),
        "mc_n_samples": 16,
        "mc_subset": subset,
    }
    (OUT / "ece_deep.json").write_text(json.dumps(payload, indent=2) + "\n")

    lines = ["task\tbin\tconfidence\taccuracy\tcount"]
    for task, diag in (("donor", d_diag), ("acceptor", a_diag)):
        for i, (c, a, n) in enumerate(zip(diag.bin_confidence, diag.bin_accuracy, diag.bin_counts)):
            lines.append(f"{task}\t{i}\t{c:.6f}\t{a:.6f}\t{n}")
    (OUT / "reliability_diagram_deep.tsv").write_text("\n".join(lines) + "\n")

    # SVG for donor
    xs = [c for c, n in zip(d_diag.bin_confidence, d_diag.bin_counts) if n > 0]
    ys = [a for a, n in zip(d_diag.bin_accuracy, d_diag.bin_counts) if n > 0]
    w, h, pad = 420, 420, 50

    def xy(c: float, a: float) -> tuple[float, float]:
        return pad + c * (w - 2 * pad), h - pad - a * (h - 2 * pad)

    poly = " ".join(f"{xy(c, a)[0]:.1f},{xy(c, a)[1]:.1f}" for c, a in zip(xs, ys))
    x0, y0 = xy(0, 0)
    x1, y1 = xy(1, 1)
    circles = "\n".join(
        f'<circle cx="{xy(c, a)[0]:.1f}" cy="{xy(c, a)[1]:.1f}" r="4" fill="#1f4e79"/>' for c, a in zip(xs, ys)
    )
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}">
  <rect width="100%" height="100%" fill="white"/>
  <line x1="{x0:.1f}" y1="{y0:.1f}" x2="{x1:.1f}" y2="{y1:.1f}" stroke="#999" stroke-dasharray="4"/>
  <polyline fill="none" stroke="#1f4e79" stroke-width="2" points="{poly}"/>
  {circles}
  <text x="{pad}" y="24" font-family="Helvetica" font-size="14">Donor reliability (ECE={d_diag.ece:.3f})</text>
</svg>
"""
    (OUT / "reliability_diagram_deep.svg").write_text(svg)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
