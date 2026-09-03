"""Diagnostic scoring with MC-dropout variance (C++ still uses mean-only TSVs)."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from .backbones import one_hot_encode_windows
from .calibration import mc_dropout_predict


def write_mc_dropout_diagnostics(
    model: nn.Module,
    windows: list[str],
    positions: list[int],
    output_path: Path,
    n_samples: int = 10,
    batch_size: int = 256,
    device: str | torch.device = "cpu",
) -> None:
    """Write position, mean logits..., variance... for uncertainty diagnostics."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model = model.to(device)
    with output_path.open("w") as out:
        out.write("position\tmean_0\tmean_1\tvar_0\tvar_1\n")
        for start in range(0, len(windows), batch_size):
            batch_w = windows[start : start + batch_size]
            batch_p = positions[start : start + batch_size]
            x = one_hot_encode_windows(batch_w).to(device)
            mean, var = mc_dropout_predict(model, x, n_samples=n_samples)
            mean = mean.cpu()
            var = var.cpu()
            for i, pos in enumerate(batch_p):
                m0 = float(mean[i, 0])
                m1 = float(mean[i, 1]) if mean.size(1) > 1 else 0.0
                v0 = float(var[i, 0])
                v1 = float(var[i, 1]) if var.size(1) > 1 else 0.0
                out.write(f"{pos}\t{m0:.8f}\t{m1:.8f}\t{v0:.8f}\t{v1:.8f}\n")
