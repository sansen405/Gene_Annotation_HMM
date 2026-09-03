"""Calibration and uncertainty utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


@dataclass
class ReliabilityDiagram:
    bin_confidence: list[float]
    bin_accuracy: list[float]
    bin_counts: list[int]
    ece: float


def expected_calibration_error(
    confidences: np.ndarray,
    correct: np.ndarray,
    n_bins: int = 15,
) -> ReliabilityDiagram:
    confidences = np.asarray(confidences, dtype=np.float64)
    correct = np.asarray(correct, dtype=np.float64)
    assert confidences.shape == correct.shape
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_conf: list[float] = []
    bin_acc: list[float] = []
    bin_counts: list[int] = []
    ece = 0.0
    n = len(confidences)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (confidences > lo) & (confidences <= hi) if i > 0 else (confidences >= lo) & (confidences <= hi)
        count = int(mask.sum())
        bin_counts.append(count)
        if count == 0:
            bin_conf.append(0.0)
            bin_acc.append(0.0)
            continue
        c = float(confidences[mask].mean())
        a = float(correct[mask].mean())
        bin_conf.append(c)
        bin_acc.append(a)
        ece += (count / max(n, 1)) * abs(a - c)
    return ReliabilityDiagram(bin_conf, bin_acc, bin_counts, float(ece))


@torch.no_grad()
def mc_dropout_predict(
    model: nn.Module,
    x: torch.Tensor,
    n_samples: int = 10,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Enable dropout at eval time; return mean and variance of outputs."""
    was_training = model.training
    model.train()  # keep dropout active
    samples = []
    for _ in range(n_samples):
        samples.append(model(x).detach())
    stacked = torch.stack(samples, dim=0)
    mean = stacked.mean(dim=0)
    var = stacked.var(dim=0, unbiased=False)
    if not was_training:
        model.eval()
    return mean, var


def logits_to_confidence(logits: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(logits)
