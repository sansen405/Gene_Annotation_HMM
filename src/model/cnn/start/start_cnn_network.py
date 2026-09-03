from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

_REPO_MODEL = Path(__file__).resolve().parents[1]
if str(_REPO_MODEL) not in sys.path:
    sys.path.insert(0, str(_REPO_MODEL))

from emissions_nn.backbones import one_hot_encode_windows  # noqa: E402
from emissions_nn.models import StartEmissionModel  # noqa: E402


class StartCNN(nn.Module):
    """Legacy DilatedCNN start model (checkpoint-compatible module names)."""

    POOL_BINS = 8

    def __init__(self, window_size: int = 121, hidden_channels: int = 128) -> None:
        super().__init__()
        self.window_size = window_size
        self.backbone_name = "dilated_cnn"
        center = window_size // 2
        self.start_slice = (max(0, center - 20), min(window_size, center + 60))

        def _backbone() -> nn.Sequential:
            return nn.Sequential(
                nn.Conv1d(4, hidden_channels, kernel_size=7, padding=3),
                nn.BatchNorm1d(hidden_channels),
                nn.ReLU(),
                nn.Dropout(p=0.2),
                nn.Conv1d(hidden_channels, hidden_channels, kernel_size=5, padding=2),
                nn.BatchNorm1d(hidden_channels),
                nn.ReLU(),
                nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=2, dilation=2),
                nn.BatchNorm1d(hidden_channels),
                nn.ReLU(),
                nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=4, dilation=4),
                nn.BatchNorm1d(hidden_channels),
                nn.ReLU(),
                nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=8, dilation=8),
                nn.BatchNorm1d(hidden_channels),
                nn.ReLU(),
                nn.AdaptiveMaxPool1d(StartCNN.POOL_BINS),
            )

        self.start_features = _backbone()
        head_in = hidden_channels * StartCNN.POOL_BINS
        self.start_head = nn.Sequential(nn.Dropout(p=0.3), nn.Linear(head_in, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s_lo, s_hi = self.start_slice
        start_feat = self.start_features(x[:, :, s_lo:s_hi]).flatten(1)
        return self.start_head(start_feat)


def build_start_model(
    window_size: int = 121,
    backbone: str = "dilated_cnn",
    hidden: int = 128,
) -> nn.Module:
    name = backbone.lower().replace("-", "_")
    if name in ("dilated_cnn", "cnn", "dilated"):
        return StartCNN(window_size=window_size, hidden_channels=hidden)
    return StartEmissionModel(window_size=window_size, backbone=name, hidden=hidden)


__all__ = ["StartCNN", "build_start_model", "one_hot_encode_windows", "StartEmissionModel"]
