"""Shared neural emission backbones for structured sequence modeling."""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn

BACKBONE_NAMES = ("dilated_cnn", "bilstm", "transformer")
BackboneName = Literal["dilated_cnn", "bilstm", "transformer"]


def one_hot_encode_windows(windows: list[str]) -> torch.Tensor:
    encoded = torch.zeros((len(windows), 4, len(windows[0])), dtype=torch.float32)
    channel_by_base = {"A": 0, "C": 1, "G": 2, "T": 3}
    for row, window in enumerate(windows):
        for col, base in enumerate(window.upper()):
            channel = channel_by_base.get(base)
            if channel is not None:
                encoded[row, channel, col] = 1.0
    return encoded


class DilatedCNNEncoder(nn.Module):
    """Dilated temporal CNN encoder (existing production backbone)."""

    POOL_BINS = 8

    def __init__(self, hidden_channels: int = 128, dropout: float = 0.2) -> None:
        super().__init__()
        self.hidden_channels = hidden_channels
        self.net = nn.Sequential(
            nn.Conv1d(4, hidden_channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Dropout(p=dropout),
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
            nn.AdaptiveMaxPool1d(self.POOL_BINS),
        )
        self.out_dim = hidden_channels * self.POOL_BINS

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 4, L)
        return self.net(x).flatten(1)


class BiLSTMEncoder(nn.Module):
    def __init__(self, hidden_size: int = 128, num_layers: int = 2, dropout: float = 0.2) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.input_proj = nn.Linear(4, hidden_size)
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.out_dim = hidden_size * 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 4, L) -> (B, L, 4)
        seq = x.transpose(1, 2)
        seq = self.input_proj(seq)
        out, _ = self.lstm(seq)
        return out.mean(dim=1)


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        d_model: int = 128,
        n_head: int = 4,
        n_layer: int = 2,
        max_len: int = 256,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.input_proj = nn.Linear(4, d_model)
        self.pos = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layer)
        self.out_dim = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 4, L) -> (B, L, 4)
        seq = x.transpose(1, 2)
        h = self.input_proj(seq)
        length = h.size(1)
        h = h + self.pos[:, :length, :]
        h = self.encoder(h)
        return h.mean(dim=1)


def build_encoder(
    name: BackboneName | str,
    hidden: int = 128,
    dropout: float = 0.2,
) -> nn.Module:
    name = name.lower().replace("-", "_")
    if name in ("dilated_cnn", "cnn", "dilated"):
        return DilatedCNNEncoder(hidden_channels=hidden, dropout=dropout)
    if name in ("bilstm", "lstm"):
        return BiLSTMEncoder(hidden_size=hidden, dropout=dropout)
    if name in ("transformer", "tf"):
        return TransformerEncoder(d_model=hidden, dropout=dropout)
    raise ValueError(f"Unknown backbone: {name}. Expected one of {BACKBONE_NAMES}")
