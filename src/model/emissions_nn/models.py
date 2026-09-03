"""Task heads that consume shared encoder features."""

from __future__ import annotations

import torch
from torch import nn

from .backbones import BackboneName, build_encoder


class SpliceEmissionModel(nn.Module):
    """Donor/acceptor logits from a pluggable backbone."""

    def __init__(
        self,
        window_size: int = 121,
        backbone: BackboneName | str = "dilated_cnn",
        hidden: int = 128,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.backbone_name = backbone
        center = window_size // 2
        self.donor_slice = (max(0, center - 10), min(window_size, center + 21))
        self.acceptor_slice = (max(0, center - 50), min(window_size, center + 11))
        self.donor_encoder = build_encoder(backbone, hidden=hidden)
        self.acceptor_encoder = build_encoder(backbone, hidden=hidden)
        self.donor_head = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(self.donor_encoder.out_dim, 1))
        self.acceptor_head = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(self.acceptor_encoder.out_dim, 1))

    def encode_donor(self, x: torch.Tensor) -> torch.Tensor:
        d_lo, d_hi = self.donor_slice
        return self.donor_encoder(x[:, :, d_lo:d_hi])

    def encode_acceptor(self, x: torch.Tensor) -> torch.Tensor:
        a_lo, a_hi = self.acceptor_slice
        return self.acceptor_encoder(x[:, :, a_lo:a_hi])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        donor_logit = self.donor_head(self.encode_donor(x))
        acceptor_logit = self.acceptor_head(self.encode_acceptor(x))
        return torch.cat([donor_logit, acceptor_logit], dim=1)


class StartEmissionModel(nn.Module):
    def __init__(
        self,
        window_size: int = 121,
        backbone: BackboneName | str = "dilated_cnn",
        hidden: int = 128,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.backbone_name = backbone
        center = window_size // 2
        self.start_slice = (max(0, center - 20), min(window_size, center + 60))
        self.encoder = build_encoder(backbone, hidden=hidden)
        self.start_head = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(self.encoder.out_dim, 1))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        s_lo, s_hi = self.start_slice
        return self.encoder(x[:, :, s_lo:s_hi])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.start_head(self.encode(x))


class MultiTaskEmissionModel(nn.Module):
    """Shared backbone with donor/acceptor/start heads."""

    def __init__(
        self,
        window_size: int = 121,
        backbone: BackboneName | str = "dilated_cnn",
        hidden: int = 128,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.backbone_name = backbone
        self.encoder = build_encoder(backbone, hidden=hidden)
        self.donor_head = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(self.encoder.out_dim, 1))
        self.acceptor_head = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(self.encoder.out_dim, 1))
        self.start_head = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(self.encoder.out_dim, 1))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        feat = self.encode(x)
        return {
            "donor": self.donor_head(feat),
            "acceptor": self.acceptor_head(feat),
            "start": self.start_head(feat),
            "features": feat,
        }
