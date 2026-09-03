"""Windowed masked nucleotide language-model pretraining."""

from __future__ import annotations

import random
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .backbones import build_encoder, one_hot_encode_windows


BASES = "ACGT"
MASK_INDEX = 4  # fifth channel for mask token in MLM input


class MaskedWindowDataset(Dataset):
    def __init__(
        self,
        sequences: list[str],
        window: int = 121,
        mask_prob: float = 0.15,
        samples_per_seq: int = 64,
        seed: int = 0,
    ) -> None:
        self.window = window
        self.mask_prob = mask_prob
        self.examples: list[str] = []
        rng = random.Random(seed)
        for seq in sequences:
            seq = "".join(b if b in BASES else "N" for b in seq.upper())
            if len(seq) < window:
                continue
            for _ in range(samples_per_seq):
                start = rng.randrange(0, len(seq) - window + 1)
                self.examples.append(seq[start : start + window])

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        window = self.examples[idx]
        # target: class indices 0-3 for ACGT, 4 for N
        target = torch.full((self.window,), 4, dtype=torch.long)
        channel_by_base = {"A": 0, "C": 1, "G": 2, "T": 3}
        for i, b in enumerate(window):
            if b in channel_by_base:
                target[i] = channel_by_base[b]

        x = one_hot_encode_windows([window])[0]  # (4, L)
        mask = torch.zeros(self.window, dtype=torch.bool)
        for i in range(self.window):
            if target[i] == 4:
                continue
            if random.random() < self.mask_prob:
                mask[i] = True
                # 80% zeros (mask), 10% random, 10% keep
                r = random.random()
                if r < 0.8:
                    x[:, i] = 0.0
                elif r < 0.9:
                    x[:, i] = 0.0
                    x[random.randrange(4), i] = 1.0
        return x, target, mask


class NucleotideMLM(nn.Module):
    def __init__(self, backbone: str = "transformer", hidden: int = 128) -> None:
        super().__init__()
        self.encoder = build_encoder(backbone, hidden=hidden)
        # For MLM we need per-position features; use a lightweight position head on raw conv/lstm path.
        # Project encoder global features is insufficient — use a position-wise head via 1x1 conv on input+encode.
        self.backbone_name = backbone
        if backbone in ("dilated_cnn", "cnn", "dilated"):
            self.pos_encoder = nn.Sequential(
                nn.Conv1d(4, hidden, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.Conv1d(hidden, hidden, kernel_size=3, padding=2, dilation=2),
                nn.ReLU(),
            )
            self.pred = nn.Conv1d(hidden, 4, kernel_size=1)
        elif backbone in ("bilstm", "lstm"):
            self.input_proj = nn.Linear(4, hidden)
            self.lstm = nn.LSTM(hidden, hidden, num_layers=2, batch_first=True, bidirectional=True)
            self.pred = nn.Linear(hidden * 2, 4)
        else:
            self.input_proj = nn.Linear(4, hidden)
            self.pos = nn.Parameter(torch.zeros(1, 256, hidden))
            nn.init.normal_(self.pos, std=0.02)
            layer = nn.TransformerEncoderLayer(
                d_model=hidden, nhead=4, dim_feedforward=hidden * 4, batch_first=True, activation="gelu"
            )
            self.tf = nn.TransformerEncoder(layer, num_layers=2)
            self.pred = nn.Linear(hidden, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # returns (B, L, 4) logits
        if hasattr(self, "pos_encoder"):
            h = self.pos_encoder(x)
            return self.pred(h).transpose(1, 2)
        seq = x.transpose(1, 2)
        if hasattr(self, "lstm"):
            h = self.input_proj(seq)
            h, _ = self.lstm(h)
            return self.pred(h)
        h = self.input_proj(seq)
        h = h + self.pos[:, : h.size(1), :]
        h = self.tf(h)
        return self.pred(h)


def load_fasta_sequences(paths: list[Path], max_bases: int | None = None) -> list[str]:
    sequences: list[str] = []
    total = 0
    for path in paths:
        parts: list[str] = []
        with path.open() as handle:
            for line in handle:
                if line.startswith(">"):
                    if parts:
                        sequences.append("".join(parts))
                        total += len(parts and "".join(parts) or "")
                        parts = []
                    continue
                parts.append(line.strip().upper())
                if max_bases is not None and total + sum(len(p) for p in parts) >= max_bases:
                    break
        if parts:
            sequences.append("".join(parts))
            total += len(sequences[-1])
        if max_bases is not None and total >= max_bases:
            break
    return sequences


def train_mlm(
    model: NucleotideMLM,
    sequences: list[str],
    epochs: int = 2,
    batch_size: int = 64,
    window: int = 121,
    samples_per_seq: int = 32,
    device: str | torch.device = "cpu",
) -> dict[str, float]:
    dataset = MaskedWindowDataset(sequences, window=window, samples_per_seq=samples_per_seq)
    if len(dataset) == 0:
        raise ValueError("No MLM windows could be sampled from provided sequences.")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss(ignore_index=4)
    token_count = 0
    last_loss = 0.0
    model.train()
    for _ in range(epochs):
        for x, target, mask in loader:
            x = x.to(device)
            target = target.to(device)
            mask = mask.to(device)
            logits = model(x)  # (B, L, 4)
            # only supervise masked positions that are ACGT
            supervised = mask & (target < 4)
            if supervised.sum() == 0:
                continue
            loss = loss_fn(logits[supervised], target[supervised])
            opt.zero_grad()
            loss.backward()
            opt.step()
            last_loss = float(loss.item())
            token_count += int(supervised.sum().item())
    return {"tokens_supervised": float(token_count), "last_loss": last_loss, "n_windows": float(len(dataset))}
