"""Segment-level linear-chain CRF / soft-Viterbi for emission fine-tuning."""

from __future__ import annotations

import torch
from torch import nn


# Reduced biology-inspired labels for short segments.
LABELS = ("intergenic", "coding", "intron", "donor", "acceptor", "start", "stop")
LABEL_TO_IDX = {name: i for i, name in enumerate(LABELS)}
NUM_LABELS = len(LABELS)


def log_sum_exp(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    m, _ = x.max(dim=dim, keepdim=True)
    return (m + torch.log(torch.clamp(torch.exp(x - m).sum(dim=dim, keepdim=True), min=1e-12))).squeeze(dim)


class SegmentCRF(nn.Module):
    """Linear-chain CRF over reduced gene-structure labels."""

    def __init__(self, num_labels: int = NUM_LABELS) -> None:
        super().__init__()
        self.num_labels = num_labels
        self.transitions = nn.Parameter(torch.zeros(num_labels, num_labels))
        self.start = nn.Parameter(torch.zeros(num_labels))
        self.end = nn.Parameter(torch.zeros(num_labels))
        # Soft biology mask: discourage illegal bigrams via large negative init bias
        with torch.no_grad():
            illegal = torch.zeros(num_labels, num_labels)
            # donor should not follow acceptor directly, etc. — mild prior, not hard constraint
            illegal[LABEL_TO_IDX["donor"], LABEL_TO_IDX["acceptor"]] = -2.0
            illegal[LABEL_TO_IDX["start"], LABEL_TO_IDX["stop"]] = -2.0
            self.transitions.add_(illegal)

    def _score_path(self, emissions: torch.Tensor, tags: torch.Tensor) -> torch.Tensor:
        # emissions: (T, C), tags: (T,)
        score = self.start[tags[0]] + emissions[0, tags[0]]
        for t in range(1, emissions.size(0)):
            score = score + self.transitions[tags[t - 1], tags[t]] + emissions[t, tags[t]]
        score = score + self.end[tags[-1]]
        return score

    def _partition(self, emissions: torch.Tensor) -> torch.Tensor:
        # emissions: (T, C)
        alpha = self.start + emissions[0]
        for t in range(1, emissions.size(0)):
            scores = alpha.unsqueeze(1) + self.transitions + emissions[t].unsqueeze(0)
            alpha = log_sum_exp(scores, dim=0)
        return log_sum_exp(alpha + self.end, dim=0)

    def nll(self, emissions: torch.Tensor, tags: torch.Tensor) -> torch.Tensor:
        # emissions (B, T, C), tags (B, T)
        losses = []
        for b in range(emissions.size(0)):
            gold = self._score_path(emissions[b], tags[b])
            z = self._partition(emissions[b])
            losses.append(z - gold)
        return torch.stack(losses).mean()

    def soft_viterbi(self, emissions: torch.Tensor) -> torch.Tensor:
        """Return marginal posterior probabilities via forward-backward (B, T, C)."""
        batch = []
        for b in range(emissions.size(0)):
            em = emissions[b]
            t_len, c = em.shape
            alpha = torch.zeros(t_len, c, device=em.device)
            beta = torch.zeros(t_len, c, device=em.device)
            alpha[0] = self.start + em[0]
            for t in range(1, t_len):
                scores = alpha[t - 1].unsqueeze(1) + self.transitions + em[t].unsqueeze(0)
                alpha[t] = log_sum_exp(scores, dim=0)
            beta[-1] = self.end
            for t in range(t_len - 2, -1, -1):
                scores = self.transitions + em[t + 1].unsqueeze(0) + beta[t + 1].unsqueeze(0)
                beta[t] = log_sum_exp(scores, dim=1)
            log_z = log_sum_exp(alpha[-1] + self.end, dim=0)
            posterior = torch.exp(alpha + beta - log_z)
            batch.append(posterior)
        return torch.stack(batch, dim=0)


class EmissionToCRFUnary(nn.Module):
    """Map multi-task logits / features to CRF unary potentials."""

    def __init__(self, in_dim: int, num_labels: int = NUM_LABELS) -> None:
        super().__init__()
        self.proj = nn.Linear(in_dim, num_labels)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # features: (B, T, D) or (B, D) expanded
        return self.proj(features)
