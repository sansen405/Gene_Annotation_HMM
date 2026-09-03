"""CORAL domain adaptation for encoder features."""

from __future__ import annotations

import torch


def coral_loss(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """CORAL: align second-order statistics of source/target features.

    source/target: (N, D)
    """
    if source.size(0) < 2 or target.size(0) < 2:
        return source.new_zeros(())

    def _cov(x: torch.Tensor) -> torch.Tensor:
        x = x - x.mean(dim=0, keepdim=True)
        return (x.t() @ x) / (x.size(0) - 1)

    d = source.size(1)
    diff = _cov(source) - _cov(target)
    return (diff * diff).sum() / (4.0 * d * d)


def gradient_reversal(x: torch.Tensor, lambd: float = 1.0) -> torch.Tensor:
    """Simple gradient reversal via custom autograd for adversarial DA."""

    class _Rev(torch.autograd.Function):
        @staticmethod
        def forward(ctx, inp, lambd_):
            ctx.lambd = lambd_
            return inp.view_as(inp)

        @staticmethod
        def backward(ctx, grad_output):
            return -ctx.lambd * grad_output, None

    return _Rev.apply(x, lambd)
