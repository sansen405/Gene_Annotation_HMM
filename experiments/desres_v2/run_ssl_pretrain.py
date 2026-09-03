#!/usr/bin/env python3
"""Windowed MLM pretrain + scratch vs pretrained fine-tune compare (Phase 2)."""

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
from emissions_nn.models import MultiTaskEmissionModel
from emissions_nn.pretrain import NucleotideMLM, load_fasta_sequences, train_mlm

OUT = REPO / "experiments" / "desres_v2" / "ssl"


def _finetune(encoder_init: dict | None, windows: list[str], labels: torch.Tensor, steps: int = 40) -> float:
    model = MultiTaskEmissionModel(backbone="transformer", hidden=64)
    if encoder_init is not None:
        # Load overlapping keys from MLM position-wise transformer into shared encoder if present
        own = model.state_dict()
        mapped = {}
        for k, v in encoder_init.items():
            # best-effort: ignore non-matching
            if k in own and own[k].shape == v.shape:
                mapped[k] = v
        own.update(mapped)
        model.load_state_dict(own)
    x = one_hot_encode_windows(windows)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    bce = nn.BCEWithLogitsLoss()
    model.train()
    last = 0.0
    for _ in range(steps):
        out = model(x)
        loss = bce(out["donor"], labels) + bce(out["acceptor"], labels) + bce(out["start"], labels)
        opt.zero_grad()
        loss.backward()
        opt.step()
        last = float(loss.item())
    return last


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fasta_paths = []
    for sp in ("s_pombe", "s_japonicus", "s_octosporus", "s_cryophilus"):
        p = REPO / "genome_data" / "fission_yeasts" / sp / "train" / f"{sp}_train.fna"
        if p.exists():
            fasta_paths.append(p)
    # Optional fungi span
    fungi_root = REPO / "genome_data" / "fungi_diverse"
    if fungi_root.exists():
        for p in sorted(fungi_root.rglob("*_train.fna"))[:4]:
            fasta_paths.append(p)

    # Prefer many shorter contigs so window sampling covers diversity; cap total bases.
    sequences = load_fasta_sequences(fasta_paths, max_bases=5_000_000) if fasta_paths else ["ATGC" * 8000]
    # If FASTA load collapsed to few long chromosomes, chunk for denser window sampling.
    chunked: list[str] = []
    chunk = 50_000
    for seq in sequences:
        if len(seq) <= chunk:
            chunked.append(seq)
            continue
        for i in range(0, len(seq) - chunk + 1, chunk):
            chunked.append(seq[i : i + chunk])
            if len(chunked) >= 200:
                break
        if len(chunked) >= 200:
            break
    sequences = chunked or sequences
    mlm = NucleotideMLM(backbone="transformer", hidden=64)
    stats = train_mlm(
        mlm,
        sequences,
        epochs=1,
        batch_size=64,
        window=121,
        samples_per_seq=64,
        device="cpu",
    )
    torch.save({"state_dict": mlm.state_dict(), "stats": stats}, OUT / "mlm_pretrained.pt")

    rng = np.random.default_rng(7)
    bases = np.array(list("ACGT"))
    windows = ["".join(bases[rng.integers(0, 4, size=121)].tolist()) for _ in range(96)]
    labels = torch.tensor(
        [[1.0] if w[60:62] == "GT" else [0.0] for w in windows], dtype=torch.float32
    )

    scratch_loss = _finetune(None, windows, labels)
    pretrained_loss = _finetune(mlm.state_dict(), windows, labels)

    payload = {
        **stats,
        "scratch_finetune_bce": scratch_loss,
        "pretrained_finetune_bce": pretrained_loss,
        "pretrained_better": pretrained_loss <= scratch_loss,
        "fasta_files": [str(p.relative_to(REPO)) for p in fasta_paths],
        "max_bases_loaded": 5_000_000,
        "note": "Laptop smoke: token count is measured; scale samples_per_seq for larger X.",
    }
    (OUT / "scratch_vs_pretrained.json").write_text(json.dumps(payload, indent=2) + "\n")
    (OUT / "token_count.txt").write_text(
        f"tokens_supervised={int(stats['tokens_supervised'])}\n"
        f"n_windows={int(stats['n_windows'])}\n"
        f"claim: pretrained encoder on {int(stats['tokens_supervised'])} masked nucleotides "
        f"(windowed MLM) before supervised fine-tuning\n"
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
