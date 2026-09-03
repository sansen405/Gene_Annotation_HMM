#!/usr/bin/env python3
"""Dwell diagnostic: geometric A self-loop mean vs empirical / HSMM intron lengths.

Reads training GFF-derived intron body lengths indirectly by asking the C++
validation binary is heavy; this script instead consumes a simple lengths TSV
(one integer per line) plus an optional self-loop log-prob, and writes a table
comparing geometric / histogram / negative-binomial mean dwells.

Example:
  python3 experiments/version_5/duration/dwell_diagnostic.py \\
    --lengths experiments/version_5/duration/example_lengths.txt \\
    --self-loop-log-prob -0.02 \\
    --max-length 500 \\
    --out experiments/version_5/duration/dwell_summary.tsv
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path


def percentile(sorted_vals: list[int], p: float) -> float:
    if not sorted_vals:
        return float("nan")
    idx = int(p * (len(sorted_vals) - 1))
    return float(sorted_vals[idx])


def histogram_mean(lengths: list[int], max_length: int) -> float:
    counts = [1.0] * (max_length + 1)
    counts[0] = 0.0
    for length in lengths:
        if 1 <= length <= max_length:
            counts[length] += 1.0
    total = sum(counts[1:])
    if total <= 0:
        return float("nan")
    return sum(i * counts[i] for i in range(1, max_length + 1)) / total


def negative_binomial_mean(lengths: list[int], max_length: int) -> float:
    usable = [L for L in lengths if 1 <= L <= max_length]
    if not usable:
        return float("nan")
    mean = sum(usable) / len(usable)
    var = sum(L * L for L in usable) / len(usable) - mean * mean
    var = max(var, mean + 1e-6)
    r = (mean * mean) / (var - mean) if var > mean else mean
    r = max(1e-3, r)
    p = min(1.0 - 1e-9, max(1e-9, r / (r + mean)))
    unnorm = []
    for k in range(1, max_length + 1):
        # log C(k+r-1, k) + k log(1-p) + r log p
        log_pmf = (
            math.lgamma(k + r) - math.lgamma(k + 1.0) - math.lgamma(r)
            + k * math.log(1.0 - p)
            + r * math.log(p)
        )
        unnorm.append(math.exp(log_pmf))
    total = sum(unnorm)
    if total <= 0:
        return float("nan")
    return sum((i + 1) * unnorm[i] for i in range(max_length)) / total


def geometric_mean_from_self_loop(log_a: float) -> float:
    a = math.exp(log_a)
    if a >= 1.0:
        return float("inf")
    return 1.0 / (1.0 - a)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lengths", type=Path, required=True)
    parser.add_argument("--self-loop-log-prob", type=float, default=-0.02)
    parser.add_argument("--max-length", type=int, default=0, help="0 => use empirical p95")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    lengths = [
        int(line.strip())
        for line in args.lengths.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    lengths_sorted = sorted(lengths)
    max_length = args.max_length or int(percentile(lengths_sorted, 0.95))

    rows = [
        ("n_lengths", len(lengths)),
        ("max_length", max_length),
        ("empirical_mean", sum(lengths) / len(lengths) if lengths else float("nan")),
        ("empirical_p50", percentile(lengths_sorted, 0.50)),
        ("empirical_p95", percentile(lengths_sorted, 0.95)),
        ("geometric_mean_from_A", geometric_mean_from_self_loop(args.self_loop_log_prob)),
        ("histogram_mean", histogram_mean(lengths, max_length)),
        ("negative_binomial_mean", negative_binomial_mean(lengths, max_length)),
    ]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as handle:
        handle.write("metric\tvalue\n")
        for name, value in rows:
            handle.write(f"{name}\t{value}\n")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
