#!/usr/bin/env python3
"""Run Helixer (or helixer_post GFF) on holdout FASTAs and score vs gold.

Requires a working Helixer install, e.g.:
  pip install helixer  # heavy; needs TensorFlow
  # or use the official Docker image and point --helixer-cmd at a wrapper.

If Helixer is unavailable, use --predictions-dir with precomputed GFFs named
  <species>.gff (same schema as Augustus baseline scoring).
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

# Reuse Augustus scoring helpers
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_augustus_baseline import (  # type: ignore
    ChromResult,
    evaluate_species as _unused,  # noqa: F401
    fasta_length,
    format_report,
    load_profile,
    paint_masks,
    parse_gff_cds,
    score_masks,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROFILE = REPO_ROOT / "src/genome_profiles/fission_yeasts/fission_yeasts.json"
DEFAULT_OUT = REPO_ROOT / "validation/baselines/helixer"


def run_helixer(fasta: Path, out_gff: Path, helixer_cmd: list[str], lineage: str) -> float:
    out_gff.parent.mkdir(parents=True, exist_ok=True)
    # Helixer CLI variants differ; support a generic command template with placeholders.
    cmd = []
    for token in helixer_cmd:
        cmd.append(
            token.replace("{fasta}", str(fasta))
            .replace("{out}", str(out_gff))
            .replace("{lineage}", lineage)
        )
    start = time.perf_counter()
    subprocess.run(cmd, check=True)
    return time.perf_counter() - start


def evaluate_one(
    entry: dict,
    out_dir: Path,
    helixer_cmd: list[str] | None,
    lineage: str,
    skip_run: bool,
) -> ChromResult:
    name = entry["name"]
    fasta = REPO_ROOT / entry["test_fasta"]
    gold_gff = REPO_ROOT / entry["test_gff"]
    pred_gff = out_dir / f"{name}.gff"
    result = ChromResult(name=name, length=fasta_length(fasta))

    if not skip_run:
        if helixer_cmd is None:
            raise RuntimeError(
                "Helixer binary/command not configured and --skip-run not set. "
                "Pass --helixer-cmd or provide GFFs with --skip-run."
            )
        if not pred_gff.exists() or pred_gff.stat().st_size == 0:
            result.wall_seconds = run_helixer(fasta, pred_gff, helixer_cmd, lineage)
    if not pred_gff.exists():
        raise FileNotFoundError(f"Missing Helixer predictions: {pred_gff}")

    gold_by_chrom = parse_gff_cds(gold_gff)
    pred_by_chrom = parse_gff_cds(pred_gff)
    chrom = next(iter(gold_by_chrom)) if gold_by_chrom else next(iter(pred_by_chrom))
    gold_cds = gold_by_chrom.get(chrom, [])
    pred_cds = pred_by_chrom.get(chrom, [])
    if not pred_cds and pred_by_chrom:
        pred_cds = [iv for ivs in pred_by_chrom.values() for iv in ivs]

    g_coding, g_intron, g_starts, g_stops, g_genes = paint_masks(result.length, gold_cds)
    # Helixer (fungi lineage) already includes the stop codon in CDS — do not extend.
    p_coding, p_intron, p_starts, p_stops, p_genes = paint_masks(
        result.length, pred_cds, extend_stop_codon=False
    )
    coding, intron, start, stop = score_masks(
        g_coding, g_intron, p_coding, p_intron, g_starts, g_stops, p_starts, p_stops
    )
    result.coding = coding
    result.intron = intron
    result.start = start
    result.stop = stop
    result.gold_genes = g_genes
    result.predicted_genes = p_genes
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--helixer-cmd",
        nargs="+",
        default=None,
        help="Command tokens; supports {fasta} {out} {lineage} placeholders",
    )
    parser.add_argument("--lineage", default="fungi")
    parser.add_argument("--skip-run", action="store_true")
    parser.add_argument(
        "--hmm-results",
        type=Path,
        default=REPO_ROOT / "validation/results/version_4/combined.txt",
    )
    parser.add_argument(
        "--augustus-results",
        type=Path,
        default=REPO_ROOT / "validation/baselines/augustus/combined.txt",
    )
    args = parser.parse_args()

    if args.helixer_cmd is None and not args.skip_run:
        # Auto-detect common entry points
        for candidate in ("Helixer.py", "helixer"):
            path = shutil.which(candidate)
            if path:
                args.helixer_cmd = [
                    path,
                    "--fasta-path",
                    "{fasta}",
                    "--gff-output-path",
                    "{out}",
                    "--lineage",
                    "{lineage}",
                ]
                break

    species = load_profile(args.profile.resolve())
    results = [
        evaluate_one(entry, args.out_dir.resolve(), args.helixer_cmd, args.lineage, args.skip_run)
        for entry in species
    ]

    # format_report labels Augustus; rewrite header for Helixer
    report = format_report(results, args.hmm_results.resolve())
    report = report.replace(
        "=== Augustus baseline (schizosaccharomyces_pombe model) ===",
        "=== Helixer baseline (fungi lineage model via Docker) ===",
    )
    report = report.replace("Augustus uses the schizosaccharomyces_pombe parameter set for all ",
                            "Helixer fungi lineage model applied to all ")
    report = report.replace(
        "Augustus CDS is extended by 3 bp at the terminal exon to include the stop codon (stopCodonExcludedFromCDS). ",
        "Helixer CDS already includes the stop codon (no +3 extension). ",
    )
    report = report.replace("Augustus", "Helixer")

    # Append three-way table if Augustus report exists
    if args.augustus_results.exists():
        report += "\n=== Three-way note ===\n"
        report += f"Augustus report: {args.augustus_results}\n"
        report += f"HMM report:      {args.hmm_results}\n"
        report += "Compare Coding/Intron F1 and start/stop P/R across the three files.\n"

    args.out_dir.mkdir(parents=True, exist_ok=True)
    path = args.out_dir / "combined.txt"
    path.write_text(report)
    print(report)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
