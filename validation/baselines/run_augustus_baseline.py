#!/usr/bin/env python3
"""Run Augustus on fission-yeast holdouts and score vs gold GFF.

Compares nucleotide-level coding / intron precision-recall-F1 and start/stop
boundary exact-match rates against the same test FASTA/GFF pairs used by
validation/full_genome_validation.cpp.

Requires: augustus on PATH (brew install augustus).
Uses the schizosaccharomyces_pombe species model for all four Schizosaccharomyces
holdouts (closest available Augustus parameter set).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROFILE = REPO_ROOT / "src/genome_profiles/fission_yeasts/fission_yeasts.json"
DEFAULT_OUT = REPO_ROOT / "validation/baselines/augustus"
AUGUSTUS_SPECIES = "schizosaccharomyces_pombe"


@dataclass
class BinaryMetrics:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0

    def add(self, other: "BinaryMetrics") -> None:
        self.tp += other.tp
        self.fp += other.fp
        self.fn += other.fn
        self.tn += other.tn

    @property
    def precision(self) -> float:
        den = self.tp + self.fp
        return self.tp / den if den else 0.0

    @property
    def recall(self) -> float:
        den = self.tp + self.fn
        return self.tp / den if den else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0


@dataclass
class BoundaryMetrics:
    predicted: int = 0
    gold: int = 0
    exact: int = 0

    def add(self, other: "BoundaryMetrics") -> None:
        self.predicted += other.predicted
        self.gold += other.gold
        self.exact += other.exact

    @property
    def precision(self) -> float:
        return self.exact / self.predicted if self.predicted else 0.0

    @property
    def recall(self) -> float:
        return self.exact / self.gold if self.gold else 0.0


@dataclass
class ChromResult:
    name: str
    length: int = 0
    coding: BinaryMetrics = field(default_factory=BinaryMetrics)
    intron: BinaryMetrics = field(default_factory=BinaryMetrics)
    start: BoundaryMetrics = field(default_factory=BoundaryMetrics)
    stop: BoundaryMetrics = field(default_factory=BoundaryMetrics)
    predicted_genes: int = 0
    gold_genes: int = 0
    wall_seconds: float = 0.0


def load_profile(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    return data["dataset"]["species"]


def fasta_length(path: Path) -> int:
    length = 0
    with path.open() as handle:
        for line in handle:
            if line.startswith(">"):
                continue
            length += len(line.strip())
    return length


def parse_gff_cds(path: Path) -> dict[str, list[tuple[int, int, str, str]]]:
    """Return chrom -> list of (start0, end_exclusive, strand, transcript_id)."""
    by_chrom: dict[str, list[tuple[int, int, str, str]]] = defaultdict(list)
    with path.open() as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9:
                continue
            chrom, _src, ftype, start, end, _score, strand, _phase, attrs = parts[:9]
            if ftype not in {"CDS", "cds"}:
                continue
            tid = _transcript_id(attrs, ftype)
            # GFF is 1-based inclusive
            by_chrom[chrom].append((int(start) - 1, int(end), strand, tid))
    return by_chrom


def _transcript_id(attrs: str, ftype: str) -> str:
    fields = {}
    for token in attrs.split(";"):
        token = token.strip()
        if not token:
            continue
        if "=" in token:
            key, val = token.split("=", 1)
        elif " " in token:
            key, val = token.split(" ", 1)
            val = val.strip('"')
        else:
            continue
        fields[key] = val
    for key in ("Parent", "transcript_id", "ID", "gene_id"):
        if key in fields:
            return fields[key]
    return f"anon_{ftype}"


def paint_masks(
    length: int,
    cds_intervals: list[tuple[int, int, str, str]],
    *,
    extend_stop_codon: bool = False,
) -> tuple[list[bool], list[bool], set[int], set[int], int]:
    """Build coding/intron masks and start/stop sets for both strands merged.

    If extend_stop_codon is True (Augustus default with stopCodonExcludedFromCDS),
    extend the terminal CDS by 3 bp so stop coordinates match NCBI-style GFFs that
    include the stop codon in CDS.
    """
    coding = [False] * length
    intron = [False] * length
    starts: set[int] = set()
    stops: set[int] = set()

    by_tx: dict[tuple[str, str], list[tuple[int, int]]] = defaultdict(list)
    for start, end, strand, tid in cds_intervals:
        by_tx[(strand, tid)].append((start, end))

    genes = 0
    for (strand, _tid), exons in by_tx.items():
        exons = sorted(exons)
        if not exons:
            continue
        genes += 1
        if extend_stop_codon:
            if strand == "+":
                last_start, last_end = exons[-1]
                exons[-1] = (last_start, min(length, last_end + 3))
            else:
                first_start, first_end = exons[0]
                exons[0] = (max(0, first_start - 3), first_end)
        for start, end in exons:
            for i in range(max(0, start), min(length, end)):
                coding[i] = True
        if strand == "+":
            starts.add(exons[0][0])
            stops.add(exons[-1][1] - 1)
        else:
            starts.add(exons[-1][1] - 1)
            stops.add(exons[0][0])
        for i in range(len(exons) - 1):
            gap_start = exons[i][1]
            gap_end = exons[i + 1][0]
            for j in range(max(0, gap_start), min(length, gap_end)):
                if not coding[j]:
                    intron[j] = True
    return coding, intron, starts, stops, genes


def score_masks(
    gold_coding: list[bool],
    gold_intron: list[bool],
    pred_coding: list[bool],
    pred_intron: list[bool],
    gold_starts: set[int],
    gold_stops: set[int],
    pred_starts: set[int],
    pred_stops: set[int],
) -> tuple[BinaryMetrics, BinaryMetrics, BoundaryMetrics, BoundaryMetrics]:
    coding = BinaryMetrics()
    intron = BinaryMetrics()
    for g, p in zip(gold_coding, pred_coding):
        if g and p:
            coding.tp += 1
        elif not g and p:
            coding.fp += 1
        elif g and not p:
            coding.fn += 1
        else:
            coding.tn += 1
    for g, p in zip(gold_intron, pred_intron):
        if g and p:
            intron.tp += 1
        elif not g and p:
            intron.fp += 1
        elif g and not p:
            intron.fn += 1
        else:
            intron.tn += 1

    start = BoundaryMetrics(
        predicted=len(pred_starts),
        gold=len(gold_starts),
        exact=len(pred_starts & gold_starts),
    )
    stop = BoundaryMetrics(
        predicted=len(pred_stops),
        gold=len(gold_stops),
        exact=len(pred_stops & gold_stops),
    )
    return coding, intron, start, stop


def run_augustus(fasta: Path, out_gff: Path, species: str) -> float:
    out_gff.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "augustus",
        f"--species={species}",
        "--gff3=on",
        "--strand=both",
        str(fasta),
    ]
    start = time.perf_counter()
    with out_gff.open("w") as handle:
        subprocess.run(cmd, check=True, stdout=handle, stderr=subprocess.PIPE, text=True)
    return time.perf_counter() - start


def evaluate_species(entry: dict, out_dir: Path, species_model: str, skip_run: bool) -> ChromResult:
    name = entry["name"]
    fasta = REPO_ROOT / entry["test_fasta"]
    gold_gff = REPO_ROOT / entry["test_gff"]
    pred_gff = out_dir / f"{name}.gff"

    result = ChromResult(name=name, length=fasta_length(fasta))
    if not skip_run or not pred_gff.exists():
        result.wall_seconds = run_augustus(fasta, pred_gff, species_model)
    else:
        result.wall_seconds = 0.0

    gold_by_chrom = parse_gff_cds(gold_gff)
    pred_by_chrom = parse_gff_cds(pred_gff)

    # Holdout FASTAs are single-chromosome
    chrom = next(iter(gold_by_chrom)) if gold_by_chrom else next(iter(pred_by_chrom), "unknown")
    gold_cds = gold_by_chrom.get(chrom, [])
    # Augustus may use a truncated seqname; fall back to union of all chrom keys
    pred_cds = pred_by_chrom.get(chrom, [])
    if not pred_cds and pred_by_chrom:
        pred_cds = [iv for ivs in pred_by_chrom.values() for iv in ivs]

    g_coding, g_intron, g_starts, g_stops, g_genes = paint_masks(result.length, gold_cds)
    p_coding, p_intron, p_starts, p_stops, p_genes = paint_masks(
        result.length, pred_cds, extend_stop_codon=True
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


def format_report(results: list[ChromResult], hmm_combined: Path | None) -> str:
    combined_coding = BinaryMetrics()
    combined_intron = BinaryMetrics()
    combined_start = BoundaryMetrics()
    combined_stop = BoundaryMetrics()
    total_bases = 0
    lines = ["=== Augustus baseline (schizosaccharomyces_pombe model) ===", ""]

    for r in results:
        combined_coding.add(r.coding)
        combined_intron.add(r.intron)
        combined_start.add(r.start)
        combined_stop.add(r.stop)
        total_bases += r.length
        lines.append(f"--- {r.name} (len={r.length:,}, wall={r.wall_seconds:.1f}s) ---")
        lines.append(
            f"coding  P={r.coding.precision:.4f} R={r.coding.recall:.4f} F1={r.coding.f1:.4f}"
        )
        lines.append(
            f"intron  P={r.intron.precision:.4f} R={r.intron.recall:.4f} F1={r.intron.f1:.4f}"
        )
        lines.append(
            f"start   P={r.start.precision:.4f} R={r.start.recall:.4f} "
            f"exact={r.start.exact} pred={r.start.predicted} gold={r.start.gold}"
        )
        lines.append(
            f"stop    P={r.stop.precision:.4f} R={r.stop.recall:.4f} "
            f"exact={r.stop.exact} pred={r.stop.predicted} gold={r.stop.gold}"
        )
        lines.append(f"genes   predicted={r.predicted_genes} gold={r.gold_genes}")
        lines.append("")

    lines.append("=== Combined (all holdout chromosomes) ===")
    lines.append(f"Evaluated bases                   {total_bases}")
    lines.append(
        f"coding  P={combined_coding.precision:.4f} R={combined_coding.recall:.4f} "
        f"F1={combined_coding.f1:.4f}"
    )
    lines.append(
        f"intron  P={combined_intron.precision:.4f} R={combined_intron.recall:.4f} "
        f"F1={combined_intron.f1:.4f}"
    )
    lines.append(
        f"start   P={combined_start.precision:.4f} R={combined_start.recall:.4f} "
        f"exact={combined_start.exact} pred={combined_start.predicted} gold={combined_start.gold}"
    )
    lines.append(
        f"stop    P={combined_stop.precision:.4f} R={combined_stop.recall:.4f} "
        f"exact={combined_stop.exact} pred={combined_stop.predicted} gold={combined_stop.gold}"
    )
    lines.append("")

    # Side-by-side with HMM if available
    hmm = _parse_hmm_combined(hmm_combined) if hmm_combined and hmm_combined.exists() else None
    lines.append("=== Side-by-side (nucleotide metrics on same holdout FASTAs) ===")
    lines.append(
        f"{'Metric':<22} {'Augustus':>12} {'This HMM (V4.2)':>16}"
    )
    lines.append(
        f"{'Coding F1':<22} {combined_coding.f1:12.4f} "
        f"{(hmm['coding_f1'] if hmm else float('nan')):16.4f}"
    )
    lines.append(
        f"{'Intron F1':<22} {combined_intron.f1:12.4f} "
        f"{(hmm['intron_f1'] if hmm else float('nan')):16.4f}"
    )
    lines.append(
        f"{'Start boundary P':<22} {combined_start.precision:12.4f} "
        f"{(hmm['start_p'] if hmm else float('nan')):16.4f}"
    )
    lines.append(
        f"{'Start boundary R':<22} {combined_start.recall:12.4f} "
        f"{(hmm['start_r'] if hmm else float('nan')):16.4f}"
    )
    lines.append(
        f"{'Stop boundary P':<22} {combined_stop.precision:12.4f} "
        f"{(hmm['stop_p'] if hmm else float('nan')):16.4f}"
    )
    lines.append(
        f"{'Stop boundary R':<22} {combined_stop.recall:12.4f} "
        f"{(hmm['stop_r'] if hmm else float('nan')):16.4f}"
    )
    lines.append("")
    lines.append(
        "Notes: Augustus uses the schizosaccharomyces_pombe parameter set for all "
        "four species (trained parameters unavailable for japonicus/octosporus/"
        "cryophilus). Coding/intron labels are painted from CDS features on both "
        "strands into one mask per chromosome (intron = gaps between consecutive "
        "CDS of the same transcript). Augustus CDS is extended by 3 bp at the "
        "terminal exon to include the stop codon (stopCodonExcludedFromCDS). "
        "HMM metrics from validation/results/version_4/combined.txt evaluate each "
        "strand separately and sum (~2× bases); nucleotide F1 is still the fairest "
        "available head-to-head on these FASTAs. HMM also reports donor/acceptor."
    )
    return "\n".join(lines) + "\n"


def _parse_hmm_combined(path: Path) -> dict[str, float]:
    text = path.read_text()
    out: dict[str, float] = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 4 and parts[0] == "coding":
            out["coding_f1"] = float(parts[3])
        elif len(parts) >= 4 and parts[0] == "intron":
            out["intron_f1"] = float(parts[3])
        elif len(parts) >= 3 and parts[0] == "start":
            out["start_p"] = float(parts[1])
            out["start_r"] = float(parts[2])
        elif len(parts) >= 3 and parts[0] == "stop":
            out["stop_p"] = float(parts[1])
            out["stop_r"] = float(parts[2])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--species-model", default=AUGUSTUS_SPECIES)
    parser.add_argument("--skip-run", action="store_true", help="Reuse existing Augustus GFFs")
    parser.add_argument(
        "--hmm-results",
        type=Path,
        default=REPO_ROOT / "validation/results/version_4/combined.txt",
    )
    args = parser.parse_args()

    species = load_profile(args.profile.resolve())
    results = [
        evaluate_species(entry, args.out_dir.resolve(), args.species_model, args.skip_run)
        for entry in species
    ]
    report = format_report(results, args.hmm_results.resolve())
    args.out_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.out_dir / "combined.txt"
    report_path.write_text(report)
    print(report)
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
