#!/usr/bin/env python3
"""Parse validation combined.txt files and write DESRES V2 deep master tables."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "experiments" / "desres_v2"


def parse_combined(path: Path) -> dict[str, float | int | str]:
    text = path.read_text() if path.exists() else ""
    out: dict[str, float | int | str] = {"path": str(path)}

    def grab(pattern: str, cast=float):
        m = re.search(pattern, text, re.I | re.M)
        return cast(m.group(1)) if m else None

    out["bases"] = grab(r"Evaluated bases\s+(\d+)", int)
    out["exact21"] = grab(r"Exact 21-state accuracy\s+([0-9.]+)")
    # classification rows
    m = re.search(r"^coding\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)", text, re.M)
    if m:
        out["coding_p"], out["coding_r"], out["coding_f1"] = map(float, m.groups())
    m = re.search(r"^intron\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)", text, re.M)
    if m:
        out["intron_p"], out["intron_r"], out["intron_f1"] = map(float, m.groups())
    # structure block: exon/gene under Structure Metrics
    struct = text.split("Structure Metrics", 1)[-1] if "Structure Metrics" in text else text
    m = re.search(r"^exon\s+([0-9.]+)\s+([0-9.]+)", struct, re.M)
    if m:
        out["exon_p"], out["exon_r"] = map(float, m.groups())
    m = re.search(r"^gene\s+([0-9.]+)\s+([0-9.]+)", struct, re.M)
    if m:
        out["gene_p"], out["gene_r"] = map(float, m.groups())
    return out


def bootstrap_ci_from_species(val_dir: Path) -> dict[str, object]:
    """Cheap chromosome-level bootstrap: resample species files with replacement."""
    import numpy as np

    rows = []
    for p in sorted(val_dir.glob("*.txt")):
        if p.name == "combined.txt":
            continue
        d = parse_combined(p)
        if d.get("gene_p") is not None and d.get("intron_f1") is not None:
            rows.append(d)
    if len(rows) < 2:
        return {"n_species": len(rows), "note": "need ≥2 species files"}
    rng = np.random.default_rng(0)
    gene = []
    intron = []
    n = len(rows)
    for _ in range(1000):
        idx = rng.integers(0, n, size=n)
        gene.append(float(np.mean([rows[i]["gene_p"] for i in idx])))
        intron.append(float(np.mean([rows[i]["intron_f1"] for i in idx])))
    return {
        "n_species": n,
        "gene_p_mean": float(np.mean(gene)),
        "gene_p_ci95": [float(np.percentile(gene, 2.5)), float(np.percentile(gene, 97.5))],
        "intron_f1_mean": float(np.mean(intron)),
        "intron_f1_ci95": [float(np.percentile(intron, 2.5)), float(np.percentile(intron, 97.5))],
    }


def phase_backbone() -> None:
    lines = [
        "backbone\tgene_p\tgene_r\texon_p\texon_r\tintron_f1\tcoding_f1\texact21\tbases\tvalidation_path"
    ]
    summary = {}
    for bb in ("dilated_cnn", "bilstm", "transformer"):
        path = OUT / "backbone_ablation" / bb / "validation" / "combined.txt"
        d = parse_combined(path)
        summary[bb] = d
        if (OUT / "backbone_ablation" / bb / "validation").exists():
            d["bootstrap"] = bootstrap_ci_from_species(OUT / "backbone_ablation" / bb / "validation")
        lines.append(
            f"{bb}\t{d.get('gene_p','')}\t{d.get('gene_r','')}\t{d.get('exon_p','')}\t"
            f"{d.get('exon_r','')}\t{d.get('intron_f1','')}\t{d.get('coding_f1','')}\t"
            f"{d.get('exact21','')}\t{d.get('bases','')}\t{path}"
        )
    out = OUT / "backbone_ablation" / "results_table_deep.tsv"
    out.write_text("\n".join(lines) + "\n")
    (OUT / "backbone_ablation" / "results_deep.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print(f"wrote {out}")


def phase_master() -> None:
    master_lines = [
        "component\tsetting\tgene_p\tintron_f1\tcoding_f1\tece\ttokens\truntime_note"
    ]

    # backbones
    for bb in ("dilated_cnn", "bilstm", "transformer"):
        d = parse_combined(OUT / "backbone_ablation" / bb / "validation" / "combined.txt")
        master_lines.append(
            f"backbone\t{bb}\t{d.get('gene_p','')}\t{d.get('intron_f1','')}\t{d.get('coding_f1','')}\t\t\tdeep_holdout"
        )

    ece = ""
    ece_path = OUT / "calibration" / "ece_deep.json"
    if ece_path.exists():
        ece = json.loads(ece_path.read_text()).get("ece_donor", "")

    tokens = ""
    ssl_path = OUT / "ssl" / "scratch_vs_pretrained_deep.json"
    if ssl_path.exists():
        tokens = json.loads(ssl_path.read_text()).get("tokens_supervised", "")

    # V5 transfer baselines
    master_lines.append("transfer\tfission_holdout_V5\t0.8080\t0.8413\t0.9757\t\t\tversion_5")
    master_lines.append("transfer\tfungi_zeroshot\t0.3898\t0.6298\t0.8716\t\t\ttransfer_summary")
    master_lines.append("transfer\tfungi_indomain\t0.4557\t0.6397\t0.8785\t\t\ttransfer_summary")

    coral = parse_combined(OUT / "coral" / "validation" / "combined.txt")
    master_lines.append(
        f"adaptation\tcoral_adapted\t{coral.get('gene_p','')}\t{coral.get('intron_f1','')}\t"
        f"{coral.get('coding_f1','')}\t\t\tfungi_holdout"
    )

    crf_path = OUT / "segment_crf" / "ablation_deep.json"
    crf_note = ""
    if crf_path.exists():
        crf = json.loads(crf_path.read_text())
        crf_note = f"ce={crf.get('emission_only_boundary_f1')};crf={crf.get('crf_boundary_f1')}"

    master_lines.append(f"calibration\tdeep\t\t\t\t{ece}\t\treal_holdout_candidates")
    master_lines.append(f"ssl\tdeep\t\t\t\t\t{tokens}\tscratch_vs_pretrained")
    master_lines.append(f"segment_crf\tdeep\t\t\t\t\t\t{crf_note}")

    out = OUT / "results" / "master_ablation_deep.tsv"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(master_lines) + "\n")

    # Also refresh methods-facing summary
    summary = {
        "backbone_table": str(OUT / "backbone_ablation" / "results_table_deep.tsv"),
        "master_table": str(out),
        "ece": ece,
        "ssl_tokens": tokens,
        "coral": {k: coral.get(k) for k in ("gene_p", "intron_f1", "coding_f1")},
    }
    (OUT / "results" / "deep_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["backbone", "master"], required=True)
    args = ap.parse_args()
    if args.phase == "backbone":
        phase_backbone()
    else:
        phase_master()


if __name__ == "__main__":
    main()
