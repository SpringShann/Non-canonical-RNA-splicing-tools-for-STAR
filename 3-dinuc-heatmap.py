#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate donor/acceptor 2-bp heatmaps from a tab-delimited motif table.

Expected input format (header required):
    chr    start    end    strand    strand_source    motif    intron_length    gene_id    unique_reads    multi_reads

Key behavior:
  - Reads the "motif" column only, formatted as DONOR/ACCEPTOR (e.g. TG/GA)
  - Keeps only valid 2-bp motifs composed of A/C/G/T
  - Outputs BOTH:
      1) all donor/acceptor sites (including canonical)
      2) filtered donor/acceptor sites excluding canonical donor GT and acceptor AG

Outputs:
  - count TSVs
  - frequency TSVs
  - high-resolution transparent heatmaps (counts / freq)

Command line:
python 3-dinuc-heatmap.py \
  --input output/2-noncanonical-motifs.txt \
  --outdir output/0325_heatmap_out \
  --format png
"""

#--format svg

#python3 3.5-dinuc-heatmap.py --input output/2-noncanonical-motifs.txt --outdir output/0325_heatmap_out --norm power --gamma 0.5

import argparse
import csv
import os
import re
from collections import Counter

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

BASES = ["A", "C", "G", "T"]
IDX = {b: i for i, b in enumerate(BASES)}
MOTIF_RE = re.compile(r"^([ACGT]{2})/([ACGT]{2})$")
DEFAULT_POWER_GAMMA = 0.5


def parse_args():
    ap = argparse.ArgumentParser(
        description="Build donor/acceptor dinucleotide heatmaps from a tab-delimited motif table."
    )
    ap.add_argument(
        "--input",
        "--txt",
        dest="input_path",
        required=True,
        help="Tab-delimited text file with a 'motif' column (e.g. TG/GA).",
    )
    ap.add_argument("--outdir", required=True, help="Output directory")
    ap.add_argument(
        "--dpi", type=int, default=600, help="Output resolution (default: 600)"
    )
    ap.add_argument(
        "--format",
        choices=["png", "pdf", "svg"],
        default="png",
        help="Output format (default: png)",
    )
    ap.add_argument(
        "--transparent",
        action="store_true",
        default=True,
        help="Transparent background (enabled by default)",
    )
    ap.add_argument(
        "--norm",
        choices=["linear", "log", "power"],
        default="linear",
        help="Color normalization for visualization only (default: linear)",
    )
    ap.add_argument(
        "--gamma",
        type=float,
        default=DEFAULT_POWER_GAMMA,
        help="Gamma for power normalization (default: 0.5)",
    )
    return ap.parse_args()


def extract_sites_from_motif(motif_value: str):
    """Parse a motif like 'TG/GA' into donor and acceptor 2-mers, without filtering."""
    if motif_value is None:
        return None, None

    motif = str(motif_value).strip().upper()
    match = MOTIF_RE.fullmatch(motif)
    if not match:
        return None, None

    donor_site, acceptor_site = match.group(1), match.group(2)
    return donor_site, acceptor_site


def read_sites_from_table(input_path: str):
    """Read donor and acceptor sites from the motif column of a tab-delimited table."""
    donor_sites_all = []
    acceptor_sites_all = []

    total_rows = 0
    parsed_rows = 0
    invalid_motif_rows = 0

    with open(input_path, "r", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")

        if reader.fieldnames is None:
            raise ValueError("Input file is empty or missing a header row.")
        if "motif" not in reader.fieldnames:
            raise ValueError(
                f"Missing required column 'motif'. Found columns: {', '.join(reader.fieldnames)}"
            )

        for row in reader:
            total_rows += 1
            d_site, a_site = extract_sites_from_motif(row.get("motif"))

            if d_site is None and a_site is None:
                invalid_motif_rows += 1
                continue

            parsed_rows += 1
            if d_site:
                donor_sites_all.append(d_site)
            if a_site:
                acceptor_sites_all.append(a_site)

    donor_sites_filtered = [s for s in donor_sites_all if s != "GT"]
    acceptor_sites_filtered = [s for s in acceptor_sites_all if s != "AG"]

    stats = {
        "total_rows": total_rows,
        "parsed_rows": parsed_rows,
        "invalid_motif_rows": invalid_motif_rows,
        "have_d_all": len(donor_sites_all),
        "have_a_all": len(acceptor_sites_all),
        "have_d_filtered": len(donor_sites_filtered),
        "have_a_filtered": len(acceptor_sites_filtered),
        "excluded_donor_GT": len(donor_sites_all) - len(donor_sites_filtered),
        "excluded_acceptor_AG": len(acceptor_sites_all) - len(acceptor_sites_filtered),
    }

    return donor_sites_all, acceptor_sites_all, donor_sites_filtered, acceptor_sites_filtered, stats


def count_dinucs(site_list):
    cnt = Counter()
    for s in site_list:
        if re.fullmatch(r"[ACGT]{2}", s):
            cnt[s] += 1
    return cnt


def counter_to_matrix(cnt: Counter):
    mat = np.zeros((4, 4), dtype=float)
    for b1 in BASES:
        for b2 in BASES:
            mat[IDX[b1], IDX[b2]] = cnt.get(b1 + b2, 0)
    return mat


def save_matrix_tsv(path, mat, as_freq=False):
    with open(path, "w", newline="") as f:
        header = [""] + BASES
        f.write("\t".join(header) + "\n")
        for i, b1 in enumerate(BASES):
            row = [b1] + [
                f"{mat[i, j]:.6f}" if as_freq else str(int(mat[i, j]))
                for j in range(4)
            ]
            f.write("\t".join(row) + "\n")


def masked_matrix_for_display(mat, grey_cells=None):
    if not grey_cells:
        return np.ma.array(mat, mask=np.zeros_like(mat, dtype=bool))

    mask = np.zeros_like(mat, dtype=bool)
    for i, j in grey_cells:
        if 0 <= i < mat.shape[0] and 0 <= j < mat.shape[1]:
            mask[i, j] = True
    return np.ma.array(mat, mask=mask)


def build_color_norm(display_mat, norm_mode, gamma, masked):
    """
    Color normalization affects visualization only.
    Numeric annotations remain the original matrix values.
    """
    positive_vals = np.asarray(display_mat)[np.asarray(display_mat) > 0]

    if norm_mode == "linear":
        vmax = float(np.max(display_mat)) if np.max(display_mat) > 0 else 1.0
        return mpl.colors.Normalize(vmin=0.0, vmax=vmax)

    if norm_mode == "power":
        vmax = float(np.max(display_mat)) if np.max(display_mat) > 0 else 1.0
        return mpl.colors.PowerNorm(gamma=gamma, vmin=0.0, vmax=vmax)

    if norm_mode == "log":
        if positive_vals.size == 0:
            return mpl.colors.Normalize(vmin=0.0, vmax=1.0)

        # Mask zeros for LogNorm visualization only; labels still show original values.
        masked.mask |= np.asarray(display_mat) <= 0
        vmin = float(np.min(positive_vals))
        vmax = float(np.max(positive_vals))
        if vmin == vmax:
            vmax = vmin * 1.000001
        return mpl.colors.LogNorm(vmin=vmin, vmax=vmax)

    raise ValueError(f"Unsupported normalization mode: {norm_mode}")


def plot_heatmap(
    path_no_ext,
    mat,
    dpi=600,
    fmt="png",
    transparent=True,
    grey_cells=None,
    norm_mode="linear",
    gamma=DEFAULT_POWER_GAMMA,
    annotation_decimals=2,
):
    mpl.rcParams["savefig.transparent"] = transparent
    fig, ax = plt.subplots(figsize=(4.0, 3.5))
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")

    display_mat = mat.copy()

    if grey_cells:
        for i, j in grey_cells:
            if 0 <= i < display_mat.shape[0] and 0 <= j < display_mat.shape[1]:
                display_mat[i, j] = 0.0

    masked = masked_matrix_for_display(display_mat, grey_cells=grey_cells)
    cmap = mpl.cm.get_cmap().copy()
    cmap.set_bad("#D9D9D9")
    norm = build_color_norm(display_mat, norm_mode, gamma, masked)
    im = ax.imshow(masked, aspect="equal", interpolation="nearest", cmap=cmap, norm=norm)

    if grey_cells:
        grey_overlay = np.full(display_mat.shape, np.nan)
        for i, j in grey_cells:
            if 0 <= i < grey_overlay.shape[0] and 0 <= j < grey_overlay.shape[1]:
                grey_overlay[i, j] = 1.0
        grey_cmap = mpl.colors.ListedColormap(["#BFBFBF"])
        ax.imshow(
            grey_overlay,
            aspect="equal",
            interpolation="nearest",
            cmap=grey_cmap,
            vmin=0,
            vmax=1,
        )

    ax.set_xticks(range(4))
    ax.set_xticklabels(BASES)
    ax.set_yticks(range(4))
    ax.set_yticklabels(BASES)

    cbar = fig.colorbar(im)
    cbar.ax.set_facecolor("none")
    cbar.outline.set_visible(False)

    for i in range(4):
        for j in range(4):
            v = display_mat[i, j]
            txt = (
                f"{v:.{annotation_decimals}f}"
                if 0 < v < 1
                else (f"{v:.0f}" if v >= 1 else "0")
            )
            ax.text(j, i, txt, ha="center", va="center", fontsize=9, color="black")

    for spine in ax.spines.values():
        spine.set_linewidth(0.6)

    plt.tight_layout(pad=0.5)
    out_path = f"{path_no_ext}.{fmt}"
    fig.savefig(
        out_path,
        dpi=dpi,
        transparent=transparent,
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.close(fig)
    return out_path


def save_bundle(
    outdir,
    prefix,
    mat,
    freq,
    dpi,
    fmt,
    transparent,
    grey_cells=None,
    norm_mode="linear",
    gamma=DEFAULT_POWER_GAMMA,
):
    save_matrix_tsv(os.path.join(outdir, f"{prefix}_counts.tsv"), mat, as_freq=False)
    save_matrix_tsv(os.path.join(outdir, f"{prefix}_freq.tsv"), freq, as_freq=True)

    plot_heatmap(
        os.path.join(outdir, f"{prefix}_heatmap_counts_{norm_mode}"),
        mat,
        dpi=dpi,
        fmt=fmt,
        transparent=transparent,
        grey_cells=grey_cells,
        norm_mode=norm_mode,
        gamma=gamma,
        annotation_decimals=2,
    )
    plot_heatmap(
        os.path.join(outdir, f"{prefix}_heatmap_freq_{norm_mode}"),
        freq,
        dpi=dpi,
        fmt=fmt,
        transparent=transparent,
        grey_cells=grey_cells,
        norm_mode=norm_mode,
        gamma=gamma,
        annotation_decimals=3,
    )


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    (
        donor_sites_all,
        acceptor_sites_all,
        donor_sites_filtered,
        acceptor_sites_filtered,
        stats,
    ) = read_sites_from_table(args.input_path)

    print(f"[INFO] Input rows read: {stats['total_rows']}")
    print(f"[INFO] Rows with usable motif entries: {stats['parsed_rows']}")
    print(f"[INFO] Rows skipped (invalid motif): {stats['invalid_motif_rows']}")
    print(f"[INFO] Donor sites (all): {stats['have_d_all']}")
    print(f"[INFO] Acceptor sites (all): {stats['have_a_all']}")
    print(f"[INFO] Donor sites (filtered, GT excluded): {stats['have_d_filtered']} [excluded GT: {stats['excluded_donor_GT']}]")
    print(f"[INFO] Acceptor sites (filtered, AG excluded): {stats['have_a_filtered']} [excluded AG: {stats['excluded_acceptor_AG']}]")

    donor_cnt_all = count_dinucs(donor_sites_all)
    accept_cnt_all = count_dinucs(acceptor_sites_all)

    donor_mat_all = counter_to_matrix(donor_cnt_all)
    accept_mat_all = counter_to_matrix(accept_cnt_all)

    donor_freq_all = donor_mat_all / donor_mat_all.sum() if donor_mat_all.sum() > 0 else donor_mat_all
    accept_freq_all = accept_mat_all / accept_mat_all.sum() if accept_mat_all.sum() > 0 else accept_mat_all

    donor_cnt_filtered = count_dinucs(donor_sites_filtered)
    accept_cnt_filtered = count_dinucs(acceptor_sites_filtered)

    donor_mat_filtered = counter_to_matrix(donor_cnt_filtered)
    accept_mat_filtered = counter_to_matrix(accept_cnt_filtered)

    donor_freq_filtered = donor_mat_filtered / donor_mat_filtered.sum() if donor_mat_filtered.sum() > 0 else donor_mat_filtered
    accept_freq_filtered = accept_mat_filtered / accept_mat_filtered.sum() if accept_mat_filtered.sum() > 0 else accept_mat_filtered

    save_bundle(
        args.outdir,
        "donor_site2_all",
        donor_mat_all,
        donor_freq_all,
        args.dpi,
        args.format,
        args.transparent,
        grey_cells=None,
        norm_mode=args.norm,
        gamma=args.gamma,
    )
    save_bundle(
        args.outdir,
        "acceptor_site2_all",
        accept_mat_all,
        accept_freq_all,
        args.dpi,
        args.format,
        args.transparent,
        grey_cells=None,
        norm_mode=args.norm,
        gamma=args.gamma,
    )

    donor_gt_cell = [(IDX["G"], IDX["T"])]
    acceptor_ag_cell = [(IDX["A"], IDX["G"])]

    save_bundle(
        args.outdir,
        "donor_site2_filtered_exclGT",
        donor_mat_filtered,
        donor_freq_filtered,
        args.dpi,
        args.format,
        args.transparent,
        grey_cells=donor_gt_cell,
        norm_mode=args.norm,
        gamma=args.gamma,
    )
    save_bundle(
        args.outdir,
        "acceptor_site2_filtered_exclAG",
        accept_mat_filtered,
        accept_freq_filtered,
        args.dpi,
        args.format,
        args.transparent,
        grey_cells=acceptor_ag_cell,
        norm_mode=args.norm,
        gamma=args.gamma,
    )

    print(f"[INFO] Heatmap color normalization: {args.norm}")
    if args.norm == "power":
        print(f"[INFO] PowerNorm gamma: {args.gamma}")
    print(f"[OK] Output directory: {args.outdir}")


if __name__ == "__main__":
    main()
