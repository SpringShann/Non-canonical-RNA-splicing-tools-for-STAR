#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Same as dinuc_site2_0909.py, with added filtering:
  - donor site: exclude "GT"
  - acceptor site: exclude "AG"

Outputs: count/frequency TSVs and high-resolution transparent heatmaps (counts / freq)
"""

import argparse
import os
import re
from collections import Counter
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

BASES = ["A", "C", "G", "T"]
IDX = {b: i for i, b in enumerate(BASES)}

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", required=True, help="TSV containing #context lines (site 2-mers in brackets)")
    ap.add_argument("--outdir", required=True, help="Output directory")
    ap.add_argument("--dpi", type=int, default=600, help="Output resolution (default: 600)")
    ap.add_argument("--format", choices=["png", "pdf", "svg"], default="png",
                    help="Output format (default: png)")
    ap.add_argument("--transparent", action="store_true", default=True,
                    help="Transparent background (enabled by default)")
    return ap.parse_args()

def extract_sites_from_context(line: str):
    cols = line.rstrip("\n").split("\t")
    if len(cols) < 5:
        return None, None
    donor_field = cols[2].strip().upper()
    accept_field = cols[4].strip().upper()
    m_d = re.search(r"\[([ACGTN]{1,})\]", donor_field)
    m_a = re.search(r"\[([ACGTN]{1,})\]", acceptor_field)
    donor_site = m_d.group(1) if m_d else ""
    acceptor_site = m_a.group(1) if m_a else ""
    # Keep only valid 2-bp sites
    if not re.fullmatch(r"[ACGT]{2}", donor_site or ""):
        donor_site = ""
    if not re.fullmatch(r"[ACGT]{2}", acceptor_site or ""):
        acceptor_site = ""
    # Exclude canonical sites: donor GT, acceptor AG
    if donor_site == "GT":
        donor_site = ""
    if acceptor_site == "AG":
        acceptor_site = ""
    if not donor_site and not acceptor_site:
        return None, None
    return donor_site, acceptor_site

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
    with open(path, "w") as f:
        header = [""] + BASES
        f.write("\t".join(header) + "\n")
        for i, b1 in enumerate(BASES):
            row = [b1] + [f"{mat[i, j]:.6f}" if as_freq else str(int(mat[i, j])) for j in range(4)]
            f.write("\t".join(row) + "\n")

def plot_heatmap(path_no_ext, mat, dpi=600, fmt="png", transparent=True):
    mpl.rcParams["savefig.transparent"] = transparent
    fig, ax = plt.subplots(figsize=(4.0, 3.5))
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")

    im = ax.imshow(mat, aspect="equal", interpolation="nearest")
    ax.set_xticks(range(4)); ax.set_xticklabels(BASES)
    ax.set_yticks(range(4)); ax.set_yticklabels(BASES)

    cbar = fig.colorbar(im)
    cbar.ax.set_facecolor("none")
    cbar.outline.set_visible(False)

    for i in range(4):
        for j in range(4):
            v = mat[i, j]
            txt = f"{v:.2f}" if 0 < v < 1 else (f"{v:.0f}" if v >= 1 else "0")
            ax.text(j, i, txt, ha="center", va="center", fontsize=9, color="black")

    for spine in ax.spines.values():
        spine.set_linewidth(0.6)

    plt.tight_layout(pad=0.5)
    out_path = f"{path_no_ext}.{fmt}"
    fig.savefig(out_path, dpi=dpi, transparent=transparent,
                bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return out_path

def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    donor_sites, acceptor_sites = [], []
    total = have_d = have_a = 0
    with open(args.tsv, "r") as f:
        for line in f:
            if not line.startswith("#context"):
                continue
            total += 1
            d_site, a_site = extract_sites_from_context(line)
            if d_site:
                donor_sites.append(d_site); have_d += 1
            if a_site:
                acceptor_sites.append(a_site); have_a += 1

    print(f"[INFO] Parsed #context lines: {total}")
    print(f"[INFO] Valid donor sites (GT excluded): {have_d}")
    print(f"[INFO] Valid acceptor sites (AG excluded): {have_a}")

    donor_cnt = count_dinucs(donor_sites)
    accept_cnt = count_dinucs(acceptor_sites)

    donor_mat = counter_to_matrix(donor_cnt)
    accept_mat = counter_to_matrix(acceptor_cnt)

    donor_freq = donor_mat / donor_mat.sum() if donor_mat.sum() > 0 else donor_mat
    accept_freq = accept_mat / accept_mat.sum() if accept_mat.sum() > 0 else accept_mat

    save_matrix_tsv(os.path.join(args.outdir, "donor_site2_exclGT_counts.tsv"), donor_mat, as_freq=False)
    save_matrix_tsv(os.path.join(args.outdir, "donor_site2_exclGT_freq.tsv"), donor_freq, as_freq=True)
    save_matrix_tsv(os.path.join(args.outdir, "acceptor_site2_exclAG_counts.tsv"), accept_mat, as_freq=False)
    save_matrix_tsv(os.path.join(args.outdir, "acceptor_site2_exclAG_freq.tsv"), accept_freq, as_freq=True)

    plot_heatmap(os.path.join(args.outdir, "donor_site2_exclGT_heatmap_counts"),
                 donor_mat, dpi=args.dpi, fmt=args.format, transparent=args.transparent)
    plot_heatmap(os.path.join(args.outdir, "donor_site2_exclGT_heatmap_freq"),
                 donor_freq, dpi=args.dpi, fmt=args.format, transparent=args.transparent)
    plot_heatmap(os.path.join(args.outdir, "acceptor_site2_exclAG_heatmap_counts"),
                 accept_mat, dpi=args.dpi, fmt=args.format, transparent=args.transparent)
    plot_heatmap(os.path.join(args.outdir, "acceptor_site2_exclAG_heatmap_freq"),
                 accept_freq, dpi=args.dpi, fmt=args.format, transparent=args.transparent)

    print(f"[OK] Output directory: {args.outdir}")

if __name__ == "__main__":
    main()