#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Optional: Mann–Whitney U test
try:
    from scipy.stats import mannwhitneyu
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

# ===== Plot style: font, transparency, high resolution =====
plt.rcParams.update({
    "font.family": "Helvetica",           # Falls back if Helvetica is unavailable
    "savefig.dpi": 800,                   # High resolution
    "savefig.transparent": True,          # Transparent export
    "figure.facecolor": "none",           # Transparent canvas
    "axes.facecolor": "none",             # Transparent axes
    "legend.frameon": False,              # No legend frame
})

BLUE = "#1f77b4"   # Canonical
ORANGE = "#ff7f0e" # Non-canonical

def read_col_as_len(path: str, col_index_1based: int) -> np.ndarray:
    df = pd.read_csv(path, sep="\t", header=None, dtype=str, engine="python")
    if df.shape[1] < col_index_1based:
        raise ValueError(f"Not enough columns for {col_index_1based}: {path}")
    s = pd.to_numeric(df.iloc[:, col_index_1based - 1], errors="coerce")
    s = s[(s > 0) & np.isfinite(s)]
    return s.to_numpy(dtype=float)

def ecdf(values: np.ndarray):
    if values.size == 0:
        return np.array([]), np.array([])
    x = np.sort(values)
    y = np.arange(1, x.size + 1) / x.size
    return x, y

def plot_cdf_log10(canon_len, noncan_len, out_prefix):
    canon_log = np.log10(canon_len)
    noncan_log = np.log10(noncan_len)
    x1, y1 = ecdf(canon_log)
    x2, y2 = ecdf(noncan_log)

    plt.figure(figsize=(6.5, 4.3))
    if x1.size:
        plt.step(x1, y1, where="post", color=BLUE, label="Canonical", linewidth=1.6)
    if x2.size:
        plt.step(x2, y2, where="post", color=ORANGE, label="Non-canonical", linewidth=1.6)
    plt.xlabel("log10(Intron length, bp)")
    plt.ylabel("CDF")
    plt.legend(loc="lower center", ncol=2)
    plt.tight_layout()
    for ext in ("png", "pdf", "svg"):
        plt.savefig(f"{out_prefix}.{ext}", dpi=600, transparent=True)
    plt.close()

def plot_box_logy(canon_len, noncan_len, out_prefix):
    data = [canon_len, noncan_len]
    labels = ["Canonical", "Non-canonical"]

    fig, ax = plt.subplots(figsize=(4.8, 4.3))
    bp = ax.boxplot(
        data,
        labels=labels,
        patch_artist=True,
        showfliers=True,
        widths=0.65
    )
    for patch, color in zip(bp["boxes"], [BLUE, ORANGE]):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)
    for med in bp["medians"]:
        med.set_color("black")
        med.set_linewidth(1.6)
    for k in ["whiskers", "caps"]:
        for line in bp[k]:
            line.set_color("black")
            line.set_linewidth(1.2)
    means = [np.mean(x) if len(x) else np.nan for x in data]
    ax.scatter([1, 2], means, s=22, c="white", edgecolors="black", linewidths=0.8, zorder=3, label="Mean")

    ax.set_yscale("log")
    ax.set_ylabel("Intron length (bp, log10 scale)")
    ax.legend(loc="upper right")
    plt.tight_layout()
    for ext in ("png", "pdf", "svg"):
        plt.savefig(f"{out_prefix}.{ext}", dpi=600, transparent=True)
    plt.close()

def summary_stats(x: np.ndarray) -> dict:
    if x.size == 0:
        return dict(n=0, min=np.nan, q1=np.nan, median=np.nan, mean=np.nan, q3=np.nan, max=np.nan)
    q1, med, q3 = np.percentile(x, [25, 50, 75])
    return dict(
        n=int(x.size),
        min=float(np.min(x)),
        q1=float(q1),
        median=float(med),
        mean=float(np.mean(x)),
        q3=float(q3),
        max=float(np.max(x))
    )

def main():
    ap = argparse.ArgumentParser(
        description="Plot intron length CDF (log10) and boxplot (log y) for canonical vs non-canonical, high-res & transparent"
    )
    ap.add_argument("canonical_txt", help="Canonical input file (column 7 = intron length)")
    ap.add_argument("noncanonical_txt", help="Non-canonical input file (column 7 = intron length)")
    ap.add_argument("outdir", help="Output directory")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print(f"[INFO] Reading canonical (col7) from: {args.canonical_txt}")
    canon_len = read_col_as_len(args.canonical_txt, 7)
    print(f"[INFO] Reading non-canonical (col7) from: {args.noncanonical_txt}")
    noncan_len = read_col_as_len(args.noncanonical_txt, 7)

    plot_cdf_log10(canon_len, noncan_len, os.path.join(args.outdir, "intron_length_cdf_log10"))
    plot_box_logy(canon_len, noncan_len, os.path.join(args.outdir, "intron_length_boxplot_log10"))
    print("[OK] Figures saved (PNG + PDF + SVG, 600 dpi, transparent background).")

    s_c = summary_stats(canon_len)
    s_n = summary_stats(noncan_len)
    out_txt = os.path.join(args.outdir, "summary_and_stats.txt")
    lines = [
        "== Summary (Canonical) ==",
        str(s_c),
        "",
        "== Summary (Non-canonical) ==",
        str(s_n),
        "",
        "== Wilcoxon rank-sum test =="
    ]
    if HAVE_SCIPY and canon_len.size and noncan_len.size:
        stat, p = mannwhitneyu(canon_len, noncan_len, alternative="two-sided")
        lines += [f"U = {stat:.0f}", f"p-value = {p:.3e}"]
    else:
        lines += ["(scipy not installed or sample is empty; test skipped)"]
    with open(out_txt, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[OK] Summary & stats -> {out_txt}")

if __name__ == "__main__":
    main()