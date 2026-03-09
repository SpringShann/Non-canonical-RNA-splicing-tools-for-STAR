#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""
Example:
python 4-intron-length-comparison.py \
    output/2-canonical-motifs.txt \
    output/2-noncanonical-motifs.txt \
    output/2-U2_U12_like.txt \
    output/2-non-U2_U12_like.txt \
    output/intron_out_allgroups
"""

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
    "axes.linewidth": 1.2,
})

# Colors
BLUE = "#80a6e2"        # Canonical
ORANGE = "#ff7f0e"      # All non-canonical
YELLOW = "#fbdd85"      # U2/U12-like
RED = "#cf3d3e"         # non-U2/U12-like

GROUPS = [
    ("Canonical", BLUE),
    ("All non-canonical", ORANGE),
    ("U2/U12-like", YELLOW),
    ("non-U2/U12-like", RED),
]


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


def summary_stats(x: np.ndarray) -> dict:
    if x.size == 0:
        return dict(
            n=0, min=np.nan, q1=np.nan, median=np.nan,
            mean=np.nan, q3=np.nan, max=np.nan
        )
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


def mannwhitney_result(x: np.ndarray, y: np.ndarray):
    """
    Return (U, p) or (None, None) if unavailable.
    """
    if HAVE_SCIPY and x.size and y.size:
        stat, p = mannwhitneyu(x, y, alternative="two-sided")
        return stat, p
    return None, None


def mannwhitney_report(x: np.ndarray, y: np.ndarray, label_x: str, label_y: str):
    lines = [f"== {label_x} vs {label_y} =="]
    stat, p = mannwhitney_result(x, y)
    if stat is not None:
        lines.append(f"U = {stat:.0f}")
        lines.append(f"p-value = {p:.3e}")
    else:
        lines.append("(scipy not installed or sample is empty; test skipped)")
    lines.append("")
    return lines


def p_to_stars(p):
    """
    Convert p-value to significance stars.
    """
    if p is None or np.isnan(p):
        return None
    if p < 1e-4:
        return "****"
    elif p < 1e-3:
        return "***"
    elif p < 1e-2:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return None


def add_sig_bracket_log(ax, x1, x2, y, h_factor, text, lw=1.2, fontsize=11):
    """
    Draw significance bracket on a log-scale y-axis.
    """
    y_top = y * h_factor
    ax.plot(
        [x1, x1, x2, x2],
        [y, y_top, y_top, y],
        lw=lw,
        c="black",
        clip_on=False
    )
    ax.text(
        (x1 + x2) / 2,
        y_top * 0.88,
        text,
        ha="center",
        va="bottom",
        fontsize=fontsize,
        color="black"
    )


def compute_sig_annotations(datasets):
    """
    Compute selected pairwise comparisons and return only significant ones.

    Returns
    -------
    sig_results : list of dict
        each dict has:
        {
            "i": x-position of group1 (1-based),
            "j": x-position of group2 (1-based),
            "label1": group1 name,
            "label2": group2 name,
            "p": p-value,
            "stars": star string
        }
    """
    comparisons = [
        (2, 1),  # All non-canonical vs Canonical
        (3, 1),  # U2/U12-like vs Canonical
        (4, 1),  # non-U2/U12-like vs Canonical
        (3, 4),  # U2/U12-like vs non-U2/U12-like
    ]

    sig_results = []
    for i, j in comparisons:
        label_i, data_i, _ = datasets[i - 1]
        label_j, data_j, _ = datasets[j - 1]
        _, p = mannwhitney_result(data_i, data_j)
        stars = p_to_stars(p)
        if stars is not None:
            sig_results.append({
                "i": i,
                "j": j,
                "label1": label_i,
                "label2": label_j,
                "p": p,
                "stars": stars
            })
    return sig_results


def plot_cdf_log10(datasets, out_prefix):
    plt.figure(figsize=(7.6, 5.0))

    for label, values, color in datasets:
        if values.size == 0:
            continue
        values_log = np.log10(values)
        x, y = ecdf(values_log)
        if x.size:
            plt.step(x, y, where="post", color=color, label=label, linewidth=1.8)

    plt.xlabel("log10(Intron length, bp)")
    plt.ylabel("CDF")
    plt.legend(loc="lower center", ncol=2)
    plt.tight_layout()

    for ext in ("png", "pdf", "svg"):
        plt.savefig(f"{out_prefix}.{ext}", dpi=600, transparent=True)
    plt.close()


def plot_box_logy(datasets, out_prefix, sig_annotations=None):
    labels = [x[0] for x in datasets]
    data = [x[1] for x in datasets]
    colors = [x[2] for x in datasets]

    fig, ax = plt.subplots(figsize=(7.2, 5.6))

    flierprops = dict(
        marker='o',
        markerfacecolor='none',   # hollow
        markeredgecolor='black',
        markeredgewidth=0.8,
        markersize=3.2,
        linestyle='none',
        alpha=1.0                 # fully opaque
    )

    bp = ax.boxplot(
        data,
        labels=labels,
        patch_artist=True,
        showfliers=True,
        widths=0.62,
        flierprops=flierprops
    )

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)
        patch.set_edgecolor("black")
        patch.set_linewidth(1.2)

    for med in bp["medians"]:
        med.set_color("black")
        med.set_linewidth(1.8)

    for k in ["whiskers", "caps"]:
        for line in bp[k]:
            line.set_color("black")
            line.set_linewidth(1.2)

    ax.set_yscale("log")
    ax.set_ylabel("Intron length (bp, log scale)")
    plt.xticks(rotation=15)

    # Make figure cleaner and avoid overlap with top border
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add significance annotations only for significant comparisons
    if sig_annotations:
        nonempty = [d for d in data if len(d) > 0]
        if nonempty:
            all_values = np.concatenate(nonempty)
            y_max = np.max(all_values)

            # Start high enough above data, but leave headroom for stacked brackets
            base_y = y_max * 1.35
            h_factor = 1.06
            step_factor = 1.5

            # shorter brackets first to reduce visual clutter
            sig_annotations = sorted(
                sig_annotations,
                key=lambda x: (abs(x["j"] - x["i"]), min(x["i"], x["j"]), max(x["i"], x["j"]))
            )

            current_y = base_y
            for ann in sig_annotations:
                x1 = min(ann["i"], ann["j"])
                x2 = max(ann["i"], ann["j"])
                add_sig_bracket_log(
                    ax=ax,
                    x1=x1,
                    x2=x2,
                    y=current_y,
                    h_factor=h_factor,
                    text=ann["stars"]
                )
                current_y *= step_factor

            # Add enough headroom so brackets/stars do not touch canvas edge
            ax.set_ylim(top=current_y * 1.28)

    plt.tight_layout()
    plt.subplots_adjust(top=0.88)  # extra margin above plot area

    for ext in ("png", "pdf", "svg"):
        plt.savefig(f"{out_prefix}.{ext}", dpi=600, transparent=True)
    plt.close()


def main():
    ap = argparse.ArgumentParser(
        description="Plot 4-group intron length CDF (log10) and boxplot (log y), high-res & transparent, with significance stars."
    )
    ap.add_argument("canonical_txt", help="Canonical input file (column 7 = intron length)")
    ap.add_argument("all_noncanonical_txt", help="All non-canonical input file (column 7 = intron length)")
    ap.add_argument("u2u12_like_txt", help="U2/U12-like non-canonical input file (column 7 = intron length)")
    ap.add_argument("non_u2u12_like_txt", help="non-U2/U12-like input file (column 7 = intron length)")
    ap.add_argument("outdir", help="Output directory")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print(f"[INFO] Reading canonical (col7) from: {args.canonical_txt}")
    canon_len = read_col_as_len(args.canonical_txt, 7)

    print(f"[INFO] Reading all non-canonical (col7) from: {args.all_noncanonical_txt}")
    all_noncan_len = read_col_as_len(args.all_noncanonical_txt, 7)

    print(f"[INFO] Reading U2/U12-like (col7) from: {args.u2u12_like_txt}")
    u2u12_len = read_col_as_len(args.u2u12_like_txt, 7)

    print(f"[INFO] Reading non-U2/U12-like (col7) from: {args.non_u2u12_like_txt}")
    non_u2u12_len = read_col_as_len(args.non_u2u12_like_txt, 7)

    datasets = [
        ("Canonical", canon_len, BLUE),
        ("All non-canonical", all_noncan_len, ORANGE),
        ("U2/U12-like", u2u12_len, YELLOW),
        ("non-U2/U12-like", non_u2u12_len, RED),
    ]

    sig_annotations = compute_sig_annotations(datasets)

    plot_cdf_log10(
        datasets,
        os.path.join(args.outdir, "intron_length_cdf_log10_4groups")
    )

    plot_box_logy(
        datasets,
        os.path.join(args.outdir, "intron_length_boxplot_log_4groups"),
        sig_annotations=sig_annotations
    )

    print("[OK] Figures saved (PNG + PDF + SVG, 600 dpi, transparent background).")

    out_txt = os.path.join(args.outdir, "summary_and_stats_4groups.txt")
    lines = []

    summaries = [
        ("Canonical", canon_len),
        ("All non-canonical", all_noncan_len),
        ("U2/U12-like", u2u12_len),
        ("non-U2/U12-like", non_u2u12_len),
    ]

    for label, arr in summaries:
        lines.append(f"== Summary ({label}) ==")
        lines.append(str(summary_stats(arr)))
        lines.append("")

    lines.append("== Pairwise Mann-Whitney U tests ==")
    lines.append("")

    lines += mannwhitney_report(all_noncan_len, canon_len, "All non-canonical", "Canonical")
    lines += mannwhitney_report(u2u12_len, canon_len, "U2/U12-like", "Canonical")
    lines += mannwhitney_report(non_u2u12_len, canon_len, "non-U2/U12-like", "Canonical")
    lines += mannwhitney_report(u2u12_len, non_u2u12_len, "U2/U12-like", "non-U2/U12-like")

    lines.append("== Significance annotations plotted on boxplot ==")
    if sig_annotations:
        for ann in sig_annotations:
            lines.append(
                f'{ann["label1"]} vs {ann["label2"]}: p = {ann["p"]:.3e}, stars = {ann["stars"]}'
            )
    else:
        lines.append("No significant comparisons were annotated.")
    lines.append("")

    with open(out_txt, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"[OK] Summary & stats -> {out_txt}")


if __name__ == "__main__":
    main()
