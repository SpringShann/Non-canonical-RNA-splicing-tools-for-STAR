#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import pandas as pd

"""
python 5-count-splice-sites.py \
    output/2-canonical-motifs.txt \
    output/2-U2_U12_like.txt \
    output/2-non-U2_U12_like.txt \
    output/5-splice-site-summary.txt
"""

def reverse_motif_if_needed(motif, strand):
    """
    Standardize motif orientation based on strand.

    Parameters
    ----------
    motif : str
        Splice site motif in format like 'GT/AG'
    strand : int or str
        Strand code from column 4:
        1 = forward
        2 = reverse

    Returns
    -------
    str
        Standardized motif in forward donor/acceptor order, like 'GT/AG'
    """
    if pd.isna(motif):
        return None

    motif = str(motif).strip().upper()

    if "/" not in motif:
        return motif

    parts = motif.split("/")
    if len(parts) != 2:
        return motif

    donor, acceptor = parts[0], parts[1]

    try:
        strand = int(strand)
    except Exception:
        return f"{donor}/{acceptor}"

    if strand == 2:
        donor, acceptor = acceptor, donor

    return f"{donor}/{acceptor}"


def motif_to_splice_site(motif):
    """
    Convert 'GT/AG' -> 'GTAG'
    """
    if pd.isna(motif):
        return None
    return str(motif).replace("/", "").strip().upper()


def read_and_process_file(filepath):
    """
    Read one motif table and return a Series of standardized splice sites.

    Assumes input table contains columns:
      - 'strand' (column 4)
      - 'motif'  (column 6)
    """
    df = pd.read_csv(filepath, sep="\t")

    required_cols = ["strand", "motif"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(
                f"Column '{col}' not found in {filepath}. "
                f"Found columns: {list(df.columns)}"
            )

    df["motif_standardized"] = df.apply(
        lambda row: reverse_motif_if_needed(row["motif"], row["strand"]),
        axis=1
    )

    df["splice_site"] = df["motif_standardized"].map(motif_to_splice_site)

    return df["splice_site"].dropna()


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Count splice-site motif combinations from canonical, U2/U12-like, "
            "and non-U2/U12-like motif tables, correcting strand=2 by reversing "
            "donor/acceptor order."
        )
    )
    parser.add_argument("canonical_file", help="Path to 2-canonical-motifs.txt")
    parser.add_argument("u2_u12_like_file", help="Path to 2-U2_U12_like.txt")
    parser.add_argument("non_u2_u12_like_file", help="Path to 2-non-U2_U12_like.txt")
    parser.add_argument("output_file", help="Output summary txt file")

    args = parser.parse_args()

    # Read and process all three files
    canonical_sites = read_and_process_file(args.canonical_file)
    u2_u12_like_sites = read_and_process_file(args.u2_u12_like_file)
    non_u2_u12_like_sites = read_and_process_file(args.non_u2_u12_like_file)

    # Combine all splice sites
    all_sites = pd.concat(
        [canonical_sites, u2_u12_like_sites, non_u2_u12_like_sites],
        ignore_index=True
    )

    # Count occurrences
    summary = all_sites.value_counts().reset_index()
    summary.columns = ["splice_site", "count"]

    # Calculate percentage
    total = summary["count"].sum()
    summary["percentage"] = summary["count"] / total * 100

    # Sort by count descending
    summary = summary.sort_values(by="count", ascending=False).reset_index(drop=True)

    # Format percentage
    summary["percentage"] = summary["percentage"].map(lambda x: f"{x:.6f}")

    # Write output
    summary.to_csv(args.output_file, sep="\t", index=False)

    print(f"Done. Output written to: {args.output_file}")
    print(f"Total splice sites counted: {total}")


if __name__ == "__main__":
    main()