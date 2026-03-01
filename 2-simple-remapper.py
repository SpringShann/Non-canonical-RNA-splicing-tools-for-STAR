#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import math
import random
import subprocess
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# =========================================================
# File paths (edit here)
# =========================================================
gtf_path = "input/genomic.gtf"
genome_fasta = "input/GCF_000001635.27_GRCm39_genomic.fna"
sj_path = "input/SRR23308049_SJ.out.tab"

canonical_output_path = "output/2-canonical-motifs.txt"
noncanonical_output_path = "output/2-noncanonical-motifs.txt"
noncanonical_scored_output_path = "output/2-noncanonical-scored.txt"
u2_u12_like_output_path = "output/2-U2_U12_like.txt"
non_u2_u12_output_path = "output/2-non-U2_U12_like.txt"
stats_output_path = "output/2-pwm_stats.txt"

donor_pwm_output_path = "output/2-donor_pwm.tsv"
acceptor_pwm_output_path = "output/2-acceptor_pwm.tsv"

# =========================================================
# Adjustable window settings
# Donor:
#   positive strand  -> [start-3, start+5]
#   negative strand  -> corresponding genomic window, then reverse-complement
# Acceptor:
#   positive strand  -> [end-19, end+3]
#   negative strand  -> corresponding genomic window, then reverse-complement
# =========================================================
donor_window_left = 3
donor_window_right = 5

acceptor_window_left = 19
acceptor_window_right = 3

# =========================================================
# Other settings
# =========================================================
min_unique_reads = 2              # keep junctions with unique_reads > 1
pseudocount = 0.5                 # PWM pseudocount
scramble_percentile = 95          # threshold = percentile of scrambled total scores
random_seed = 42                  # reproducible scrambling

BASES = ["A", "C", "G", "T"]
BASE_TO_IDX = {b: i for i, b in enumerate(BASES)}

random.seed(random_seed)
np.random.seed(random_seed)


# =========================================================
# Basic helpers
# =========================================================
def rev_comp(seq: str) -> str:
    tbl = str.maketrans("ACGTNacgtn", "TGCANtgcan")
    return seq.translate(tbl)[::-1]


def get_seq(chrom: str, pos1: int, pos2: int) -> str:
    """
    1-based inclusive genomic sequence fetch via samtools faidx.
    """
    region = f"{chrom}:{pos1}-{pos2}"
    cmd = ["samtools", "faidx", genome_fasta, region]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"samtools faidx failed for {region}: {result.stderr}")

    lines = result.stdout.strip().split("\n")
    seq = "".join(lines[1:]).upper()
    return seq


def safe_get_seq(chrom: str, pos1: int, pos2: int, expected_len: Optional[int] = None) -> Optional[str]:
    """
    Safe wrapper for sequence fetch.
    Returns None if coordinates are invalid or sequence length is unexpected.
    """
    if pos1 < 1 or pos2 < pos1:
        return None

    try:
        seq = get_seq(chrom, pos1, pos2)
    except Exception:
        return None

    if expected_len is not None and len(seq) != expected_len:
        return None

    if not re.fullmatch(r"[ACGTN]+", seq):
        return None

    return seq


def shuffle_seq(seq: str) -> str:
    """
    Randomly shuffle a sequence while preserving base composition.
    """
    chars = list(seq)
    random.shuffle(chars)
    return "".join(chars)


# =========================================================
# GTF loading
# =========================================================
def load_gtf_gene_intervals(gtf_fp: str) -> pd.DataFrame:
    gene_records = []

    with open(gtf_fp, "r") as f:
        for line in f:
            if line.startswith("#"):
                continue

            fields = line.strip().split("\t")
            if len(fields) < 9 or fields[2] != "gene":
                continue

            chrom = fields[0]
            start = int(fields[3])
            end = int(fields[4])
            strand = fields[6]  # "+" or "-"
            attributes = fields[8]

            match = re.search(r'gene_id "([^"]+)"', attributes)
            gene_id = match.group(1) if match else "NA"

            gene_records.append((chrom, start, end, gene_id, strand))

    return pd.DataFrame(gene_records, columns=["chr", "start", "end", "gene_id", "strand"])


def infer_gene_and_strand(
    gtf_df: pd.DataFrame,
    chrom: str,
    start: int,
    end: int
) -> Tuple[str, str]:
    """
    Find an overlapping gene and its strand.
    If multiple genes overlap, use the first one (same as the original script logic).
    """
    genes = gtf_df[
        (gtf_df["chr"] == chrom) &
        (gtf_df["start"] <= start) &
        (gtf_df["end"] >= end)
    ]

    if not genes.empty:
        gene_id = genes.iloc[0]["gene_id"]
        gene_strand = genes.iloc[0]["strand"]
    else:
        gene_id = "NA"
        gene_strand = "NA"

    return gene_id, gene_strand


def resolve_strand(star_strand: int, gene_strand: str) -> Tuple[int, str]:
    """
    Preserve the original remapper logic:
      - if STAR strand is 1 or 2, keep it
      - if STAR strand is 0, infer from GTF if possible
    """
    strand_source = "STAR"
    final_strand = star_strand

    if star_strand == 0:
        if gene_strand == "+":
            final_strand = 1
            strand_source = "GTF"
        elif gene_strand == "-":
            final_strand = 2
            strand_source = "GTF"
        else:
            final_strand = 0
            strand_source = "Unknown"

    return final_strand, strand_source


# =========================================================
# Original-format motif extraction
# (kept compatible with the original script output)
# =========================================================
def extract_simple_motif(chrom: str, start: int, end: int, strand: int) -> str:
    """
    Keep the original script logic for motif output:
      donor    = genome[start, start+1]
      acceptor = genome[end-1, end]
      if strand == 2, reverse-complement each side
    """
    try:
        donor = safe_get_seq(chrom, start, start + 1, expected_len=2)
        acceptor = safe_get_seq(chrom, end - 1, end, expected_len=2)

        if donor is None or acceptor is None:
            return "NA/NA"

        if strand == 2:
            donor = rev_comp(donor)
            acceptor = rev_comp(acceptor)

        return f"{donor}/{acceptor}"
    except Exception:
        return "NA/NA"


# =========================================================
# Transcript-oriented PWM windows
# =========================================================
def extract_donor_window(chrom: str, start: int, end: int, strand: int) -> Optional[str]:
    """
    Donor window in transcript 5'->3' orientation.

    Positive strand:
      genomic [start - donor_window_left, start + donor_window_right]

    Negative strand:
      donor is at the genomic right end.
      Extract the corresponding genomic window, then reverse-complement.
      genomic [end - donor_window_right, end + donor_window_left]
    """
    expected_len = donor_window_left + donor_window_right + 1

    if strand == 1:
        seq = safe_get_seq(
            chrom,
            start - donor_window_left,
            start + donor_window_right,
            expected_len=expected_len
        )
        return seq

    if strand == 2:
        seq = safe_get_seq(
            chrom,
            end - donor_window_right,
            end + donor_window_left,
            expected_len=expected_len
        )
        if seq is None:
            return None
        return rev_comp(seq)

    return None


def extract_acceptor_window(chrom: str, start: int, end: int, strand: int) -> Optional[str]:
    """
    Acceptor window in transcript 5'->3' orientation.

    Positive strand:
      genomic [end - acceptor_window_left, end + acceptor_window_right]

    Negative strand:
      acceptor is at the genomic left end.
      Extract the corresponding genomic window, then reverse-complement.
      genomic [start - acceptor_window_right, start + acceptor_window_left]
    """
    expected_len = acceptor_window_left + acceptor_window_right + 1

    if strand == 1:
        seq = safe_get_seq(
            chrom,
            end - acceptor_window_left,
            end + acceptor_window_right,
            expected_len=expected_len
        )
        return seq

    if strand == 2:
        seq = safe_get_seq(
            chrom,
            start - acceptor_window_right,
            start + acceptor_window_left,
            expected_len=expected_len
        )
        if seq is None:
            return None
        return rev_comp(seq)

    return None


# =========================================================
# PWM functions
# =========================================================
def build_pwm(seqs: List[str], pseudocount_value: float = 0.5) -> Optional[np.ndarray]:
    """
    Build a log2-odds PWM from equal-length sequences.
    Background is assumed to be uniform (0.25 for each base).
    """
    if not seqs:
        return None

    seq_len = len(seqs[0])

    for s in seqs:
        if len(s) != seq_len:
            raise ValueError("All sequences must have the same length.")
        if not re.fullmatch(r"[ACGT]+", s):
            raise ValueError("PWM training sequences must contain only A/C/G/T.")

    counts = np.zeros((4, seq_len), dtype=float)

    for s in seqs:
        for i, base in enumerate(s):
            counts[BASE_TO_IDX[base], i] += 1.0

    freqs = (counts + pseudocount_value) / (len(seqs) + 4.0 * pseudocount_value)

    bg = 0.25
    pwm = np.log2(freqs / bg)

    return pwm


def score_seq(seq: str, pwm: np.ndarray) -> Optional[float]:
    """
    Score a sequence against a PWM.
    Returns None if the sequence contains non-ACGT bases or length mismatch.
    """
    if seq is None:
        return None

    if len(seq) != pwm.shape[1]:
        return None

    if not re.fullmatch(r"[ACGT]+", seq):
        return None

    score = 0.0
    for i, base in enumerate(seq):
        score += float(pwm[BASE_TO_IDX[base], i])

    return score


def save_pwm_tsv(path: str, pwm: np.ndarray) -> None:
    """
    Save a PWM as a TSV table with bases in rows and positions in columns.
    """
    if pwm is None:
        return

    with open(path, "w") as f:
        header = ["base"] + [f"pos_{i+1}" for i in range(pwm.shape[1])]
        f.write("\t".join(header) + "\n")

        for base in BASES:
            row = [base] + [f"{pwm[BASE_TO_IDX[base], j]:.6f}" for j in range(pwm.shape[1])]
            f.write("\t".join(row) + "\n")


# =========================================================
# Junction table building
# =========================================================
def build_junction_records(gtf_df: pd.DataFrame, sj_df: pd.DataFrame) -> List[Dict]:
    """
    Build a unified record list for all filtered junctions.
    """
    records = []

    for _, row in sj_df.iterrows():
        chrom = row["chr"]
        start = int(row["start"])
        end = int(row["end"])
        star_strand = int(row["strand"])
        motif_code = int(row["motif_code"])
        annotation = int(row["annotation"])
        unique_reads = int(row["unique_reads"])
        multi_reads = int(row["multi_reads"])
        max_overhang = int(row["max_overhang"])

        gene_id, gene_strand = infer_gene_and_strand(gtf_df, chrom, start, end)
        final_strand, strand_source = resolve_strand(star_strand, gene_strand)

        motif = extract_simple_motif(chrom, start, end, final_strand)
        intron_length = end - start

        record = {
            "chr": chrom,
            "start": start,
            "end": end,
            "strand": final_strand,
            "strand_source": strand_source,
            "motif": motif,
            "intron_length": intron_length,
            "gene_id": gene_id,
            "unique_reads": unique_reads,
            "multi_reads": multi_reads,
            # internal fields
            "motif_code": motif_code,
            "annotation": annotation,
            "max_overhang": max_overhang,
        }

        records.append(record)

    return records


def to_original_format_df(records: List[Dict]) -> pd.DataFrame:
    """
    Convert records to the original output format.
    """
    cols = [
        "chr", "start", "end", "strand", "strand_source", "motif", "intron_length",
        "gene_id", "unique_reads", "multi_reads"
    ]

    df = pd.DataFrame([{c: r[c] for c in cols} for r in records])
    if not df.empty:
        df = df.sort_values(by=["intron_length", "unique_reads"], ascending=[True, False])

    return df


# =========================================================
# Main workflow
# =========================================================
def main():
    print("[INFO] Loading GTF...")
    gtf_df = load_gtf_gene_intervals(gtf_path)

    print("[INFO] Reading SJ.out.tab...")
    colnames = [
        "chr", "start", "end", "strand", "motif_code", "annotation",
        "unique_reads", "multi_reads", "max_overhang"
    ]
    sj_df = pd.read_csv(sj_path, sep="\t", header=None, names=colnames)

    # Global read-support filter
    sj_df = sj_df[sj_df["unique_reads"] >= min_unique_reads].copy()

    print(f"[INFO] Junctions after unique_reads >= {min_unique_reads}: {len(sj_df)}")

    # Build unified record list
    all_records = build_junction_records(gtf_df, sj_df)

    # Split canonical / non-canonical
    canonical_records = [r for r in all_records if r["motif_code"] != 0]
    noncanonical_records = [r for r in all_records if r["motif_code"] == 0]

    # Step 1: output canonical list
    canonical_df = to_original_format_df(canonical_records)
    canonical_df.to_csv(canonical_output_path, sep="\t", index=False)
    print(f"[OK] Canonical junctions written: {canonical_output_path}")

    # Step 3: output all non-canonical list
    noncanonical_df = to_original_format_df(noncanonical_records)
    noncanonical_df.to_csv(noncanonical_output_path, sep="\t", index=False)
    print(f"[OK] Non-canonical junctions written: {noncanonical_output_path}")

    # Step 2: build PWM training windows from canonical junctions
    canonical_training_records = []
    canonical_skipped_unresolved_strand = 0
    canonical_skipped_bad_window = 0

    donor_train_seqs = []
    acceptor_train_seqs = []

    for r in canonical_records:
        if r["strand"] not in (1, 2):
            canonical_skipped_unresolved_strand += 1
            continue

        donor_window = extract_donor_window(r["chr"], r["start"], r["end"], r["strand"])
        acceptor_window = extract_acceptor_window(r["chr"], r["start"], r["end"], r["strand"])

        if donor_window is None or acceptor_window is None:
            canonical_skipped_bad_window += 1
            continue

        if not re.fullmatch(r"[ACGT]+", donor_window):
            canonical_skipped_bad_window += 1
            continue

        if not re.fullmatch(r"[ACGT]+", acceptor_window):
            canonical_skipped_bad_window += 1
            continue

        donor_train_seqs.append(donor_window)
        acceptor_train_seqs.append(acceptor_window)
        canonical_training_records.append(r)

    if not donor_train_seqs or not acceptor_train_seqs:
        raise RuntimeError("No valid canonical windows available for PWM training.")

    donor_pwm = build_pwm(donor_train_seqs, pseudocount_value=pseudocount)
    acceptor_pwm = build_pwm(acceptor_train_seqs, pseudocount_value=pseudocount)

    save_pwm_tsv(donor_pwm_output_path, donor_pwm)
    save_pwm_tsv(acceptor_pwm_output_path, acceptor_pwm)

    print(f"[OK] Donor PWM written: {donor_pwm_output_path}")
    print(f"[OK] Acceptor PWM written: {acceptor_pwm_output_path}")

    # Step 3 + 4: extract non-canonical windows and score them
    scored_noncanonical_rows = []
    scoring_skipped_unresolved_strand = 0
    scoring_skipped_bad_window = 0
    scoring_skipped_non_acgt = 0

    scrambled_scores = []

    for r in noncanonical_records:
        donor_window = None
        acceptor_window = None
        donor_score = None
        acceptor_score = None
        total_score = None
        classification = None
        score_status = "NOT_SCORED"

        if r["strand"] not in (1, 2):
            scoring_skipped_unresolved_strand += 1
            score_status = "SKIPPED_UNRESOLVED_STRAND"
        else:
            donor_window = extract_donor_window(r["chr"], r["start"], r["end"], r["strand"])
            acceptor_window = extract_acceptor_window(r["chr"], r["start"], r["end"], r["strand"])

            if donor_window is None or acceptor_window is None:
                scoring_skipped_bad_window += 1
                score_status = "SKIPPED_BAD_WINDOW"
            elif not re.fullmatch(r"[ACGT]+", donor_window) or not re.fullmatch(r"[ACGT]+", acceptor_window):
                scoring_skipped_non_acgt += 1
                score_status = "SKIPPED_NON_ACGT_WINDOW"
            else:
                donor_score = score_seq(donor_window, donor_pwm)
                acceptor_score = score_seq(acceptor_window, acceptor_pwm)

                if donor_score is None or acceptor_score is None:
                    scoring_skipped_non_acgt += 1
                    score_status = "SKIPPED_NON_ACGT_WINDOW"
                else:
                    total_score = donor_score + acceptor_score
                    score_status = "SCORED"

                    # Step 5: scrambled control for thresholding
                    donor_scrambled = shuffle_seq(donor_window)
                    acceptor_scrambled = shuffle_seq(acceptor_window)

                    donor_scrambled_score = score_seq(donor_scrambled, donor_pwm)
                    acceptor_scrambled_score = score_seq(acceptor_scrambled, acceptor_pwm)

                    if donor_scrambled_score is not None and acceptor_scrambled_score is not None:
                        scrambled_scores.append(donor_scrambled_score + acceptor_scrambled_score)

        scored_noncanonical_rows.append({
            "chr": r["chr"],
            "start": r["start"],
            "end": r["end"],
            "strand": r["strand"],
            "strand_source": r["strand_source"],
            "motif": r["motif"],
            "intron_length": r["intron_length"],
            "gene_id": r["gene_id"],
            "unique_reads": r["unique_reads"],
            "multi_reads": r["multi_reads"],
            "donor_window": donor_window if donor_window is not None else "NA",
            "acceptor_window": acceptor_window if acceptor_window is not None else "NA",
            "donor_score": donor_score,
            "acceptor_score": acceptor_score,
            "total_score": total_score,
            "score_status": score_status,
            "classification": classification,
        })

    if not scrambled_scores:
        raise RuntimeError("No scrambled scores were generated. Cannot define a threshold.")

    threshold = float(np.percentile(scrambled_scores, scramble_percentile))

    # Step 6: classify scored non-canonical junctions
    u2_like_records = []
    non_u2_like_records = []

    for row in scored_noncanonical_rows:
        if row["score_status"] == "SCORED" and row["total_score"] is not None:
            if row["total_score"] > threshold:
                row["classification"] = "U2_U12_like"
                u2_like_records.append({
                    "chr": row["chr"],
                    "start": row["start"],
                    "end": row["end"],
                    "strand": row["strand"],
                    "strand_source": row["strand_source"],
                    "motif": row["motif"],
                    "intron_length": row["intron_length"],
                    "gene_id": row["gene_id"],
                    "unique_reads": row["unique_reads"],
                    "multi_reads": row["multi_reads"],
                })
            else:
                row["classification"] = "non_U2_U12_like"
                non_u2_like_records.append({
                    "chr": row["chr"],
                    "start": row["start"],
                    "end": row["end"],
                    "strand": row["strand"],
                    "strand_source": row["strand_source"],
                    "motif": row["motif"],
                    "intron_length": row["intron_length"],
                    "gene_id": row["gene_id"],
                    "unique_reads": row["unique_reads"],
                    "multi_reads": row["multi_reads"],
                })
        else:
            row["classification"] = "NOT_CLASSIFIED"

    # Write scored non-canonical master table
    scored_df = pd.DataFrame(scored_noncanonical_rows)
    if not scored_df.empty:
        scored_df = scored_df.sort_values(by=["intron_length", "unique_reads"], ascending=[True, False])
    scored_df.to_csv(noncanonical_scored_output_path, sep="\t", index=False)
    print(f"[OK] Scored non-canonical table written: {noncanonical_scored_output_path}")

    # Write classified files in the same format as step 1
    u2_df = to_original_format_df(u2_like_records)
    non_u2_df = to_original_format_df(non_u2_like_records)

    u2_df.to_csv(u2_u12_like_output_path, sep="\t", index=False)
    non_u2_df.to_csv(non_u2_u12_output_path, sep="\t", index=False)

    print(f"[OK] U2/U12-like junctions written: {u2_u12_like_output_path}")
    print(f"[OK] non-U2/U12-like junctions written: {non_u2_u12_output_path}")

    # Stats file
    with open(stats_output_path, "w") as f:
        f.write("== Input files ==\n")
        f.write(f"gtf_path\t{gtf_path}\n")
        f.write(f"genome_fasta\t{genome_fasta}\n")
        f.write(f"sj_path\t{sj_path}\n\n")

        f.write("== Window settings ==\n")
        f.write(f"donor_window_positive\t[start-{donor_window_left}, start+{donor_window_right}]\n")
        f.write(f"acceptor_window_positive\t[end-{acceptor_window_left}, end+{acceptor_window_right}]\n\n")

        f.write("== Filters and scoring settings ==\n")
        f.write(f"min_unique_reads\t{min_unique_reads}\n")
        f.write(f"pseudocount\t{pseudocount}\n")
        f.write(f"scramble_percentile\t{scramble_percentile}\n")
        f.write(f"random_seed\t{random_seed}\n\n")

        f.write("== Junction counts ==\n")
        f.write(f"total_after_unique_read_filter\t{len(all_records)}\n")
        f.write(f"canonical_total\t{len(canonical_records)}\n")
        f.write(f"noncanonical_total\t{len(noncanonical_records)}\n\n")

        f.write("== Canonical PWM training ==\n")
        f.write(f"canonical_used_for_pwm\t{len(canonical_training_records)}\n")
        f.write(f"canonical_skipped_unresolved_strand\t{canonical_skipped_unresolved_strand}\n")
        f.write(f"canonical_skipped_bad_window\t{canonical_skipped_bad_window}\n")
        f.write(f"donor_training_window_length\t{len(donor_train_seqs[0]) if donor_train_seqs else 0}\n")
        f.write(f"acceptor_training_window_length\t{len(acceptor_train_seqs[0]) if acceptor_train_seqs else 0}\n\n")

        f.write("== Non-canonical scoring ==\n")
        f.write(f"noncanonical_scored\t{sum(1 for x in scored_noncanonical_rows if x['score_status'] == 'SCORED')}\n")
        f.write(f"noncanonical_skipped_unresolved_strand\t{scoring_skipped_unresolved_strand}\n")
        f.write(f"noncanonical_skipped_bad_window\t{scoring_skipped_bad_window}\n")
        f.write(f"noncanonical_skipped_non_acgt_window\t{scoring_skipped_non_acgt}\n\n")

        f.write("== Scrambled control ==\n")
        f.write(f"scrambled_scores_n\t{len(scrambled_scores)}\n")
        f.write(f"total_score_threshold_percentile\t{scramble_percentile}\n")
        f.write(f"total_score_threshold_value\t{threshold:.6f}\n\n")

        f.write("== Final classification ==\n")
        f.write(f"U2_U12_like\t{len(u2_like_records)}\n")
        f.write(f"non_U2_U12_like\t{len(non_u2_like_records)}\n")
        f.write(f"not_classified\t{sum(1 for x in scored_noncanonical_rows if x['classification'] == 'NOT_CLASSIFIED')}\n")

    print(f"[OK] Stats written: {stats_output_path}")
    print("[DONE] Workflow completed.")


if __name__ == "__main__":
    main()
