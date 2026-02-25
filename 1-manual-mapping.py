#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Non-canonical splice junction validator
---------------------------------------
Functions:
1) Read non-canonical splice junctions from STAR SJ.out.tab
2) Filter: unique_reads > 1 and multi_reads == 0
3) Infer strand from GTF; if negative, reverse-complement sites and context
4) Output 8 columns, then one context line with 10 bp flanks in transcript 5'->3'
5) If BAM is provided, try to recover one supporting read name; otherwise use JUNC_chr_start_end

Author: You
"""

import os
import re
import sys
import subprocess
import pandas as pd

# ======== Settings ========
gtf_path       = "/genomic.gtf"
genome_fasta   = "/GCF_000001635.27_GRCm39_genomic.fna"
sj_path        = "/SRR23308049_SJ.out.tab"
# Optional: aligned BAM for read-name lookup; set to None if unavailable
bam_path       = None  # e.g. "/SRR23308049_Aligned.sortedByCoord.out.bam"
output_path    = "/noncanonical_junction_validation.tsv"
context_flank  = 10  # Flank size, including the site

# ======== Functions ========
def rev_comp(seq: str) -> str:
    tbl = str.maketrans("ACGTNacgtn", "TGCANtgcan")
    return seq.translate(tbl)[::-1]

def run_cmd(cmd):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{p.stderr}")
    return p.stdout

def faidx(chrom: str, start1: int, end1: int) -> str:
    """
    samtools faidx uses 1-based inclusive coordinates
    """
    region = f"{chrom}:{start1}-{end1}"
    out = run_cmd(["samtools", "faidx", genome_fasta, region])
    seq = "".join(out.strip().splitlines()[1:]).upper()
    return seq

def load_gtf_gene_intervals(gtf_fp: str) -> pd.DataFrame:
    records = []
    with open(gtf_fp, 'r') as f:
        for line in f:
            if not line or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 9: 
                continue
            if fields[2] != "gene":
                continue
            chrom  = fields[0]
            start  = int(fields[3])
            end    = int(fields[4])
            strand = fields[6]  # '+' or '-'
            attr   = fields[8]
            m = re.search(r'gene_id "([^"]+)"', attr)
            gene_id = m.group(1) if m else "NA"
            records.append((chrom, start, end, gene_id, strand))
    df = pd.DataFrame(records, columns=["chr","start","end","gene_id","strand"])
    return df

def infer_strand_from_gtf(gtf_df: pd.DataFrame, chrom: str, intron_start: int, intron_end: int) -> str:
    """
    Find a gene covering the intron in GTF and return '+' / '-' / 'NA'
    """
    sub = gtf_df[(gtf_df["chr"]==chrom) & (gtf_df["start"]<=intron_start) & (gtf_df["end"]>=intron_end)]
    if sub.empty:
        return "NA", "NA"
    # If multiple genes match, use the smallest span
    sub = sub.assign(span=sub["end"]-sub["start"]).sort_values("span")
    row = sub.iloc[0]
    return row["strand"], row["gene_id"]

def donor_acceptor_bases(chrom: str, start1: int, end1: int, strand_sym: str):
    """
    Return donor/acceptor 2-mers in transcript 5'->3' orientation.
    Positive strand: donor=[start,start+1], acceptor=[end-1,end]
    Negative strand: donor=rc([end-1,end]), acceptor=rc([start,start+1])
    """
    donor_gen    = faidx(chrom, start1,   start1+1)  # Genomic forward
    acceptor_gen = faidx(chrom, end1-1,   end1)

    if strand_sym == "-":
        donor_tx    = rev_comp(acceptor_gen)  # Right end is donor on negative strand
        acceptor_tx = rev_comp(donor_gen)     # Left end is acceptor on negative strand
        return donor_tx, acceptor_tx
    else:
        return donor_gen, acceptor_gen



def context_windows(chrom: str, start1: int, end1: int, strand_sym: str, flank: int=10):
    """
    Return two windows in transcript 5'->3':
      donor_up: upstream flank + [donor 2bp]
      acceptor_down: [acceptor 2bp] + downstream flank

    Positive strand:
    upstream = left of start1; downstream = right of end1
    Negative strand:
    upstream = right of end1 after RC; downstream = left of start1 after RC
    """
    # Site dinucleotides in genomic forward orientation
    donor_site_gen    = faidx(chrom, start1,   start1+1)   # Left end
    acceptor_site_gen = faidx(chrom, end1-1,   end1)       # Right end

    if strand_sym == "-":
        # Build in transcript orientation for negative strand
        donor_site_tx    = rev_comp(acceptor_site_gen)     # Right end -> donor
        acceptor_site_tx = rev_comp(donor_site_gen)        # Left end -> acceptor

        # donor upstream = genomic right of end; acceptor downstream = genomic left of start
        d_up_tx   = rev_comp(faidx(chrom, end1+1,       end1+flank)       if flank>0 else "")
        a_down_tx = rev_comp(faidx(chrom, max(1, start1-flank), start1-1) if flank>0 else "")

        donor_up      = f"{d_up_tx}[{donor_site_tx}]"
        acceptor_down = f"[{acceptor_site_tx}]{a_down_tx}"

    else:
        # Positive strand uses genomic orientation directly
        donor_site_tx    = donor_site_gen
        acceptor_site_tx = acceptor_site_gen

        d_up   = faidx(chrom, max(1, start1-flank), start1-1) if flank>0 else ""
        a_down = faidx(chrom, end1+1, end1+flank)             if flank>0 else ""

        donor_up      = f"{d_up}[{donor_site_tx}]"
        acceptor_down = f"[{acceptor_site_tx}]{a_down}"

    return donor_up, acceptor_down





def load_sj(sj_fp: str) -> pd.DataFrame:
    cols = ["chr","start","end","strand_code","motif_code","annotation","unique_reads","multi_reads","max_overhang"]
    df = pd.read_csv(sj_fp, sep="\t", header=None, names=cols)
    return df

def star_strand_to_symbol(code: int) -> str:
    # STAR: 0=undefined, 1='+', 2='-'
    return "+" if code==1 else "-" if code==2 else "."

def find_one_supporting_read_name_from_bam(bam_fp: str, chrom: str, start1: int, end1: int) -> str:
    """
    Try to recover one read that exactly supports the junction.
    Strategy: scan reads in [start1-1, end1+1], parse CIGAR 'N', and match the intron span.
    Return the first read name if found; otherwise return a placeholder ID.
    """
    try:
        region = f"{chrom}:{max(1,start1-1)}-{end1+1}"
        it = run_cmd(["samtools", "view", bam_fp, region]).splitlines()
        for line in it:
            f = line.split("\t")
            if len(f) < 6: 
                continue
            qname, rname, pos, cigar = f[0], f[2], int(f[3]), f[5]
            if rname != chrom or cigar == "*" or "N" not in cigar:
                continue
            # Parse CIGAR and track reference positions
            ref_cursor = pos
            # Parse operations like 55M, 3526N, 5M
            for num, op in re.findall(r"(\d+)([MIDNSHP=X])", cigar):
                l = int(num)
                if op == "N":
                    intron_start = ref_cursor           # First intron base after previous aligned block
                    intron_end   = ref_cursor + l - 1   # Intron span on reference
                    if intron_start == start1 and intron_end == end1:
                        return qname
                    ref_cursor += l
                elif op in "MDN=X":   # These consume reference
                    ref_cursor += l
                else:
                    # I,S,H,P do not consume reference
                    pass
        return f"JUNC_{chrom}_{start1}_{end1}"
    except Exception:
        return f"JUNC_{chrom}_{start1}_{end1}"

# ======== Main ========
def main():
    # 1) Load GTF gene annotations
    gtf_df = load_gtf_gene_intervals(gtf_path)

    # 2) Load SJ and filter: non-canonical, unique>1, multi==0
    sj = load_sj(sj_path)
    sj = sj[(sj["motif_code"]==0) & (sj["unique_reads"]>1) & (sj["multi_reads"]==0)].copy()

    # 3) Iterate and write output
    out_lines = []
    for _, r in sj.iterrows():
        chrom   = r["chr"]
        start1  = int(r["start"])
        end1    = int(r["end"])
        ureads  = int(r["unique_reads"])
        mreads  = int(r["multi_reads"])
        intron_len = end1 - start1

        # 3.1 Determine strand
        star_sym = star_strand_to_symbol(int(r["strand_code"]))
        gene_strand, gene_id = infer_strand_from_gtf(gtf_df, chrom, start1, end1)
        if star_sym in ["+", "-"]:
            strand_sym = star_sym
            strand_source = "STAR"
        else:
            strand_sym = gene_strand if gene_strand in ["+","-"] else "."
            strand_source = "GTF" if strand_sym in ["+","-"] else "Unknown"

        # 3.2 Get splice-site 2-mers in final strand orientation
        try:
            donor, acceptor = donor_acceptor_bases(chrom, start1, end1, strand_sym if strand_sym in ["+","-"] else "+")
            motif = f"{donor}/{acceptor}"
        except Exception:
            motif = "NA/NA"

        # 3.3 Get read name from BAM if available; otherwise use placeholder
        if bam_path and os.path.exists(bam_path):
            read_name = find_one_supporting_read_name_from_bam(bam_path, chrom, start1, end1)
        else:
            read_name = f"JUNC_{chrom}_{start1}_{end1}"

        # 3.4 Get flanking context in final strand orientation
        try:
            donor_ctx, acceptor_ctx = context_windows(chrom, start1, end1, strand_sym if strand_sym in ["+","-"] else "+", context_flank)
        except Exception:
            donor_ctx, acceptor_ctx = "NA", "NA"

        # 4) Write two lines: one data line and one context line
        strand_out = strand_sym if strand_sym in ["+","-"] else "."
        out_fields = [
            read_name,
            str(start1),
            str(end1),
            strand_out,
            motif,
            gene_id if gene_id else "NA",
            str(intron_len),
            str(ureads),
        ]
        out_lines.append("\t".join(out_fields))
        out_lines.append(
             f"#context\tdonor_up({context_flank}bp):\t{donor_ctx}\tacceptor_down({context_flank}bp):\t{acceptor_ctx}\t[{strand_source}]"
        )

    # 5) Write output
    with open(output_path, "w") as fw:
        fw.write("\n".join(out_lines) + "\n")

    print(f"✔ Done: {output_path}")
    print("Note: the first column is from BAM lookup; without BAM, it is a placeholder ID (JUNC_chr_start_end).")

if __name__ == "__main__":
    main()