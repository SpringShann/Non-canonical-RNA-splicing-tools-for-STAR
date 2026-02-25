import pandas as pd
import subprocess
import re

# File paths
gtf_path = "/genomic.gtf"
genome_fasta = "/GCF_000001635.27_GRCm39_genomic.fna"
sj_path = "/SRR23308049_SJ.out.tab"
output_path = "/2-noncanonical-motifs.txt"

# Read gene annotations from the GTF file, including strand
gene_records = []
with open(gtf_path, 'r') as f:
    for line in f:
        if line.startswith("#"):
            continue
        fields = line.strip().split('\t')
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

gtf_df = pd.DataFrame(gene_records, columns=["chr", "start", "end", "gene_id", "strand"])

# Reverse-complement function
def rev_comp(seq):
    complement = str.maketrans("ATCGN", "TAGCN")
    return seq.translate(complement)[::-1]

# Read splice junctions
colnames = ["chr", "start", "end", "strand", "motif_code", "annotation", "unique_reads", "multi_reads", "max_overhang"]
sj_df = pd.read_csv(sj_path, sep='\t', header=None, names=colnames)

# Optional: keep only motif_code == 0 (non-canonical junctions)
sj_df = sj_df[sj_df["motif_code"] == 0]

# Filter by read support
sj_df = sj_df[sj_df["unique_reads"] > 1]

# Sequence extraction function
def get_seq(chrom, pos1, pos2):
    region = f"{chrom}:{pos1}-{pos2}"
    cmd = ["samtools", "faidx", genome_fasta, region]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    seq_lines = result.stdout.strip().split('\n')[1:]
    return ''.join(seq_lines).upper()

# Main loop
records = []
for idx, row in sj_df.iterrows():
    chr_ = row['chr']
    start = int(row['start'])
    end = int(row['end'])
    strand = row['strand']  # 0, 1, 2
    unique_reads = row['unique_reads']
    multi_reads = row['multi_reads']

    # Find overlapping gene and strand
    genes = gtf_df[
        (gtf_df['chr'] == chr_) &
        (gtf_df['start'] <= start) &
        (gtf_df['end'] >= end)
    ]
    if not genes.empty:
        gene_id = genes.iloc[0]['gene_id']
        gene_strand = genes.iloc[0]['strand']  # "+" or "-"
    else:
        gene_id = "NA"
        gene_strand = "NA"

    # Infer strand from GTF if needed
    strand_source = "STAR"
    if strand == 0:
        if gene_strand == "+":
            strand = 1
            strand_source = "GTF"
        elif gene_strand == "-":
            strand = 2
            strand_source = "GTF"
        else:
            strand_source = "Unknown"

    # Extract splice sites and reverse-complement if needed
    try:
        donor = get_seq(chr_, start, start + 1)
        acceptor = get_seq(chr_, end - 1, end)

        if strand == 2:
            donor = rev_comp(donor)
            acceptor = rev_comp(acceptor)

        motif = f"{donor}/{acceptor}"
    except Exception as e:
        motif = "NA/NA"

    intron_len = end - start

    records.append([
        chr_, start, end, strand, strand_source, motif, intron_len, gene_id,
        unique_reads, multi_reads
    ])

# Build output DataFrame
out_df = pd.DataFrame(records, columns=[
    "chr", "start", "end", "strand", "strand_source", "motif", "intron_length",
    "gene_id", "unique_reads", "multi_reads"
])

# Sort output
out_df = out_df.sort_values(by=["intron_length", "unique_reads"], ascending=[True, False])

# Write output
out_df.to_csv(output_path, sep='\t', index=False)
print(f"✔ Output written: {output_path}")