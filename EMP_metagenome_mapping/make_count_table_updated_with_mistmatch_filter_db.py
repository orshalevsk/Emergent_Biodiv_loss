#!/usr/bin/env python3

import os
import sys
import csv
import argparse
import re
from collections import defaultdict
from Bio import SeqIO
from concurrent.futures import ProcessPoolExecutor, as_completed

def parse_fasta_headers(fasta_file):
    """
    Extract contig headers and map them to their respective genome groups.
    
    Parameters:
    - fasta_file: Path to the database FASTA file.
    
    Returns:
    - contig_to_group: Dictionary mapping contig names to their genome groups.
    - groups: Set of unique genome groups.
    """
    contig_to_group = {}
    groups = set()
    
    with open(fasta_file, 'r') as f:
        for record in SeqIO.parse(f, 'fasta'):
            contig_id = record.id.strip()
            # Split the contig_id at '_>' and take the first part as the genome group
            if '_>' in contig_id:
                group = contig_id.split('_>')[0]
            else:
                # If '_>' not found, assign the entire contig_id as the group
                group = contig_id
            contig_to_group[contig_id] = group
            groups.add(group)
    
    return contig_to_group, groups

def count_matches_in_sam(sam_file, contig_to_group, groups, min_mapq=0, min_mismatch_ratio=0):
    """
    Count the number of matches to each genome group and unknowns in a SAM file,
    applying filters for MAPQ and mismatches proportion.
    
    Parameters:
    - sam_file: Path to the SAM file.
    - contig_to_group: Dictionary mapping contig names to their genome groups.
    - groups: Set of unique genome groups.
    - min_mapq: Minimum MAPQ score to consider an alignment.
    - min_mismatch_ratio: Minimum alignment length to mismatches ratio.
    
    Returns:
    - counts: Dictionary with group names and 'unknown' as keys and their respective counts.
    """
    counts = defaultdict(int)
    with open(sam_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            if line.startswith('@'):
                continue  # Skip header lines
            fields = line.strip().split('\t')
            if len(fields) < 11:
                print(f"Warning: Line {line_num} in {sam_file} is malformed.")
                continue  # Invalid SAM line
            flag = int(fields[1])
            mapq = int(fields[4])
            rname = fields[2].strip()
            cigar = fields[5]

            if flag & 0x4:
                # Read is unmapped
                counts['unknown'] += 1
                continue

            if mapq < min_mapq:
                # Below minimum MAPQ threshold
                counts['unknown'] += 1
                continue

            # Extract the NM tag (number of mismatches)
            nm = None
            for field in fields[11:]:
                if field.startswith('NM:i:'):
                    nm = int(field[5:])
                    break

            if nm is None:
                print(f"Warning: NM tag not found in line {line_num} in {sam_file}. Skipping alignment.")
                counts['unknown'] += 1
                continue

            # Compute alignment length from CIGAR string
            # Only consider 'M', '=', 'X' operations
            cigar_tuples = re.findall(r'(\d+)([MIDNSHP=X])', cigar)
            alignment_length = 0
            for length, op in cigar_tuples:
                length = int(length)
                if op in ('M', '=', 'X'):
                    alignment_length += length

            if nm == 0:
                mismatch_ratio = float('inf')
            else:
                mismatch_ratio = alignment_length / nm

            if mismatch_ratio < min_mismatch_ratio:
                # Below minimum mismatches proportion threshold
                counts['unknown'] += 1
                continue

            if rname in contig_to_group:
                group = contig_to_group[rname]
                counts[group] += 1
            else:
                # Read is mapped to a reference not in the database
                counts['unknown'] += 1
    return counts

def main():
    parser = argparse.ArgumentParser(description='Generate a grouped counts table from SAM alignments.')
    parser.add_argument('-d', '--database', required=True, help='Path to the database FASTA file (e.g., simplified_pilon.fasta)')
    parser.add_argument('-s', '--sam_dir', required=True, help='Directory containing SAM files')
    parser.add_argument('-o', '--output', default='table_counts.csv', help='Output CSV file name')
    parser.add_argument('--min_mapq', type=int, default=0, help='Minimum MAPQ score to consider an alignment (default: 0)')
    parser.add_argument('--min_mismatch_ratio', type=float, default=0, help='Minimum alignment length to mismatches ratio to consider an alignment (default: 0)')
    parser.add_argument('--threads', type=int, default=1, help='Number of threads/processes to use (default: 1)')
    args = parser.parse_args()
    
    db_fasta = args.database
    sam_dir = args.sam_dir
    output_csv = args.output
    min_mapq = args.min_mapq
    min_mismatch_ratio = args.min_mismatch_ratio
    num_threads = args.threads
    
    # Verify that the database FASTA file exists
    if not os.path.isfile(db_fasta):
        print(f"Error: Database FASTA file '{db_fasta}' not found.")
        sys.exit(1)
    
    # Verify that the SAM directory exists
    if not os.path.isdir(sam_dir):
        print(f"Error: SAM directory '{sam_dir}' not found.")
        sys.exit(1)
    
    # Parse contig headers and map to groups
    print("Parsing contig headers from the database...")
    contig_to_group, groups = parse_fasta_headers(db_fasta)
    print(f"Found {len(groups)} unique genome groups in the database.")
    
    # Find all SAM files in the SAM directory
    sam_files = [f for f in os.listdir(sam_dir) if f.endswith('.sam')]
    if not sam_files:
        print(f"No SAM files found in directory '{sam_dir}'.")
        sys.exit(1)
    
    print(f"Found {len(sam_files)} SAM files to process.")
    
    # Prepare CSV headers
    sorted_groups = sorted(groups)
    headers = ['sample'] + sorted_groups + ['unknown']
    
    # Initialize a dictionary to hold counts for each sample
    counts_dict = {}
    
    # Process SAM files in parallel
    print(f"Processing SAM files using {num_threads} threads...")
    with ProcessPoolExecutor(max_workers=num_threads) as executor:
        future_to_sample = {}
        for sam_file in sam_files:
            sample_name = os.path.splitext(sam_file)[0]
            sam_path = os.path.join(sam_dir, sam_file)
            future = executor.submit(
                count_matches_in_sam,
                sam_path,
                contig_to_group,
                groups,
                min_mapq,
                min_mismatch_ratio
            )
            future_to_sample[future] = sample_name
        
        for future in as_completed(future_to_sample):
            sample_name = future_to_sample[future]
            try:
                counts = future.result()
                counts_dict[sample_name] = counts
                print(f"Finished processing sample: {sample_name}")
            except Exception as exc:
                print(f"Sample {sample_name} generated an exception: {exc}")
    
    # Write counts to the CSV file
    print(f"Writing counts to '{output_csv}'...")
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()
        for sample in sorted(counts_dict.keys()):
            counts = counts_dict[sample]
            row = {'sample': sample}
            for group in sorted_groups:
                row[group] = counts.get(group, 0)
            row['unknown'] = counts.get('unknown', 0)
            writer.writerow(row)
    
    print(f"Counts table successfully saved to '{output_csv}'.")

if __name__ == '__main__':
    main()

