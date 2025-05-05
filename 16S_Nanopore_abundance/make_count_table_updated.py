import os
import re
import glob
import argparse
import pandas as pd

def extract_sample_name(filename):
    match = re.search(r'OS\d+_(.+?)_filtered_combined_rev_complemented_nonchimeras\.bam\.count$', filename)
    if match:
        return match.group(1)
    return None

def main(input_dir, output_file):
    # Initialize an empty DataFrame to store the counts
    otu_table = pd.DataFrame()

    # Define the search pattern
    file_pattern = os.path.join(input_dir, "OS20241206_*_filtered_combined_rev_complemented_nonchimeras.bam.count")

    count_files = sorted(glob.glob(file_pattern))
    if not count_files:
        print(f"No count files found in {input_dir}")
        return

    print(f"Found {len(count_files)} count files...")

    for count_file in count_files:
        sample_name = extract_sample_name(os.path.basename(count_file))
        if not sample_name:
            print(f"Warning: Could not parse sample name from {count_file}")
            continue

        # Load the data
        df = pd.read_csv(
            count_file,
            sep='\t',
            header=None,
            usecols=[0, 2, 3],
            index_col=0,
            names=['species_name', 'mapped_read', 'unmapped_read']
        )

        # Set unmapped as total if '*'
        df.loc['*', 'mapped_read'] = df.loc['*', 'unmapped_read']
        df = df[['mapped_read']].T
        df.index = [sample_name]

        otu_table = pd.concat([otu_table, df]) if not otu_table.empty else df

    # Fill missing values with 0
    otu_table.fillna(0, inplace=True)
    otu_table.to_csv(output_file, index=True)
    print(f"OTU count table saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate OTU count table from individual BAM count files.")
    parser.add_argument("--input_dir", type=str, default="./analysis", help="Directory containing *.count files")
    parser.add_argument("--output_file", type=str, default="otu_count_table.csv", help="Output CSV file name")

    args = parser.parse_args()
    main(args.input_dir, args.output_file)
