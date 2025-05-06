#!/bin/bash

set -e

# Define variables
DB_INDEX="filtered_database.mmi"
QUERY_DIR="fasta"
OUTPUT_DIR="sam_alignments"

# Create the output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Function to perform alignment for a single FASTA file
align_fasta() {
    local fasta_file="$1"
    local base_name
    base_name=$(basename "$fasta_file" .fasta)
    local sam_output="$OUTPUT_DIR/${base_name}.sam"

    echo "Aligning $fasta_file -> $sam_output"

    minimap2 -ax sr --secondary=no "$DB_INDEX" "$fasta_file" > "$sam_output"
}

export -f align_fasta  # Export the function for GNU Parallel
export DB_INDEX
export OUTPUT_DIR

# Find all .fasta files in the query directory
fasta_files=("$QUERY_DIR"/*.fasta)

# Check if there are any FASTA files to process
if [ ! -e "${fasta_files[0]}" ]; then
    echo "No .fasta files found in the '$QUERY_DIR' directory."
    exit 1
fi

# Determine the number of available CPU cores
if command -v nproc > /dev/null 2>&1; then
    THREADS=$(nproc)
elif [[ "$OSTYPE" == "darwin"* ]]; then
    THREADS=$(sysctl -n hw.ncpu)
else
    THREADS=4  # Default to 4 if OS is not recognized
fi

echo "Starting alignment using $THREADS threads..."

# Run alignments in parallel
parallel -j 50 align_fasta ::: "${fasta_files[@]}"

echo "All alignments completed successfully."
