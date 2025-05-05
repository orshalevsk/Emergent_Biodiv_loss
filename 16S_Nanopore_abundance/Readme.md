# Nanopore Demultiplexing & Analysis Pipeline

This repository contains a reproducible pipeline to process from Oxford Nanopore sequencing 16S data a relative abundance matrix. It performs quality filtering, demultiplexing, chimera removal, and alignment.

## Repository Contents

- `pipeline.sh` – Main Bash pipeline script for end-to-end processing.
- `make_count_table_updated.py` – Python script for generating OTU count tables.
- `config.env` – Configuration file to define user-specific paths and parameters.
- `rev_cmplt_minibar_output.py` – Script for reverse-complementing demultiplexed reads (if required).
- `README.md` – This file.

## Requirements

The pipeline uses multiple conda (https://docs.conda.io/en/latest/) environments to isolate tool dependencies.

Required tools/environments:
- `seqtk` / `seqkit`
- `nanofilt`
- `minibar`
- `vsearch`
- `minimap2`
- `samtools`
- Python 3.7+ with `pandas`

You can create the environments from scratch or export them from `environment.yml` files (not provided here but recommended).

## Input

1. **Raw FASTQ** file from Oxford Nanopore sequencing.
2. **Barcode CSV** file compatible with `minibar`.
3. **Reference FASTA** file for chimera filtering and read mapping.

## Usage

### 1. Configure your environment
Edit the `config.env` file with your paths and parameters. Example:

```bash
# config.env
FASTQ_FILE=/path/to/your/input.fastq.gz
SEQ_DIR=/path/to/fastq_splits
BARCODE_FILE=/path/to/barcodes.csv
OUTPUT_DIR=/path/to/output
MINIBAR_SCRIPT=/path/to/minibar.py
REV_COMP_SCRIPT=/path/to/rev_cmplt_minibar_output.py
REFERENCE_DB=/path/to/reference.fasta
SAMPLE_PREFIX=YourPrefix
Q_THRE=15
MIN_LEN=1500
MAX_LEN=1700
THREADS=50
FLIP_SUFFIX=_rev_complemented

### 2. Run the pipeline
# Modify paths in config.env as needed
bash pipeline.sh
