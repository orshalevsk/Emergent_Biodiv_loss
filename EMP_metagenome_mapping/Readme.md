# EMP Metagenomic Mapping Pipeline

This repository contains a modular and parallelized pipeline for processing **metagenomic sequencing reads from the Earth Microbiome Project (EMP)** by mapping them to a **custom database of curated microbial genomes**. It performs quality control, demultiplexing, chimera filtering, alignment, and summarization into a **relative abundance table**.


## Features

- Accepts raw Nanopore FASTQ files (e.g., from EMP)  
- Quality filters and length trims reads using NanoFilt  
- Removes chimeric reads using VSEARCH  
- Aligns non-chimeric reads to a full-genome database using Minimap2  
- Summarizes mapped and unmapped reads to generate an OTU-like abundance table  

## Installation

### Prerequisites

- [Miniconda](https://docs.conda.io/en/latest/)  
- GNU Parallel  
- Bash (Linux/macOS)

### Create Conda Environments

```bash
conda env create -f envs/nanofilt_env.yml
conda env create -f envs/minibar_env.yml
conda env create -f envs/minimap_env.yml
conda env create -f envs/vsearch_env.yml
conda env create -f envs/samtools_env.yml
conda env create -f envs/pynano.yml
```

---

## Configuration

Edit `config.env` with your paths and settings:

```bash
FASTQ_FILE=/data/emp/exp103_pass.fastq.gz
SEQ_DIR=/data/emp/split_reads
BARCODE_FILE=/data/emp/barcodes.csv
OUTPUT_DIR=/data/emp/output

MINIBAR_SCRIPT=scripts/minibar.py
REV_COMP_SCRIPT=scripts/rev_cmplt_minibar_output.py
REFERENCE_DB=/data/db/filtered_database.fasta

FLIP_SUFFIX=_rev_complemented
SAMPLE_PREFIX=OS20241206

THREADS=50
Q_THRE=15
MIN_LEN=1500
MAX_LEN=1700
```

---

## Usage

### 1. Run the full pipeline

```bash
bash pipeline.sh
```

### 2. Run only alignment on FASTA files

```bash
bash align_fasta.sh
```

### 3. Optional: Clean unwanted genomes from reference

```bash
bash clean_uneeded_genomes.sh
```

### 4. Summarize mapped reads into table

**Basic version:**

```bash
python make_count_table_updated.py --input_dir ./analysis --output_file otu_count_table.csv
```

**With genome filtering:**

```bash
python make_count_table_updated_with_mistmatch_filter_db.py
```


## Use Case

This pipeline is designed for:
- **Mapping EMP metagenomic reads** to curated reference genomes
- **Detecting presence/absence** of known strains in environmental samples
- **Estimating relative abundances** without relying on 16S or marker-gene approaches



