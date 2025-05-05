#!/bin/bash
set -e  # exit on error

# Load configuration
source config.env

# Activate environment
conda activate seqtk
seqkit split2 "$FASTQ_FILE" -p 100 -f
conda deactivate

# Define function to process one FASTQ split
process_seq_file () {
    local seq_file=$1
    local sample_prefix="${SAMPLE_PREFIX}_$(basename "${seq_file}" | grep -oP '\d{8}')_"
    local seq_file_filtered="${seq_file%.fastq.gz}_filtered_q${Q_THRE}_min${MIN_LEN}_max${MAX_LEN}.fastq"

    temp_unzipped="${seq_file%.gz}"
    conda run -n nanofilt gunzip -c "${seq_file}" > "${temp_unzipped}"
    conda run -n nanofilt NanoFilt -q $Q_THRE -l $MIN_LEN --maxlength $MAX_LEN "$temp_unzipped" > "$seq_file_filtered"
    rm "$temp_unzipped"

    # Demultiplex
    cd "$OUTPUT_DIR"
    conda run -n minibar_env python3 "$MINIBAR_SCRIPT" "$BARCODE_FILE" "$seq_file_filtered" -F -T -P "$sample_prefix" -e 6 -E 6 -l 150
    cd -
}

export -f process_seq_file

# Generate file list and process in parallel
seq_files=()
for i in $(seq -w 1 100); do
    seq_files+=("${SEQ_DIR}/exp103_pass.part_${i}.fastq.gz")
done

conda run -n xzy_nano parallel -j $THREADS process_seq_file ::: "${seq_files[@]}"

# Combine parts
for sample in $(ls $OUTPUT_DIR | sed -E 's/.*part_[0-9]{3}_(.*)\.fastq/\1/' | sort | uniq); do
    cat ${OUTPUT_DIR}/${SAMPLE_PREFIX}part_*_${sample}.fastq > ${OUTPUT_DIR}/${SAMPLE_PREFIX}${sample}_filtered_combined.fastq
    echo "${sample} combined"
done

# Flip reverse read
for dem_file in $OUTPUT_DIR/${SAMPLE_PREFIX}*_filtered_combined.fastq; do
    conda run -n pynano python3 "$REV_COMP_SCRIPT" "$dem_file" --suffix "$FLIP_SUFFIX"
done

# Remove chimeras
conda activate vsearch
for flip_fastq in $OUTPUT_DIR/${SAMPLE_PREFIX}*${FLIP_SUFFIX}.fastq; do
    flip_fasta="${flip_fastq%.fastq}.fasta"
    vsearch --fastq_filter "$flip_fastq" --fastaout "$flip_fasta" --fastq_qmax 60
    vsearch --uchime_ref "$flip_fasta" --db "$REFERENCE_DB" \
        --chimeras "${flip_fasta%.fasta}_chimeras.fasta" \
        --nonchimeras "${flip_fasta%.fasta}_nonchimeras.fasta" \
        --uchimeout "${flip_fasta%.fasta}.uchimeout"
done

# Map with minimap2
conda activate minimap
for nochim in $OUTPUT_DIR/${SAMPLE_PREFIX}*${FLIP_SUFFIX}_nonchimeras.fasta; do
    output_sam="${nochim%.fasta}.sam"
    minimap2 -t $THREADS -ax map-ont "$REFERENCE_DB" "$nochim" > "$output_sam"
done

# Summarize with samtools
conda activate samtools
for sam in $OUTPUT_DIR/*${FLIP_SUFFIX}_nonchimeras.sam; do
    bam="${sam%.sam}.bam"
    samtools view -Sb "$sam" | samtools sort -o "$bam"
    samtools index "$bam"
    samtools view "$bam" -F 0x900 -b | samtools idxstats - > "${bam}.count"
done

python3 make_count_table_updated.py --input_dir "$OUTPUT_DIR" --output_file "${OUTPUT_DIR}/otu_count_table.csv"
