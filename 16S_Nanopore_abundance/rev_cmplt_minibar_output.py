from Bio.Seq import Seq
from Bio import SeqIO
import argparse

def reverse_complement_and_reverse_quality(record):
    if record.description.upper().find("H-") < record.description.upper().find("H+"):
        # Reverse complement the sequence
        record.seq = record.seq.reverse_complement()
        # Reverse the quality scores
        record.letter_annotations["phred_quality"] = record.letter_annotations["phred_quality"][::-1]
    return record

# Process the FASTQ file
def process_fastq(input_file, output_file):
    with open(input_file, "r") as infile, open(output_file, "w") as outfile:
        records = SeqIO.parse(infile, "fastq")
        modified_records = (reverse_complement_and_reverse_quality(record) for record in records)
        SeqIO.write(modified_records, outfile, "fastq")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reverse complement reverse reads in minibar output.")
    
    parser.add_argument("input_file")
    parser.add_argument("--suffix", default="_rev_complemented", help="suffix to attach to file name.")
    
    args = parser.parse_args()
    
    process_fastq(args.input_file, args.input_file[:-6] + args.suffix + ".fastq")
