#!/usr/bin/env python3
import os
import re
import glob
from collections import defaultdict

def parse_fasta_file(file_path):
    """
    Parse a FASTA file and return a dictionary of reads with their dmr_coordinates as keys.
    Returns a dictionary where:
        - key: dmr_coordinate (e.g., 'chr1:1003362-1003936')
        - value: list of tuples (header, sequence, dmr_index)
    """
    reads = defaultdict(list)
    current_header = ""
    current_seq = ""
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                # Save the previous read if it exists
                if current_header:
                    # Extract dmr_coordinate and dmr_index
                    match = re.search(r'dmr_index:(\d+),dmr_coordiante:([^,\s]+)', current_header)
                    if match:
                        dmr_index = match.group(1)
                        dmr_coordinate = match.group(2)
                        reads[dmr_coordinate].append((current_header, current_seq, dmr_index))
                
                # Start a new read
                current_header = line
                current_seq = ""
            else:
                current_seq += line
        
        # Don't forget to save the last read
        if current_header:
            match = re.search(r'dmr_index:(\d+),dmr_coordiante:([^,\s]+)', current_header)
            if match:
                dmr_index = match.group(1)
                dmr_coordinate = match.group(2)
                reads[dmr_coordinate].append((current_header, current_seq, dmr_index))
    
    return reads

def filter_reads(regular_file, target_file, output_file):
    """
    Filter reads from regular_file that have matching dmr_coordinates in target_file.
    Use the dmr_index from target_file when writing the filtered reads.
    """
    # Parse the target file to get dmr_coordinates and their corresponding dmr_indices
    target_reads = parse_fasta_file(target_file)
    
    # Create a mapping of dmr_coordinate to dmr_index from the target file
    target_dmr_indices = {}
    for dmr_coordinate, reads in target_reads.items():
        for _, _, dmr_index in reads:
            target_dmr_indices[dmr_coordinate] = dmr_index
    
    # Parse the regular file
    regular_reads = parse_fasta_file(regular_file)
    
    # Filter reads from regular file that have matching dmr_coordinates in target file
    filtered_reads = []
    for dmr_coordinate, reads in regular_reads.items():
        if dmr_coordinate in target_dmr_indices:
            for header, seq, _ in reads:
                # Update header with dmr_index from target file
                new_header = re.sub(
                    r'(dmr_index:)\d+', 
                    r'\g<1>' + target_dmr_indices[dmr_coordinate], 
                    header
                )
                filtered_reads.append((new_header, seq))
    
    # Write filtered reads to output file
    with open(output_file, 'w') as f:
        for header, seq in filtered_reads:
            f.write(f"{header}\n{seq}\n")
    
    return len(filtered_reads)

def main():
    # Get all target FASTA files
    target_files = glob.glob('./data/*_target_*.fa')
    
    for target_file in target_files:
        # Extract sample, label, and mode from the target file name
        filename = os.path.basename(target_file)
        match = re.match(r'(.+?)_(.+?)_target_(.+?)\.fa', filename)
        
        if match:
            sample, label, mode = match.groups()
            
            # Construct the regular file name
            regular_file = f'./data/{sample}_{label}_{mode}.fa'
            
            # Construct the output file name
            output_file = f'./data/{sample}_{label}_extracted_{mode}.fa'
            
            # Check if the regular file exists
            if os.path.exists(regular_file):
                print(f"Processing pair: {os.path.basename(regular_file)} and {filename}")
                
                # Filter reads and write to output file
                n_reads = filter_reads(regular_file, target_file, output_file)
                
                print(f"  - Wrote {n_reads} filtered reads to {os.path.basename(output_file)}")
            else:
                print(f"ERROR: Regular file {regular_file} not found for target file {filename}")

if __name__ == "__main__":
    main()
