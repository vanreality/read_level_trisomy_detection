#!/usr/bin/env python3
import os
import sys
import argparse
import pandas as pd
import numpy as np
import random
from collections import defaultdict
from multiprocessing import Pool
import logging
from tqdm import tqdm
import functools

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Extract reads from parquet file based on DMR regions')
    parser.add_argument('--input', type=str, required=True, help='Input parquet file path')
    parser.add_argument('--output', type=str, required=True, help='Output directory')
    parser.add_argument('--ncpus', type=int, default=1, help='Number of CPUs for parallel processing')
    parser.add_argument('--labels_to_keep', type=str, nargs='+', required=True, help='Labels to keep, e.g., Normal T16')
    parser.add_argument('--threshold', type=float, default=0, help='Threshold for prob_class_1 column')
    parser.add_argument('--mode', type=str, choices=['random', 'first'], required=True, help='Read extraction mode')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    return args

def calculate_coverage(df):
    """
    Calculate coverage for each base position.
    
    Args:
        df: DataFrame with filtered reads
        
    Returns:
        coverage_dict: Dictionary mapping (chr, position) to coverage
    """
    logger.info("Calculating coverage for each base position...")
    coverage_dict = defaultdict(int)
    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        chrom = row['chr']
        start = row['start']
        end = row['end']
        
        # Increment coverage for each base position in the read
        for pos in range(start, end + 1):
            coverage_dict[(chrom, pos)] += 1
    
    return coverage_dict

def find_max_coverage_positions(df, coverage_dict):
    """
    Find the maximum coverage position in each DMR region and assign indices.
    
    Args:
        df: DataFrame with filtered reads
        coverage_dict: Dictionary mapping (chr, position) to coverage
        
    Returns:
        max_positions: List of (chr, pos, coverage, dmr_index, dmr_coordinates) for max coverage positions in each DMR
    """
    logger.info("Finding maximum coverage positions in each DMR region...")
    
    # Get unique DMR regions
    dmr_regions = df[['chr_dmr', 'start_dmr', 'end_dmr']].drop_duplicates().values
    
    # Sort DMR regions for consistent indexing across all samples
    dmr_regions = sorted(dmr_regions, key=lambda x: (x[0], x[1], x[2]))
    
    max_positions = []
    for dmr_index, (chrom, start, end) in enumerate(tqdm(dmr_regions)):
        max_cov = 0
        max_pos_candidates = []
        dmr_coordinates = f"{chrom}:{start}-{end}"
        
        # Find positions with maximum coverage in this DMR
        for pos in range(start, end + 1):
            cov = coverage_dict.get((chrom, pos), 0)
            if cov > max_cov:
                max_cov = cov
                max_pos_candidates = [pos]
            elif cov == max_cov and cov > 0:
                max_pos_candidates.append(pos)
        
        # If we found positions with coverage, randomly choose one with max coverage
        if max_pos_candidates:
            chosen_pos = random.choice(max_pos_candidates)
            max_positions.append((chrom, chosen_pos, max_cov, dmr_index, dmr_coordinates))
    
    return max_positions

def process_extract_reads_chunk(chunk_data):
    """Process a chunk of max_positions for parallel read extraction."""
    chunk_index, max_positions_chunk, df, mode = chunk_data
    logger.info(f"Extracting reads for chunk {chunk_index+1} with {len(max_positions_chunk)} positions")
    
    # Group reads by sample and label
    extracted_reads = defaultdict(list)
    
    for chrom, pos, max_cov, dmr_index, dmr_coordinates in tqdm(max_positions_chunk, 
                                                               position=chunk_index, 
                                                               desc=f"Chunk {chunk_index+1}"):
        # Filter reads that cover this position
        covering_reads = df[
            (df['chr'] == chrom) & 
            (df['start'] <= pos) & 
            (df['end'] >= pos)
        ]
        
        # Group by sample
        for sample, sample_df in covering_reads.groupby('sampledown'):
            for label, label_df in sample_df.groupby('label'):
                if len(label_df) == 0:
                    continue
                
                # Find reads with maximum length
                read_lengths = label_df['end'] - label_df['start'] + 1
                max_length = read_lengths.max()
                longest_reads = label_df[read_lengths == max_length]
                
                if len(longest_reads) == 0:
                    continue
                
                if mode == 'random':
                    # Randomly select one of the longest reads
                    selected_read = longest_reads.sample(1).iloc[0]
                else:  # mode == 'first'
                    # Sort by mapping coordinate and pick the first one
                    selected_read = longest_reads.sort_values(['start', 'end']).iloc[0]
                
                # Add to extracted reads with DMR information
                extracted_reads[(sample, label)].append((
                    selected_read['name'],
                    selected_read['seq'],
                    selected_read['prob_class_1'],
                    dmr_index,
                    dmr_coordinates
                ))
    
    return extracted_reads

def extract_reads_parallel(df, max_positions, mode, ncpus):
    """
    Extract reads for each sample at each maximum coverage position using parallel processing.
    
    Args:
        df: DataFrame with filtered reads
        max_positions: List of (chr, pos, coverage, dmr_index, dmr_coordinates) for max coverage positions
        mode: Read extraction mode ('random' or 'first')
        ncpus: Number of CPUs to use
        
    Returns:
        extracted_reads: Dictionary mapping (sampledown, label) to list of read tuples
    """
    logger.info(f"Extracting reads using mode: {mode} with {ncpus} CPUs...")
    
    # Split max_positions into chunks for parallel processing
    chunk_size = max(1, len(max_positions) // ncpus)
    chunks = []
    
    for i in range(min(ncpus, len(max_positions))):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(max_positions))
        if start_idx < end_idx:  # Only add non-empty chunks
            chunks.append((i, max_positions[start_idx:end_idx], df, mode))
    
    # Process chunks in parallel
    all_extracted_reads = defaultdict(list)
    
    if ncpus > 1 and len(chunks) > 1:
        with Pool(min(ncpus, len(chunks))) as pool:
            chunk_results = pool.map(process_extract_reads_chunk, chunks)
            
            # Combine results from all chunks
            for chunk_dict in chunk_results:
                for key, reads in chunk_dict.items():
                    all_extracted_reads[key].extend(reads)
    else:
        # Process sequentially if only one chunk or one CPU
        chunk_dict = process_extract_reads_chunk(chunks[0])
        for key, reads in chunk_dict.items():
            all_extracted_reads[key].extend(reads)
    
    return all_extracted_reads

def write_output_files(max_positions, extracted_reads, output_dir):
    """
    Write output files:
    1. BED file with maximum coverage positions
    2. FASTA files for extracted reads
    
    Args:
        max_positions: List of (chr, pos, coverage, dmr_index, dmr_coordinates) for max coverage positions
        extracted_reads: Dictionary mapping (sampledown, label) to list of read tuples
        output_dir: Output directory path
    """
    logger.info("Writing output files...")
    
    # Write BED file with maximum coverage positions
    bed_path = os.path.join(output_dir, 'max_coverage_positions.bed')
    with open(bed_path, 'w') as f:
        for chrom, pos, cov, dmr_index, dmr_coordinates in max_positions:
            f.write(f"{chrom}\t{pos}\t{pos+1}\tdmr_index:{dmr_index},dmr_coordinates:{dmr_coordinates},max_coverage:{cov}\n")
    
    logger.info(f"Wrote maximum coverage positions to {bed_path}")
    
    # Write FASTA files for extracted reads
    for (sample, label), reads in extracted_reads.items():
        fasta_path = os.path.join(output_dir, f"{sample}_{label}.fa")
        with open(fasta_path, 'w') as f:
            for name, seq, prob, dmr_index, dmr_coordinates in reads:
                f.write(f">read_name:{name},sample:{sample},prob:{prob},dmr_index:{dmr_index},dmr_coordinates:{dmr_coordinates}\n{seq}\n")
        
        logger.info(f"Wrote {len(reads)} reads to {fasta_path}")

def filter_df(df, labels_to_keep, threshold):
    """Filter DataFrame based on labels and threshold."""
    return df[
        (df['label'].isin(labels_to_keep)) &
        (df['prob_class_1'] >= threshold)
    ]

def process_chunk(chunk_data):
    """Process a chunk of the DataFrame for parallel processing."""
    chunk_index, chunk_df, labels_to_keep, threshold = chunk_data
    logger.info(f"Processing chunk {chunk_index+1}")
    return filter_df(chunk_df, labels_to_keep, threshold)

def main():
    args = parse_args()
    
    logger.info(f"Processing input file: {args.input}")
    logger.info(f"Output directory: {args.output}")
    logger.info(f"Labels to keep: {args.labels_to_keep}")
    logger.info(f"Threshold: {args.threshold}")
    logger.info(f"Mode: {args.mode}")
    
    # Read and filter the parquet file
    logger.info("Reading parquet file...")
    
    # pd.read_parquet doesn't support chunksize, need to use a different approach for parallel processing
    if args.ncpus > 1:
        logger.info(f"Using {args.ncpus} CPUs for parallel processing")
        
        # First, read the parquet file metadata to get the row count
        try:
            import pyarrow.parquet as pq
            parquet_file = pq.ParquetFile(args.input)
            total_rows = parquet_file.metadata.num_rows
            logger.info(f"Total rows in parquet file: {total_rows}")
            
            # Calculate chunk sizes for manual partitioning
            chunk_size = max(1, total_rows // args.ncpus)
            chunks = []
            
            # Read the whole DataFrame
            df_full = pd.read_parquet(args.input)
            
            # Split into chunks for parallel processing
            for i in range(0, args.ncpus):
                start_idx = i * chunk_size
                end_idx = min((i + 1) * chunk_size, total_rows)
                if start_idx < end_idx:  # Only add non-empty chunks
                    chunks.append((i, df_full.iloc[start_idx:end_idx], args.labels_to_keep, args.threshold))
            
            # Process chunks in parallel
            with Pool(args.ncpus) as pool:
                filtered_chunks = pool.map(process_chunk, chunks)
            
            # Combine filtered chunks
            df = pd.concat(filtered_chunks, ignore_index=True)
            
            # Clean up to save memory
            del df_full
            del filtered_chunks
            
        except Exception as e:
            logger.warning(f"Error in parallel processing: {e}. Falling back to single-thread processing.")
            df = pd.read_parquet(args.input)
            df = filter_df(df, args.labels_to_keep, args.threshold)
    else:
        # Process the entire file at once
        df = pd.read_parquet(args.input)
        df = filter_df(df, args.labels_to_keep, args.threshold)
    
    logger.info(f"Filtered data has {len(df)} rows")
    
    # Calculate coverage for each base position
    coverage_dict = calculate_coverage(df)
    logger.info(f"Calculated coverage for {len(coverage_dict)} positions")
    
    # Find maximum coverage positions in each DMR
    max_positions = find_max_coverage_positions(df, coverage_dict)
    logger.info(f"Found {len(max_positions)} maximum coverage positions")
    
    # Extract reads with parallel processing
    extracted_reads = extract_reads_parallel(df, max_positions, args.mode, args.ncpus)
    total_reads = sum(len(reads) for reads in extracted_reads.values())
    logger.info(f"Extracted {total_reads} reads for {len(extracted_reads)} sample-label combinations")
    
    # Write output files
    write_output_files(max_positions, extracted_reads, args.output)
    
    logger.info("Processing completed successfully")

if __name__ == "__main__":
    main()
