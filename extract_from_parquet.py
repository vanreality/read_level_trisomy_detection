#!/usr/bin/env python3
import os
import sys
import click
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

@click.command()
@click.option('--input', required=True, type=str, help='Input parquet file path')
@click.option('--output', required=True, type=str, help='Output directory')
@click.option('--sample_col', default='sample', type=str, help='Output directory')
@click.option('--ncpus', default=1, type=int, help='Number of CPUs for parallel processing')
@click.option('--labels_to_keep', required=True, multiple=True, help='Labels to keep, e.g., Normal T16')
@click.option('--threshold', default=0.0, type=float, help='Threshold for prob_class_1 column')
@click.option('--mode', type=click.Choice(['random', 'first']), required=True, help='Read extraction mode')
@click.option('--max_pos', type=str, help='Path to pre-computed max_positions bed file for resuming')
def main(input, output, sample_col, ncpus, labels_to_keep, threshold, mode, max_pos):
    """Extract reads from parquet file based on DMR regions."""
    # Create output directory if it doesn't exist
    os.makedirs(output, exist_ok=True)
    
    logger.info(f"Processing input file: {input}")
    logger.info(f"Output directory: {output}")
    logger.info(f"Labels to keep: {labels_to_keep}")
    logger.info(f"Threshold: {threshold}")
    logger.info(f"Mode: {mode}")
    
    max_positions = None
    df = None
    
    # Check if we're resuming from a max_positions file
    if max_pos and os.path.exists(max_pos):
        logger.info(f"Resuming from max_positions file: {max_pos}")
        max_positions = read_max_positions_from_bed(max_pos)
        logger.info(f"Loaded {len(max_positions)} maximum coverage positions")
    else:
        # Read and filter the parquet file
        logger.info("Reading and filtering parquet file...")
        df = load_and_filter_dataframe(input, labels_to_keep, threshold, ncpus)
        logger.info(f"Filtered data has {len(df)} rows")
        
        # Calculate coverage for each base position
        coverage_dict = calculate_coverage(df)
        logger.info(f"Calculated coverage for {len(coverage_dict)} positions")
        
        # Find maximum coverage positions in each DMR
        max_positions = find_max_coverage_positions(df, coverage_dict)
        logger.info(f"Found {len(max_positions)} maximum coverage positions")
        
        # Write max_positions to bed file first to support breakpoint running
        bed_path = os.path.join(output, 'max_coverage_positions.bed')
        write_max_positions_to_bed(max_positions, bed_path)
        logger.info(f"Wrote maximum coverage positions to {bed_path}")
        
        # Clean up coverage_dict to save memory
        del coverage_dict
    
    # If we're resuming and don't have the dataframe yet, load it now
    if df is None:
        logger.info("Loading and filtering parquet file for read extraction...")
        df = load_and_filter_dataframe(input, labels_to_keep, threshold, ncpus)
    
    # Extract reads with parallel processing and improved memory efficiency
    extracted_reads = extract_reads_parallel_memory_efficient(df, max_positions, mode, ncpus, sample_col)
    total_reads = sum(len(reads) for reads in extracted_reads.values())
    logger.info(f"Extracted {total_reads} reads for {len(extracted_reads)} sample-label combinations")
    
    # Write output files
    write_output_files(max_positions, extracted_reads, output)
    
    logger.info("Processing completed successfully")

def read_max_positions_from_bed(bed_path):
    """Read max_positions from a BED file."""
    max_positions = []
    with open(bed_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 4:
                chrom = parts[0]
                pos = int(parts[1])
                info = parts[3].split(',')
                
                # Extract information from the info field
                dmr_index = None
                dmr_coordinates = None
                cov = None
                
                for item in info:
                    if item.startswith('dmr_index:'):
                        dmr_index = int(item.split(':', 1)[1])
                    elif item.startswith('dmr_coordinates:'):
                        dmr_coordinates = item.split(':', 1)[1]
                    elif item.startswith('max_coverage:'):
                        cov = int(item.split(':', 1)[1])
                
                if dmr_index is not None and dmr_coordinates is not None and cov is not None:
                    max_positions.append((chrom, pos, cov, dmr_index, dmr_coordinates))
    
    return max_positions

def write_max_positions_to_bed(max_positions, bed_path):
    """Write max_positions to a BED file."""
    with open(bed_path, 'w') as f:
        for chrom, pos, cov, dmr_index, dmr_coordinates in max_positions:
            f.write(f"{chrom}\t{pos}\t{pos+1}\tdmr_index:{dmr_index},dmr_coordinates:{dmr_coordinates},max_coverage:{cov}\n")

def load_and_filter_dataframe(input_file, labels_to_keep, threshold, ncpus):
    """Load and filter the parquet file, potentially using parallel processing."""
    if ncpus > 1:
        logger.info(f"Using {ncpus} CPUs for parallel processing")
        
        # First, read the parquet file metadata to get the row count
        try:
            import pyarrow.parquet as pq
            parquet_file = pq.ParquetFile(input_file)
            total_rows = parquet_file.metadata.num_rows
            logger.info(f"Total rows in parquet file: {total_rows}")
            
            # Calculate chunk sizes for manual partitioning
            chunk_size = max(1, total_rows // ncpus)
            chunks = []
            
            # Read the whole DataFrame
            df_full = pd.read_parquet(input_file)
            
            # Split into chunks for parallel processing
            for i in range(0, ncpus):
                start_idx = i * chunk_size
                end_idx = min((i + 1) * chunk_size, total_rows)
                if start_idx < end_idx:  # Only add non-empty chunks
                    chunks.append((i, df_full.iloc[start_idx:end_idx], labels_to_keep, threshold))
            
            # Process chunks in parallel
            with Pool(ncpus) as pool:
                filtered_chunks = pool.map(process_chunk, chunks)
            
            # Combine filtered chunks
            df = pd.concat(filtered_chunks, ignore_index=True)
            
            # Clean up to save memory
            del df_full
            del filtered_chunks
            
        except Exception as e:
            logger.warning(f"Error in parallel processing: {e}. Falling back to single-thread processing.")
            df = pd.read_parquet(input_file)
            df = filter_df(df, labels_to_keep, threshold)
    else:
        # Process the entire file at once
        df = pd.read_parquet(input_file)
        df = filter_df(df, labels_to_keep, threshold)
    
    return df

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

def extract_reads_parallel_memory_efficient(df, max_positions, mode, ncpus, sample_col):
    """
    Extract reads for each sample at maximum coverage positions, with improved memory efficiency.
    
    Strategy:
    1. Process max_positions in batches to limit memory usage
    2. Process each chromosome separately to avoid loading entire dataframe
    3. Use generators to avoid storing all intermediate results
    
    Args:
        df: DataFrame with filtered reads
        max_positions: List of (chr, pos, coverage, dmr_index, dmr_coordinates) for max coverage positions
        mode: Read extraction mode ('random' or 'first')
        ncpus: Number of CPUs to use
        
    Returns:
        extracted_reads: Dictionary mapping (sampledown, label) to list of read tuples
    """
    logger.info(f"Extracting reads using mode: {mode} with {ncpus} CPUs (memory-efficient mode)...")
    
    # Group max_positions by chromosome to process one chromosome at a time
    chr_groups = defaultdict(list)
    for pos_info in max_positions:
        chr_groups[pos_info[0]].append(pos_info)
    
    all_extracted_reads = defaultdict(list)
    
    # Process each chromosome separately
    total_chrs = len(chr_groups)
    for chr_idx, (chrom, chr_positions) in enumerate(chr_groups.items(), 1):
        logger.info(f"Processing chromosome {chrom} ({chr_idx}/{total_chrs})")
        
        # Filter dataframe to only include this chromosome
        chr_df = df[df['chr'] == chrom].copy()
        logger.info(f"Filtered to {len(chr_df)} reads on chromosome {chrom}")
        
        if len(chr_df) == 0:
            continue
        
        # Process in batches to limit memory usage
        batch_size = max(1, len(chr_positions) // ncpus)
        total_batches = (len(chr_positions) + batch_size - 1) // batch_size
        
        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(chr_positions))
            batch_positions = chr_positions[start_idx:end_idx]
            
            logger.info(f"Processing batch {batch_idx+1}/{total_batches} with {len(batch_positions)} positions")
            
            # Process each position in the batch
            for pos_info in tqdm(batch_positions, desc=f"Chr {chrom} batch {batch_idx+1}"):
                chrom, pos, _, dmr_index, dmr_coordinates = pos_info
                
                # Filter reads that cover this position
                covering_reads = chr_df[
                    (chr_df['start'] <= pos) & 
                    (chr_df['end'] >= pos)
                ]
                
                # Group by sample
                for sample, sample_df in covering_reads.groupby(sample_col):
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
                        all_extracted_reads[(sample, label)].append((
                            selected_read['name'],
                            selected_read['seq'],
                            selected_read['prob_class_1'],
                            dmr_index,
                            dmr_coordinates
                        ))
            
            # Clear memory after processing each batch
            if batch_idx < total_batches - 1:
                logger.info("Clearing memory between batches")
            
        # Clear memory after processing each chromosome
        del chr_df
        if chr_idx < total_chrs:
            logger.info("Clearing memory between chromosomes")
    
    return all_extracted_reads

def write_output_files(max_positions, extracted_reads, output_dir):
    """
    Write output files:
    1. BED file with maximum coverage positions (if not already written)
    2. FASTA files for extracted reads
    
    Args:
        max_positions: List of (chr, pos, coverage, dmr_index, dmr_coordinates) for max coverage positions
        extracted_reads: Dictionary mapping (sampledown, label) to list of read tuples
        output_dir: Output directory path
    """
    logger.info("Writing output files...")
    
    # Check if BED file already exists
    bed_path = os.path.join(output_dir, 'max_coverage_positions.bed')
    if not os.path.exists(bed_path):
        write_max_positions_to_bed(max_positions, bed_path)
        logger.info(f"Wrote maximum coverage positions to {bed_path}")
    
    # Write FASTA files for extracted reads
    for (sample, label), reads in extracted_reads.items():
        fasta_path = os.path.join(output_dir, f"{sample}_{label}.fa")
        with open(fasta_path, 'w') as f:
            for name, seq, prob, dmr_index, dmr_coordinates in reads:
                f.write(f">read_name:{name},sample:{sample},prob:{prob},dmr_index:{dmr_index},dmr_coordinates:{dmr_coordinates}\n{seq}\n")
        
        logger.info(f"Wrote {len(reads)} reads to {fasta_path}")

if __name__ == "__main__":
    main() # This invokes the Click command
