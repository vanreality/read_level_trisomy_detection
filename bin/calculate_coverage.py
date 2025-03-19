#!/usr/bin/env python

import os
import sys
import click
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
import numpy as np
import multiprocessing as mp
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm
from rich.console import Console
from rich.panel import Panel
from rich import print as rprint
import time

# Initialize console for rich output
console = Console()

def get_file_prefix(file_path):
    """
    Extract the prefix from a file path (without extension).
    
    Args:
        file_path (str): Path to the input file
        
    Returns:
        str: Prefix of the file name without extension
    """
    base_name = os.path.basename(file_path)
    prefix = os.path.splitext(base_name)[0]
    return prefix

def validate_parquet_file(input_file):
    """
    Validate the input parquet file has required columns.
    
    Args:
        input_file (str): Path to the input parquet file
        
    Returns:
        bool: True if valid, False otherwise
        
    Raises:
        ValueError: If required columns are missing
    """
    try:
        parquet_file = pq.ParquetFile(input_file)
        schema = parquet_file.schema
        columns = schema.names
        
        required_columns = ['chr', 'start', 'end', 'chr_dmr', 'start_dmr', 'end_dmr']
        missing_columns = [col for col in required_columns if col not in columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")
        
        return True
    except Exception as e:
        raise ValueError(f"Error validating parquet file: {e}")

def process_chunk(chunk_data):
    """
    Process a chunk of the parquet data to calculate coverage.
    
    Args:
        chunk_data (pd.DataFrame): DataFrame containing a chunk of read alignments
        
    Returns:
        dict: Dictionary mapping (chr, pos) to (coverage count, chr_dmr, start_dmr, end_dmr)
    """
    coverage_dict = {}
    
    # Process each read in the chunk
    for _, row in chunk_data.iterrows():
        chrom = row['chr']
        start_pos = row['start']
        end_pos = row['end']
        
        # Get DMR information
        chr_dmr = row['chr_dmr']
        start_dmr = row['start_dmr']
        end_dmr = row['end_dmr']
        
        # Increment coverage for each position within the read alignment
        for pos in range(start_pos, end_pos + 1):
            key = (chrom, pos)
            if key not in coverage_dict:
                coverage_dict[key] = {
                    'count': 1,
                    'chr_dmr': chr_dmr,
                    'start_dmr': start_dmr,
                    'end_dmr': end_dmr
                }
            else:
                coverage_dict[key]['count'] += 1
    
    return coverage_dict

def merge_coverage_dicts(dict_list):
    """
    Merge multiple coverage dictionaries into one.
    
    Args:
        dict_list (list): List of coverage dictionaries
        
    Returns:
        dict: Merged coverage dictionary
    """
    merged_dict = {}
    for d in dict_list:
        for key, value in d.items():
            if key not in merged_dict:
                merged_dict[key] = value.copy()
            else:
                merged_dict[key]['count'] += value['count']
    return merged_dict

def calculate_coverage_parallel(input_file, num_processes=None):
    """
    Calculate coverage for each position in the genome using parallelization.
    
    Args:
        input_file (str): Path to the input parquet file
        num_processes (int, optional): Number of processes to use
        
    Returns:
        pd.DataFrame: DataFrame with chr, pos, coverage, and DMR information
    """
    if num_processes is None:
        # Use number of CPU cores available
        num_processes = mp.cpu_count()
    
    # Read the parquet file
    df = pq.read_table(input_file).to_pandas()
    
    # Determine chunk size for multiprocessing
    chunk_size = max(1, len(df) // num_processes)
    chunks = [df.iloc[i:i+chunk_size] for i in range(0, len(df), chunk_size)]
    
    console.print(f"[bold blue]Processing [/bold blue]{len(df):,}[bold blue] reads with [/bold blue]{num_processes}[bold blue] processes[/bold blue]")
    
    # Use multiprocessing to process chunks in parallel
    with mp.Pool(processes=num_processes) as pool:
        results = list(tqdm(
            pool.imap(process_chunk, chunks),
            total=len(chunks),
            desc="Processing coverage",
            unit="chunk"
        ))
    
    # Merge results from all processes
    coverage_dict = merge_coverage_dicts(results)
    
    # Convert dictionary to DataFrame
    coverage_data = []
    for (chrom, pos), data in coverage_dict.items():
        coverage_data.append({
            'chr': chrom,
            'pos': pos,
            'coverage': data['count'],
            'chr_dmr': data['chr_dmr'], 
            'start_dmr': data['start_dmr'],
            'end_dmr': data['end_dmr']
        })
    
    coverage_df = pd.DataFrame(coverage_data)
    
    # Sort the DataFrame by chromosome and position
    coverage_df = coverage_df.sort_values(['chr', 'pos']).reset_index(drop=True)
    
    return coverage_df

@click.command()
@click.option('--input', required=True, help='Path to the input parquet file with read alignments')
@click.option('--output', help='Path to the output CSV file (default: {input_prefix}_coverage.csv)')
@click.option('--ncpus', type=int, default=None, help='Number of processes to use (default: number of CPU cores)')
@click.option('--verbose', is_flag=True, help='Print verbose output')
def main(input, output, ncpus, verbose):
    """
    Calculate per-position coverage from read alignments in a parquet file.
    
    Takes a parquet file containing read alignments with 'chr', 'start', 'end', 'chr_dmr', 'start_dmr', 
    and 'end_dmr' columns and produces a CSV file with per-position coverage counts and DMR information.
    """
    start_time = time.time()
    
    try:
        # Check if the input file exists
        if not os.path.exists(input):
            console.print(f"[bold red]Error:[/bold red] Input file '{input}' does not exist", style="red")
            sys.exit(1)
        
        # Validate the parquet file
        validate_parquet_file(input)
        
        # If output path is not specified, use default naming
        if not output:
            prefix = get_file_prefix(input)
            output = f"{prefix}_coverage.csv"
        
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            console.print(f"Created output directory: {output_dir}")
        
        # Calculate coverage
        console.print(Panel.fit(
            f"[bold green]Calculating coverage for:[/bold green] {input}\n"
            f"[bold green]Output will be saved to:[/bold green] {output}",
            title="Coverage Calculation", 
            border_style="blue"
        ))
        
        # Calculate coverage with parallelization
        coverage_df = calculate_coverage_parallel(input, ncpus)
        
        # Write output CSV
        console.print(f"[bold blue]Writing coverage data to[/bold blue] {output}")
        coverage_df.to_csv(output, index=False)
        
        # Print statistics
        elapsed_time = time.time() - start_time
        console.print(Panel.fit(
            f"[bold green]Total positions processed:[/bold green] {len(coverage_df):,}\n"
            f"[bold green]Unique chromosomes:[/bold green] {coverage_df['chr'].nunique()}\n"
            f"[bold green]Processing time:[/bold green] {elapsed_time:.2f} seconds",
            title="Coverage Calculation Summary", 
            border_style="green"
        ))
        
        # Print coverage statistics per chromosome if verbose
        if verbose:
            console.print("[bold blue]Coverage statistics by chromosome:[/bold blue]")
            chrom_stats = coverage_df.groupby('chr').agg(
                mean_coverage=('coverage', 'mean'),
                max_coverage=('coverage', 'max'),
                total_positions=('coverage', 'count')
            ).reset_index()
            console.print(chrom_stats.to_string(index=False))
        
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}", style="red")
        sys.exit(1)

if __name__ == "__main__":
    main()
