#!/usr/bin/env python

import os
import sys
import click
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from rich.console import Console
from rich.panel import Panel
from rich import print as rprint
import time
import glob
import multiprocessing as mp
from functools import partial
import itertools

# Initialize console for rich output
console = Console()

def read_single_file(file_path):
    """
    Read a single coverage CSV file.
    
    Args:
        file_path (str): Path to the coverage CSV file
        
    Returns:
        tuple: (sample_name, DataFrame) or (None, None) if error occurred
        
    Raises:
        ValueError: If required columns are missing
    """
    try:
        # Get sample name from file name
        sample_name = Path(file_path).stem
        
        # Read CSV file
        df = pd.read_csv(file_path)
        
        # Validate required columns
        required_columns = ['chr', 'pos', 'coverage', 'chr_dmr', 'start_dmr', 'end_dmr']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"File {file_path} is missing required columns: {', '.join(missing_columns)}")
        
        # Add DMR key column
        df['dmr_key'] = df.apply(lambda row: f"{row['chr_dmr']}:{row['start_dmr']}-{row['end_dmr']}", axis=1)
        
        return (sample_name, df)
        
    except Exception as e:
        console.print(f"[bold red]Error reading file {file_path}:[/bold red] {str(e)}", style="red")
        return (None, None)

def read_coverage_files_parallel(input_files, num_processes):
    """
    Read multiple coverage CSV files in parallel.
    
    Args:
        input_files (list): List of input CSV file paths
        num_processes (int): Number of processes to use
        
    Returns:
        dict: Dictionary mapping sample names to DataFrames
        
    Raises:
        ValueError: If no valid files were read
    """
    console.print(f"[bold blue]Reading {len(input_files)} files using {num_processes} processes...[/bold blue]")
    
    # Use multiprocessing to read files in parallel
    with mp.Pool(processes=num_processes) as pool:
        results = list(tqdm(
            pool.imap(read_single_file, input_files),
            total=len(input_files),
            desc="Reading input files",
            unit="file"
        ))
    
    # Filter out None results and create dictionary
    sample_dfs = {name: df for name, df in results if name is not None}
    
    if not sample_dfs:
        raise ValueError("No valid files were read successfully")
    
    return sample_dfs

def create_dmr_key(row):
    """
    Create a unique key for each DMR using its coordinates.
    
    Args:
        row (pd.Series): Row containing DMR information
        
    Returns:
        str: Unique DMR key
    """
    return f"{row['chr_dmr']}:{row['start_dmr']}-{row['end_dmr']}"

def preprocess_data_for_dmr_processing(sample_dfs):
    """
    Preprocess and reorganize data for efficient DMR processing.
    
    Args:
        sample_dfs (dict): Dictionary mapping sample names to DataFrames
        
    Returns:
        tuple: (dmr_data_dict, common_dmrs, all_samples_count)
        
    Raises:
        ValueError: If there are no common DMRs across all samples
    """
    console.print("[bold blue]Preprocessing data for efficient DMR processing...[/bold blue]")
    
    # Find DMRs present in all samples
    sample_dmrs = {sample: set(df['dmr_key'].unique()) for sample, df in sample_dfs.items()}
    common_dmrs = set.intersection(*sample_dmrs.values()) if sample_dmrs else set()
    
    if not common_dmrs:
        raise ValueError("No common DMRs found across all samples")
    
    console.print(f"[bold green]Found {len(common_dmrs)} common DMRs across all samples[/bold green]")
    
    # Preprocess data: Group by DMR key for each sample
    dmr_data_dict = {}
    
    # First pass: Create a dictionary to store position info per DMR
    pos_info = {}
    
    console.print("[bold blue]Organizing data by DMR...[/bold blue]")
    for sample, df in tqdm(sample_dfs.items(), desc="Preprocessing samples"):
        # Filter to only common DMRs to reduce memory usage
        df_filtered = df[df['dmr_key'].isin(common_dmrs)]
        
        # Group data by DMR key
        for dmr_key, group in df_filtered.groupby('dmr_key'):
            # Initialize dictionary for this DMR if needed
            if dmr_key not in dmr_data_dict:
                dmr_data_dict[dmr_key] = defaultdict(dict)
            
            # Store coverage info for each position in this DMR for this sample
            for _, row in group.iterrows():
                pos_key = (row['chr'], row['pos'])
                dmr_data_dict[dmr_key][pos_key][sample] = row['coverage']
                
                # Store position info (only need to do this once)
                if pos_key not in pos_info:
                    pos_info[pos_key] = {
                        'chr': row['chr'],
                        'pos': row['pos'],
                        'chr_dmr': row['chr_dmr'],
                        'start_dmr': row['start_dmr'],
                        'end_dmr': row['end_dmr']
                    }
    
    # Add position info to the dmr_data_dict
    for dmr_key in dmr_data_dict:
        for pos_key in dmr_data_dict[dmr_key]:
            if pos_key in pos_info:
                dmr_data_dict[dmr_key][pos_key]['info'] = pos_info[pos_key]
    
    return dmr_data_dict, common_dmrs, len(sample_dfs)

def process_dmr_batch(dmr_batch, all_samples_count):
    """
    Process a batch of DMRs to find the position with max coverage in each.
    
    Args:
        dmr_batch (list): List of (dmr_key, dmr_data) tuples
        all_samples_count (int): Number of samples
        
    Returns:
        list: List of (dmr_key, position_info) tuples for successfully processed DMRs
    """
    results = []
    
    for dmr_key, dmr_data in dmr_batch:
        # Filter positions that appear in all samples
        complete_positions = {
            pos_key: pos_data 
            for pos_key, pos_data in dmr_data.items() 
            if len(pos_data) - (1 if 'info' in pos_data else 0) == all_samples_count
        }
        
        # If no positions in this DMR are covered in all samples, skip this DMR
        if not complete_positions:
            continue
            
        # Find the position with the highest mean coverage
        max_mean_cov = 0
        max_pos = None
        
        for pos_key, pos_data in complete_positions.items():
            # Calculate mean coverage (excluding the 'info' key)
            coverage_values = [cov for k, cov in pos_data.items() if k != 'info']
            mean_cov = sum(coverage_values) / len(coverage_values)
            
            if mean_cov > max_mean_cov:
                max_mean_cov = mean_cov
                max_pos = pos_key
                
        # Store the max coverage position for this DMR
        if max_pos and 'info' in complete_positions[max_pos]:
            info = complete_positions[max_pos]['info']
            position_info = {
                'chr': info['chr'],
                'pos': info['pos'],
                'mean_coverage': max_mean_cov,
                'chr_dmr': info['chr_dmr'],
                'start_dmr': info['start_dmr'],
                'end_dmr': info['end_dmr']
            }
            results.append((dmr_key, position_info))
    
    return results

def find_max_coverage_positions_parallel_optimized(dmr_data_dict, common_dmrs, all_samples_count, num_processes):
    """
    Find positions with maximum coverage in each DMR using optimized parallel processing.
    
    Args:
        dmr_data_dict (dict): Preprocessed data organized by DMR
        common_dmrs (set): Set of DMR keys common to all samples
        all_samples_count (int): Total number of samples
        num_processes (int): Number of processes to use
        
    Returns:
        pd.DataFrame: DataFrame with maximum coverage positions
        
    Raises:
        ValueError: If no maximum coverage positions found
    """
    console.print(f"[bold blue]Processing DMRs in batches using {num_processes} processes...[/bold blue]")
    
    # Prepare data for batch processing
    dmr_items = list(dmr_data_dict.items())
    
    # Determine optimal batch size based on number of DMRs and processes
    total_dmrs = len(dmr_items)
    batch_size = max(1, min(100, total_dmrs // (num_processes * 4)))
    
    # Create batches of DMRs
    batches = [dmr_items[i:i+batch_size] for i in range(0, total_dmrs, batch_size)]
    
    console.print(f"[bold green]Processing {total_dmrs} DMRs in {len(batches)} batches (batch size: {batch_size})[/bold green]")
    
    # Create a partial function with fixed parameters
    process_func = partial(process_dmr_batch, all_samples_count=all_samples_count)
    
    # Use multiprocessing to process DMR batches in parallel
    with mp.Pool(processes=num_processes) as pool:
        batch_results = list(tqdm(
            pool.imap(process_func, batches),
            total=len(batches),
            desc="Processing DMR batches",
            unit="batch"
        ))
    
    # Flatten results and convert to dictionary
    dmr_results = list(itertools.chain.from_iterable(batch_results))
    
    if not dmr_results:
        raise ValueError("No maximum coverage positions found that are covered in all samples")
    
    # Convert to dictionary and then to DataFrame
    dmr_max_positions = {dmr_key: pos_info for dmr_key, pos_info in dmr_results}
    result_df = pd.DataFrame(list(dmr_max_positions.values()))
    
    # Sort by chromosome and start position of DMR
    result_df = result_df.sort_values(['chr_dmr', 'start_dmr']).reset_index(drop=True)
    
    # Add DMR index
    result_df['dmr_index'] = range(1, len(result_df) + 1)
    
    # Reorder columns
    result_df = result_df[['chr', 'pos', 'mean_coverage', 'dmr_index', 'chr_dmr', 'start_dmr', 'end_dmr']]
    
    return result_df

@click.command()
@click.option('--input', required=True, multiple=True, help='Path(s) to input coverage CSV file(s). Can use wildcards with quotes.')
@click.option('--output', default='max_cov_pos.csv', help='Path to output CSV file (default: max_cov_pos.csv)')
@click.option('--ncpus', type=int, default=None, help='Number of processes to use (default: number of CPU cores)')
@click.option('--verbose', is_flag=True, help='Print verbose output')
def main(input, output, ncpus, verbose):
    """
    Find maximum coverage positions for each DMR across multiple samples.
    
    Takes multiple CSV files with coverage information, finds the position with
    maximum coverage in each DMR across all samples, and outputs a CSV file with
    these positions along with their mean coverage and DMR information.
    
    Input CSV files must have columns: 'chr', 'pos', 'coverage', 'chr_dmr', 'start_dmr', 'end_dmr'
    """
    start_time = time.time()
    
    try:
        # Set number of processes
        if ncpus is None:
            ncpus = mp.cpu_count()
        
        # Limit ncpus to available cores
        ncpus = min(ncpus, mp.cpu_count())
        
        # Expand wildcards if any and handle space-separated filenames
        input_files = []
        for pattern in input:
            # Check if the pattern contains spaces (multiple filenames in one argument)
            if ' ' in pattern:
                # Split by space and check if each individual file exists
                for file_path in pattern.split():
                    if os.path.exists(file_path):
                        input_files.append(file_path)
                    else:
                        console.print(f"[bold yellow]Warning:[/bold yellow] File '{file_path}' does not exist", style="yellow")
            else:
                # Handle patterns with wildcards
                matched_files = glob.glob(pattern)
                if not matched_files:
                    console.print(f"[bold yellow]Warning:[/bold yellow] No files match pattern '{pattern}'", style="yellow")
                input_files.extend(matched_files)
        
        if not input_files:
            raise ValueError("No input files found")
        
        console.print(Panel.fit(
            f"[bold green]Processing [/bold green]{len(input_files)}[bold green] coverage files[/bold green]\n"
            f"[bold green]Using [/bold green]{ncpus}[bold green] CPU cores[/bold green]\n"
            f"[bold green]Output will be saved to:[/bold green] {output}",
            title="Maximum Coverage Position Analysis", 
            border_style="blue"
        ))
        
        # Read all input CSV files in parallel
        sample_dfs = read_coverage_files_parallel(input_files, ncpus)
        
        # Preprocess data for efficient DMR processing
        dmr_data_dict, common_dmrs, all_samples_count = preprocess_data_for_dmr_processing(sample_dfs)
        
        # We can release the original dataframes to save memory
        del sample_dfs
        
        # Find maximum coverage positions in parallel with optimized approach
        max_cov_df = find_max_coverage_positions_parallel_optimized(
            dmr_data_dict,
            common_dmrs,
            all_samples_count,
            ncpus
        )
        
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            console.print(f"Created output directory: {output_dir}")
        
        # Write output CSV
        console.print(f"[bold blue]Writing maximum coverage positions to[/bold blue] {output}")
        max_cov_df.to_csv(output, index=False)
        
        # Print statistics
        elapsed_time = time.time() - start_time
        console.print(Panel.fit(
            f"[bold green]Total DMRs processed:[/bold green] {len(max_cov_df):,}\n"
            f"[bold green]Processing time:[/bold green] {elapsed_time:.2f} seconds",
            title="Analysis Summary", 
            border_style="green"
        ))
        
        # Print additional statistics if verbose
        if verbose and not max_cov_df.empty:
            console.print("[bold blue]Top DMRs by mean coverage:[/bold blue]")
            top_dmrs = max_cov_df.sort_values('mean_coverage', ascending=False).head(10)
            console.print(top_dmrs.to_string())
        
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}", style="red")
        sys.exit(1)

if __name__ == "__main__":
    main()
