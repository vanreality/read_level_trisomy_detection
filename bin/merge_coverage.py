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
        
        # Rename coverage column to include sample name
        df = df.rename(columns={'coverage': f'coverage_{sample_name}'})
        
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

def merge_sample_dataframes(sample_dfs):
    """
    Merge all sample DataFrames on position and DMR information.
    
    Args:
        sample_dfs (dict): Dictionary mapping sample names to DataFrames
        
    Returns:
        pd.DataFrame: Merged DataFrame with coverage columns for each sample
        
    Raises:
        ValueError: If merged DataFrame is empty
    """
    console.print("[bold blue]Merging sample data...[/bold blue]")
    
    # Get sample names
    sample_names = list(sample_dfs.keys())
    
    if not sample_names:
        raise ValueError("No samples to merge")
    
    # Start with the first DataFrame
    first_sample = sample_names[0]
    merged_df = sample_dfs[first_sample]
    
    # Define columns to keep during merges (exclude coverage from right dataframes)
    merge_cols = ['chr', 'pos', 'chr_dmr', 'start_dmr', 'end_dmr', 'dmr_key']
    
    # Merge with each subsequent DataFrame
    for sample in tqdm(sample_names[1:], desc="Merging samples"):
        merged_df = pd.merge(
            merged_df, 
            sample_dfs[sample], 
            on=merge_cols,
            how='inner'  # Only keep positions present in all samples
        )
    
    if merged_df.empty:
        raise ValueError("No common positions found across all samples")
    
    return merged_df

def process_batch(batch_df, coverage_cols):
    """
    Process a batch of positions to find the maximum coverage position for each DMR.
    
    Args:
        batch_df (pd.DataFrame): Batch of positions with coverage information
        coverage_cols (list): List of coverage column names
        
    Returns:
        pd.DataFrame: DataFrame with maximum coverage positions for each DMR
    """
    # Calculate mean coverage across all samples for each position
    batch_df['mean_coverage'] = batch_df[coverage_cols].mean(axis=1)
    
    # Group by DMR key and find the position with maximum mean coverage
    max_pos_indices = batch_df.groupby('dmr_key')['mean_coverage'].idxmax()
    
    # Extract the rows with maximum coverage for each DMR
    max_pos_df = batch_df.loc[max_pos_indices]
    
    # Select required columns for output
    output_columns = ['chr', 'pos', 'mean_coverage', 'chr_dmr', 'start_dmr', 'end_dmr', 'dmr_key']
    result_df = max_pos_df[output_columns]
    
    return result_df

def find_max_coverage_positions(merged_df, num_processes):
    """
    Find positions with maximum coverage in each DMR in parallel.
    
    Args:
        merged_df (pd.DataFrame): Merged DataFrame with coverage information for all samples
        num_processes (int): Number of processes to use
        
    Returns:
        pd.DataFrame: DataFrame with maximum coverage positions
    """
    console.print("[bold blue]Finding maximum coverage positions...[/bold blue]")
    
    # Get coverage columns (those starting with 'coverage_')
    coverage_cols = [col for col in merged_df.columns if col.startswith('coverage_')]
    
    # Get unique DMR keys
    dmr_keys = merged_df['dmr_key'].unique()
    console.print(f"[bold green]Processing {len(dmr_keys)} unique DMRs[/bold green]")
    
    # Process in batches
    batch_size = max(1, len(dmr_keys) // (num_processes * 4))
    dmr_batches = [dmr_keys[i:i+batch_size] for i in range(0, len(dmr_keys), batch_size)]
    
    console.print(f"[bold green]Processing in {len(dmr_batches)} batches (batch size: {batch_size})[/bold green]")
    
    # Create a function to process one batch of DMRs
    def process_dmr_batch(dmr_batch):
        # Filter DataFrame for the batch of DMRs
        batch_df = merged_df[merged_df['dmr_key'].isin(dmr_batch)]
        return process_batch(batch_df, coverage_cols)
    
    # Use multiprocessing to process DMR batches in parallel
    with mp.Pool(processes=num_processes) as pool:
        batch_results = list(tqdm(
            pool.imap(process_dmr_batch, dmr_batches),
            total=len(dmr_batches),
            desc="Processing DMR batches",
            unit="batch"
        ))
    
    # Combine results
    result_df = pd.concat(batch_results, ignore_index=True)
    
    if result_df.empty:
        raise ValueError("No maximum coverage positions found")
    
    # Sort by DMR coordinates
    result_df = result_df.sort_values(['chr_dmr', 'start_dmr']).reset_index(drop=True)
    
    # Add DMR index
    result_df['dmr_index'] = range(1, len(result_df) + 1)
    
    # Reorder and select final columns
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
        
        # Merge all DataFrames
        merged_df = merge_sample_dataframes(sample_dfs)
        
        # Count unique DMRs
        dmr_count = merged_df['dmr_key'].nunique()
        console.print(f"[bold green]Found {dmr_count} common DMRs across all samples[/bold green]")
        
        # We can release the original dataframes to save memory
        del sample_dfs
        
        # Find maximum coverage positions in parallel
        max_cov_df = find_max_coverage_positions(merged_df, ncpus)
        
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
