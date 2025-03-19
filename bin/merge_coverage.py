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

# Initialize console for rich output
console = Console()

def read_coverage_files(input_files):
    """
    Read and combine multiple coverage CSV files.
    
    Args:
        input_files (list): List of input CSV file paths
        
    Returns:
        dict: Dictionary mapping sample names to DataFrames
        
    Raises:
        ValueError: If required columns are missing in any file
    """
    sample_dfs = {}
    
    for file_path in tqdm(input_files, desc="Reading input files"):
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
            
            # Store DataFrame
            sample_dfs[sample_name] = df
            
        except Exception as e:
            console.print(f"[bold red]Error reading file {file_path}:[/bold red] {str(e)}", style="red")
            raise
    
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

def find_max_coverage_positions(sample_dfs):
    """
    Find positions with maximum coverage in each DMR across all samples.
    
    Args:
        sample_dfs (dict): Dictionary mapping sample names to DataFrames
        
    Returns:
        pd.DataFrame: DataFrame with maximum coverage positions
        
    Raises:
        ValueError: If there are no common DMRs across all samples
    """
    console.print("[bold blue]Finding maximum coverage positions...[/bold blue]")
    
    # Step 1: Create a set of all DMRs in each sample
    sample_dmrs = {}
    for sample, df in sample_dfs.items():
        # Add DMR key column
        df['dmr_key'] = df.apply(create_dmr_key, axis=1)
        # Store unique DMRs for this sample
        sample_dmrs[sample] = set(df['dmr_key'].unique())
    
    # Step 2: Find DMRs present in all samples
    common_dmrs = set.intersection(*sample_dmrs.values()) if sample_dmrs else set()
    
    if not common_dmrs:
        raise ValueError("No common DMRs found across all samples")
    
    console.print(f"[bold green]Found {len(common_dmrs)} common DMRs across all samples[/bold green]")
    
    # Step 3: For each common DMR, find the position with max coverage in each sample
    dmr_max_positions = {}
    
    for dmr_key in tqdm(common_dmrs, desc="Processing DMRs"):
        # Dictionary to store position info and coverage for each sample
        pos_coverage = defaultdict(list)
        pos_info = {}
        
        # Find all positions in this DMR for each sample
        for sample, df in sample_dfs.items():
            dmr_df = df[df['dmr_key'] == dmr_key]
            
            # If DMR has no positions in this sample, skip to next DMR
            if dmr_df.empty:
                continue
                
            # For each position in this DMR, record its coverage
            for _, row in dmr_df.iterrows():
                pos_key = (row['chr'], row['pos'])
                pos_coverage[pos_key].append(row['coverage'])
                
                # Store position info (only need to do this once)
                if pos_key not in pos_info:
                    pos_info[pos_key] = {
                        'chr': row['chr'],
                        'pos': row['pos'],
                        'chr_dmr': row['chr_dmr'],
                        'start_dmr': row['start_dmr'],
                        'end_dmr': row['end_dmr']
                    }
        
        # Filter positions that appear in all samples
        all_samples_count = len(sample_dfs)
        complete_positions = {pos: covs for pos, covs in pos_coverage.items() if len(covs) == all_samples_count}
        
        # If no positions in this DMR are covered in all samples, skip this DMR
        if not complete_positions:
            continue
            
        # Find the position with the highest mean coverage
        max_mean_cov = 0
        max_pos = None
        
        for pos, covs in complete_positions.items():
            mean_cov = sum(covs) / len(covs)
            if mean_cov > max_mean_cov:
                max_mean_cov = mean_cov
                max_pos = pos
        
        # Store the max coverage position for this DMR
        if max_pos:
            info = pos_info[max_pos]
            dmr_max_positions[dmr_key] = {
                'chr': info['chr'],
                'pos': info['pos'],
                'mean_coverage': max_mean_cov,
                'chr_dmr': info['chr_dmr'],
                'start_dmr': info['start_dmr'],
                'end_dmr': info['end_dmr']
            }
    
    # Convert to DataFrame and sort by DMR coordinates
    if not dmr_max_positions:
        raise ValueError("No maximum coverage positions found that are covered in all samples")
    
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
@click.option('--verbose', is_flag=True, help='Print verbose output')
def main(input, output, verbose):
    """
    Find maximum coverage positions for each DMR across multiple samples.
    
    Takes multiple CSV files with coverage information, finds the position with
    maximum coverage in each DMR across all samples, and outputs a CSV file with
    these positions along with their mean coverage and DMR information.
    
    Input CSV files must have columns: 'chr', 'pos', 'coverage', 'chr_dmr', 'start_dmr', 'end_dmr'
    """
    start_time = time.time()
    
    try:
        # Expand wildcards if any
        input_files = []
        for pattern in input:
            matched_files = glob.glob(pattern)
            if not matched_files:
                console.print(f"[bold yellow]Warning:[/bold yellow] No files match pattern '{pattern}'", style="yellow")
            input_files.extend(matched_files)
        
        if not input_files:
            raise ValueError("No input files found")
        
        console.print(Panel.fit(
            f"[bold green]Processing [/bold green]{len(input_files)}[bold green] coverage files[/bold green]\n"
            f"[bold green]Output will be saved to:[/bold green] {output}",
            title="Maximum Coverage Position Analysis", 
            border_style="blue"
        ))
        
        # Read all input CSV files
        sample_dfs = read_coverage_files(input_files)
        
        # Find maximum coverage positions
        max_cov_df = find_max_coverage_positions(sample_dfs)
        
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
