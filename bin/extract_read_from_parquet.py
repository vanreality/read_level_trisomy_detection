#!/usr/bin/env python

import os
import sys
import click
import pandas as pd
import pyarrow.parquet as pq
import numpy as np
import multiprocessing as mp
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm
from rich.console import Console
from rich.panel import Panel
from functools import partial
import time

# Initialize console for rich output
console = Console()

def read_max_coverage_file(max_coverage_file):
    """
    Read and process the maximum coverage positions file.
    
    Args:
        max_coverage_file (str): Path to the max coverage CSV file
        
    Returns:
        pd.DataFrame: DataFrame containing max coverage position data
        
    Raises:
        ValueError: If required columns are missing
    """
    try:
        df = pd.read_csv(max_coverage_file)
        
        # Validate required columns
        required_columns = ['chr', 'pos', 'mean_coverage', 'dmr_index', 'chr_dmr', 'start_dmr', 'end_dmr']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"Max coverage file is missing required columns: {', '.join(missing_columns)}")
        
        return df
    
    except Exception as e:
        console.print(f"[bold red]Error reading max coverage file:[/bold red] {str(e)}", style="red")
        raise

def read_methylation_file(methylation_file):
    """
    Read and process the methylation data file.
    
    Args:
        methylation_file (str): Path to the methylation CSV file
        
    Returns:
        pd.DataFrame: DataFrame containing methylation data
        
    Raises:
        ValueError: If required columns are missing
    """
    try:
        df = pd.read_csv(methylation_file)
        
        # Validate required columns
        required_columns = ['chr', 'start', 'end', 'status', 'prob_class_1']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"Methylation file is missing required columns: {', '.join(missing_columns)}")
        
        return df
    
    except Exception as e:
        console.print(f"[bold red]Error reading methylation file:[/bold red] {str(e)}", style="red")
        raise

def calculate_methylation_stats(methylation_df, dmr):
    """
    Calculate methylation statistics for CpG sites within a DMR.
    
    Args:
        methylation_df (pd.DataFrame): DataFrame containing methylation data
        dmr (dict): DMR information including coordinates
        
    Returns:
        tuple: (raw_methylation_list, prob_weighted_methylation_list)
    """
    # Filter methylation data for the current DMR
    dmr_methylation = methylation_df[
        (methylation_df['chr'] == dmr['chr_dmr']) & 
        (methylation_df['start'] >= dmr['start_dmr']) & 
        (methylation_df['end'] <= dmr['end_dmr'])
    ]
    
    if dmr_methylation.empty:
        return [], []
    
    # Group by CpG site coordinates
    grouped = dmr_methylation.groupby(['chr', 'start', 'end'])
    
    raw_methylation_list = []
    prob_weighted_methylation_list = []
    
    for _, group in grouped:
        # Calculate raw methylation rate
        raw_rate = group['status'].sum() / len(group)
        raw_methylation_list.append(raw_rate)
        
        # Calculate probability-weighted methylation rate
        prob_sum = group['prob_class_1'].sum()
        if prob_sum > 0:  # Avoid division by zero
            weighted_rate = (group['status'] * group['prob_class_1']).sum() / prob_sum
            prob_weighted_methylation_list.append(weighted_rate)
        else:
            prob_weighted_methylation_list.append(0.0)
    
    return raw_methylation_list, prob_weighted_methylation_list

def process_dmr(dmr_row, parquet_file, methylation_df, mode):
    """
    Process a single DMR to select and extract a read.
    
    Args:
        dmr_row (pd.Series): Row from max coverage file for a single DMR
        parquet_file (str): Path to the parquet file
        methylation_df (pd.DataFrame): Methylation data
        mode (str): Selection mode ('longest' or 'prob_largest')
        
    Returns:
        tuple: (dmr_index, read_data, methylation_stats) or None if no suitable read found
    """
    try:
        # Extract DMR information
        dmr = {
            'chr': dmr_row['chr'],
            'pos': dmr_row['pos'],
            'dmr_index': dmr_row['dmr_index'],
            'chr_dmr': dmr_row['chr_dmr'],
            'start_dmr': dmr_row['start_dmr'],
            'end_dmr': dmr_row['end_dmr']
        }
        
        # Read parquet data for this specific region
        table = pq.read_table(
            parquet_file,
            filters=[
                ('chr', '=', dmr['chr']),
                ('start', '<=', dmr['pos']),
                ('end', '>=', dmr['pos'])
            ]
        )
        reads_df = table.to_pandas()
        
        if reads_df.empty:
            return None
        
        # Select the best read based on the specified mode
        if mode == 'longest':
            # Add length column based on sequence length
            reads_df['length'] = reads_df['seq'].str.len()
            # Sort by length (descending) and take the first read
            reads_df = reads_df.sort_values('length', ascending=False)
        elif mode == 'prob_largest':
            # Sort by prob_class_1 (descending) and take the first read
            reads_df = reads_df.sort_values('prob_class_1', ascending=False)
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        # Get the best read
        if not reads_df.empty:
            best_read = reads_df.iloc[0]
            
            # Calculate methylation statistics for this DMR
            raw_meth, prob_meth = calculate_methylation_stats(methylation_df, dmr)
            
            return {
                'dmr_index': dmr['dmr_index'],
                'read': {
                    'name': best_read['name'],
                    'seq': best_read['seq'],
                    'prob_class_1': best_read['prob_class_1'],
                    'chr_dmr': dmr['chr_dmr'],
                    'start_dmr': dmr['start_dmr'],
                    'end_dmr': dmr['end_dmr']
                },
                'methylation': {
                    'raw': raw_meth,
                    'prob_weighted': prob_meth
                }
            }
        
        return None
    
    except Exception as e:
        console.print(f"[bold red]Error processing DMR {dmr_row['dmr_index']}:[/bold red] {str(e)}", style="red")
        return None

def extract_reads_parallel(max_coverage_df, parquet_file, methylation_df, mode, num_processes):
    """
    Extract reads from the parquet file based on max coverage positions using parallel processing.
    
    Args:
        max_coverage_df (pd.DataFrame): DataFrame with max coverage positions
        parquet_file (str): Path to the parquet file
        methylation_df (pd.DataFrame): DataFrame with methylation data
        mode (str): Selection mode ('longest' or 'prob_largest')
        num_processes (int): Number of processes to use
        
    Returns:
        list: List of selected read data
    """
    # Create a partial function with fixed parameters
    process_func = partial(
        process_dmr,
        parquet_file=parquet_file,
        methylation_df=methylation_df,
        mode=mode
    )
    
    console.print(f"[bold blue]Extracting reads using [/bold blue]{num_processes}[bold blue] processes...[/bold blue]")
    
    # Use multiprocessing to process DMRs in parallel
    with mp.Pool(processes=num_processes) as pool:
        results = list(tqdm(
            pool.imap(process_func, [row for _, row in max_coverage_df.iterrows()]),
            total=len(max_coverage_df),
            desc="Processing DMRs",
            unit="DMR"
        ))
    
    # Filter out None results (failed DMRs)
    selected_reads = [r for r in results if r is not None]
    
    return selected_reads

def write_fasta(selected_reads, output_file):
    """
    Write selected reads to a FASTA file.
    
    Args:
        selected_reads (list): List of selected read data
        output_file (str): Path to the output FASTA file
        
    Returns:
        int: Number of reads written
    """
    try:
        with open(output_file, 'w') as f:
            for read_data in selected_reads:
                dmr_index = read_data['dmr_index']
                read = read_data['read']
                methylation = read_data['methylation']
                
                # Format raw and weighted methylation lists
                raw_meth_str = ','.join([f"{x:.3f}" for x in methylation['raw']])
                prob_meth_str = ','.join([f"{x:.3f}" for x in methylation['prob_weighted']])
                
                # Create header
                header = (
                    f"read_name:{read['name']},"
                    f"dmr_index:{dmr_index},"
                    f"dmr_coordinate:{read['chr_dmr']}-{read['start_dmr']}-{read['end_dmr']},"
                    f"prob_class_1:{read['prob_class_1']:.3f},"
                    f"raw_methylation_vector:[{raw_meth_str}],"
                    f"prob_weighted_methylation_vector:[{prob_meth_str}]"
                )
                
                # Write header and sequence
                f.write(f">{header}\n")
                f.write(f"{read['seq']}\n")
        
        return len(selected_reads)
    
    except Exception as e:
        console.print(f"[bold red]Error writing FASTA file:[/bold red] {str(e)}", style="red")
        raise

@click.command()
@click.option('--input', required=True, help='Path to the input parquet file with read data')
@click.option('--max_coverage', required=True, help='Path to the max coverage CSV file')
@click.option('--methylation', required=True, help='Path to the methylation CSV file')
@click.option('--mode', required=True, type=click.Choice(['longest', 'prob_largest']), 
              help='Read selection mode: longest or prob_largest')
@click.option('--prefix', required=True, help='Prefix for output files')
@click.option('--ncpus', type=int, default=None, help='Number of processes to use (default: number of CPU cores)')
@click.option('--verbose', is_flag=True, help='Print verbose output')
def main(input, max_coverage, methylation, mode, prefix, ncpus, verbose):
    """
    Extract reads from a parquet file based on maximum coverage positions.
    
    This script processes a parquet file containing read data, a max coverage
    file containing the positions with maximum coverage for each DMR, and a
    methylation file with CpG methylation data. For each DMR, it selects one
    read based on the specified selection mode and generates a FASTA file with
    the selected reads and their methylation information.
    
    The selection mode can be either 'longest' (select the longest read) or
    'prob_largest' (select the read with the largest probability score).
    """
    start_time = time.time()
    
    try:
        # Validate input files
        for file_path, file_desc in [
            (input, "Input parquet"),
            (max_coverage, "Max coverage CSV"),
            (methylation, "Methylation CSV")
        ]:
            if not os.path.exists(file_path):
                console.print(f"[bold red]Error:[/bold red] {file_desc} file '{file_path}' does not exist", style="red")
                sys.exit(1)
        
        # Set number of processes
        if ncpus is None:
            ncpus = mp.cpu_count()
        
        # Define output file path
        output_file = f"{prefix}.fa"
        
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            console.print(f"Created output directory: {output_dir}")
        
        console.print(Panel.fit(
            f"[bold green]Extracting reads from:[/bold green] {input}\n"
            f"[bold green]Using max coverage positions from:[/bold green] {max_coverage}\n"
            f"[bold green]Using methylation data from:[/bold green] {methylation}\n"
            f"[bold green]Selection mode:[/bold green] {mode}\n"
            f"[bold green]Output will be saved to:[/bold green] {output_file}",
            title="Read Extraction Parameters", 
            border_style="blue"
        ))
        
        # Read input files
        console.print("[bold blue]Reading input files...[/bold blue]")
        max_coverage_df = read_max_coverage_file(max_coverage)
        methylation_df = read_methylation_file(methylation)
        
        if verbose:
            console.print(f"[green]Found {len(max_coverage_df)} DMRs in max coverage file[/green]")
            console.print(f"[green]Found {len(methylation_df)} methylation records[/green]")
        
        # Extract reads using parallel processing
        selected_reads = extract_reads_parallel(
            max_coverage_df,
            input,
            methylation_df,
            mode,
            ncpus
        )
        
        # Write selected reads to FASTA file
        console.print(f"[bold blue]Writing {len(selected_reads)} selected reads to[/bold blue] {output_file}")
        num_written = write_fasta(selected_reads, output_file)
        
        # Print statistics
        elapsed_time = time.time() - start_time
        console.print(Panel.fit(
            f"[bold green]Total DMRs processed:[/bold green] {len(max_coverage_df):,}\n"
            f"[bold green]Reads selected and written:[/bold green] {num_written:,}\n"
            f"[bold green]Processing time:[/bold green] {elapsed_time:.2f} seconds",
            title="Read Extraction Summary", 
            border_style="green"
        ))
        
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}", style="red")
        sys.exit(1)

if __name__ == "__main__":
    main()
