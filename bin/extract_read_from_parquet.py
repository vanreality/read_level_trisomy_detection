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

def prefilter_methylation_data(methylation_df, max_coverage_df):
    """
    Prefilter methylation data for each DMR.
    
    Args:
        methylation_df (pd.DataFrame): Complete methylation data
        max_coverage_df (pd.DataFrame): Max coverage data with DMR information
        
    Returns:
        dict: Dictionary mapping DMR indices to filtered methylation data
    """
    dmr_methylation_dict = {}
    
    for _, dmr_row in tqdm(max_coverage_df.iterrows(), total=len(max_coverage_df), 
                           desc="Prefiltering methylation data", unit="DMR"):
        dmr_index = dmr_row['dmr_index']
        chr_dmr = dmr_row['chr_dmr']
        start_dmr = dmr_row['start_dmr']
        end_dmr = dmr_row['end_dmr']
        
        # Filter methylation data for this DMR
        dmr_methylation = methylation_df[
            (methylation_df['chr'] == chr_dmr) & 
            (methylation_df['start'] >= start_dmr) & 
            (methylation_df['end'] <= end_dmr)
        ]
        
        dmr_methylation_dict[dmr_index] = dmr_methylation
    
    return dmr_methylation_dict

def read_and_prefilter_parquet(parquet_file, max_coverage_df, chunk_size=1000000):
    """
    Read parquet file once and prefilter data for each DMR.
    
    Args:
        parquet_file (str): Path to the parquet file
        max_coverage_df (pd.DataFrame): Max coverage data with DMR information
        chunk_size (int): Size of chunks to read from parquet file
        
    Returns:
        dict: Dictionary mapping DMR indices to filtered read data
    """
    # Create a dictionary to store reads by DMR
    dmr_reads_dict = defaultdict(list)
    
    # Get table metadata to check total rows
    parquet_metadata = pq.read_metadata(parquet_file)
    total_rows = parquet_metadata.num_rows
    
    console.print(f"[bold blue]Reading parquet file with [/bold blue]{total_rows:,}[bold blue] rows in chunks...[/bold blue]")
    
    # Read the parquet file in chunks to avoid memory issues
    reader = pq.ParquetFile(parquet_file)
    
    for i, batch in enumerate(tqdm(reader.iter_batches(batch_size=chunk_size), 
                                   total=(total_rows + chunk_size - 1) // chunk_size,
                                   desc="Processing parquet chunks", unit="chunk")):
        # Convert batch to DataFrame
        chunk_df = batch.to_pandas()
        
        # Process each DMR in the max coverage file
        for _, dmr_row in max_coverage_df.iterrows():
            dmr_index = dmr_row['dmr_index']
            chr_val = dmr_row['chr']
            pos = dmr_row['pos']
            
            # Filter reads that cover this position
            filtered_reads = chunk_df[
                (chunk_df['chr'] == chr_val) &
                (chunk_df['start'] <= pos) &
                (chunk_df['end'] >= pos)
            ]
            
            if not filtered_reads.empty:
                dmr_reads_dict[dmr_index].append(filtered_reads)
    
    # Concatenate DataFrames for each DMR
    for dmr_index in dmr_reads_dict:
        if dmr_reads_dict[dmr_index]:
            dmr_reads_dict[dmr_index] = pd.concat(dmr_reads_dict[dmr_index], ignore_index=True)
        else:
            dmr_reads_dict[dmr_index] = pd.DataFrame()
    
    return dmr_reads_dict

def calculate_methylation_stats(filtered_methylation_df, read_start=None, read_end=None):
    """
    Calculate methylation statistics for CpG sites within a DMR, masking sites outside the read region.
    
    Args:
        filtered_methylation_df (pd.DataFrame): Pre-filtered methylation data for a DMR
        read_start (int, optional): Start position of the read. If provided, CpG sites before this position are masked
        read_end (int, optional): End position of the read. If provided, CpG sites after this position are masked
        
    Returns:
        tuple: (raw_methylation_list, prob_weighted_methylation_list, cpg_positions)
    """
    if filtered_methylation_df.empty:
        return [], [], []
    
    # Group by CpG site coordinates
    grouped = filtered_methylation_df.groupby(['chr', 'start', 'end'])
    
    raw_methylation_list = []
    prob_weighted_methylation_list = []
    cpg_positions = []
    
    for (chr_val, start, end), group in grouped:
        # Record the CpG position
        cpg_positions.append((chr_val, start, end))
        
        # Check if this CpG site is within the read region
        if (read_start is not None and read_end is not None and 
            (start < read_start or end > read_end)):
            # CpG site is outside the read region, mask with 0
            raw_methylation_list.append(0.0)
            prob_weighted_methylation_list.append(0.0)
        else:
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
    
    return raw_methylation_list, prob_weighted_methylation_list, cpg_positions

def process_dmr(dmr_data, mode):
    """
    Process a single DMR to select and extract a read.
    
    Args:
        dmr_data (tuple): (dmr_row, filtered_reads_df, filtered_methylation_df)
        mode (str): Selection mode ('longest' or 'prob_largest')
        
    Returns:
        tuple: (dmr_index, read_data, methylation_stats) or None if no suitable read found
    """
    try:
        dmr_row, filtered_reads_df, filtered_methylation_df = dmr_data
        
        # Extract DMR information
        dmr = {
            'chr': dmr_row['chr'],
            'pos': dmr_row['pos'],
            'dmr_index': dmr_row['dmr_index'],
            'chr_dmr': dmr_row['chr_dmr'],
            'start_dmr': dmr_row['start_dmr'],
            'end_dmr': dmr_row['end_dmr']
        }
        
        if filtered_reads_df.empty:
            return None
        
        # Select the best read based on the specified mode
        if mode == 'longest':
            # Add length column based on sequence length
            filtered_reads_df['length'] = filtered_reads_df['seq'].str.len()
            # Sort by length (descending) and take the first read
            filtered_reads_df = filtered_reads_df.sort_values('length', ascending=False)
        elif mode == 'prob_largest':
            # Sort by prob_class_1 (descending) and take the first read
            filtered_reads_df = filtered_reads_df.sort_values('prob_class_1', ascending=False)
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        # Get the best read
        if not filtered_reads_df.empty:
            best_read = filtered_reads_df.iloc[0]
            
            # Calculate methylation statistics for this DMR, masking sites outside the read region
            raw_meth, prob_meth, cpg_positions = calculate_methylation_stats(
                filtered_methylation_df, 
                read_start=best_read['start'],
                read_end=best_read['end']
            )
            
            return {
                'dmr_index': dmr['dmr_index'],
                'read': {
                    'name': best_read['name'],
                    'seq': best_read['seq'],
                    'prob_class_1': best_read['prob_class_1'],
                    'chr_dmr': dmr['chr_dmr'],
                    'start_dmr': dmr['start_dmr'],
                    'end_dmr': dmr['end_dmr'],
                    'start': best_read['start'],
                    'end': best_read['end'],
                    'chr': best_read['chr']
                },
                'methylation': {
                    'raw': raw_meth,
                    'prob_weighted': prob_meth,
                    'cpg_positions': cpg_positions
                }
            }
        
        return None
    
    except Exception as e:
        console.print(f"[bold red]Error processing DMR {dmr_row['dmr_index']}:[/bold red] {str(e)}", style="red")
        return None

def extract_reads_parallel(max_coverage_df, dmr_reads_dict, dmr_methylation_dict, mode, num_processes):
    """
    Extract reads using parallel processing with pre-filtered data.
    
    Args:
        max_coverage_df (pd.DataFrame): DataFrame with max coverage positions
        dmr_reads_dict (dict): Dictionary mapping DMR indices to filtered read data
        dmr_methylation_dict (dict): Dictionary mapping DMR indices to filtered methylation data
        mode (str): Selection mode ('longest' or 'prob_largest')
        num_processes (int): Number of processes to use
        
    Returns:
        list: List of selected read data
    """
    # Create process data list
    process_data_list = []
    
    for _, dmr_row in max_coverage_df.iterrows():
        dmr_index = dmr_row['dmr_index']
        filtered_reads_df = dmr_reads_dict.get(dmr_index, pd.DataFrame())
        filtered_methylation_df = dmr_methylation_dict.get(dmr_index, pd.DataFrame())
        
        process_data_list.append((dmr_row, filtered_reads_df, filtered_methylation_df))
    
    console.print(f"[bold blue]Extracting reads using [/bold blue]{num_processes}[bold blue] processes...[/bold blue]")
    
    # Create a partial function with fixed parameters
    process_func = partial(process_dmr, mode=mode)
    
    # Use multiprocessing to process DMRs in parallel
    with mp.Pool(processes=num_processes) as pool:
        results = list(tqdm(
            pool.imap(process_func, process_data_list),
            total=len(process_data_list),
            desc="Processing DMRs",
            unit="DMR"
        ))
    
    # Filter out None results (failed DMRs)
    selected_reads = [r for r in results if r is not None]
    
    return selected_reads

def write_tsv(selected_reads, output_file):
    """
    Write selected reads to a TSV file.
    
    Args:
        selected_reads (list): List of selected read data
        output_file (str): Path to the output TSV file
        
    Returns:
        int: Number of reads written
    """
    try:
        # Create a list to store all rows
        rows = []
        
        for read_data in selected_reads:
            dmr_index = read_data['dmr_index']
            read = read_data['read']
            methylation = read_data['methylation']
            
            # Format raw and weighted methylation lists with commas
            # TSV format uses tabs as separators so commas within fields won't cause parsing issues
            raw_meth_str = ','.join([f"{x:.3f}" for x in methylation['raw']])
            prob_meth_str = ','.join([f"{x:.3f}" for x in methylation['prob_weighted']])
            
            # Create a row for the DataFrame
            row = {
                'read_name': read['name'],
                'dmr_index': dmr_index,
                'chr': read['chr_dmr'],
                'dmr_start': read['start_dmr'],
                'dmr_end': read['end_dmr'],
                'read_start': read['start'],
                'read_end': read['end'],
                'prob_class_1': read['prob_class_1'],
                'raw_methylation_vector': raw_meth_str,
                'prob_weighted_methylation_vector': prob_meth_str,
                'sequence': read['seq']
            }
            
            rows.append(row)
        
        # Create and write DataFrame to TSV (tab-separated values)
        df = pd.DataFrame(rows)
        df.to_csv(output_file, index=False, sep='\t')
        
        return len(rows)
    
    except Exception as e:
        console.print(f"[bold red]Error writing TSV file:[/bold red] {str(e)}", style="red")
        raise

@click.command()
@click.option('--input', required=True, help='Path to the input parquet file with read data')
@click.option('--max_coverage', required=True, help='Path to the max coverage CSV file')
@click.option('--methylation', required=True, help='Path to the methylation CSV file')
@click.option('--mode', required=True, type=click.Choice(['longest', 'prob_largest']), 
              help='Read selection mode: longest or prob_largest')
@click.option('--prefix', required=True, help='Prefix for output files')
@click.option('--ncpus', type=int, default=None, help='Number of processes to use (default: number of CPU cores)')
@click.option('--chunk_size', type=int, default=1000000, help='Size of chunks to read from parquet file')
@click.option('--verbose', is_flag=True, help='Print verbose output')
def main(input, max_coverage, methylation, mode, prefix, ncpus, chunk_size, verbose):
    """
    Extract reads from a parquet file based on maximum coverage positions.
    
    This script processes a parquet file containing read data, a max coverage
    file containing the positions with maximum coverage for each DMR, and a
    methylation file with CpG methylation data. For each DMR, it selects one
    read based on the specified selection mode and generates a TSV file with
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
        
        # Define output file path with .tsv extension
        output_file = f"{prefix}.tsv"
        
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
            f"[bold green]Chunk size:[/bold green] {chunk_size:,}\n"
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
        
        # Prefilter methylation data for each DMR
        dmr_methylation_dict = prefilter_methylation_data(methylation_df, max_coverage_df)
        
        # Free up memory after prefiltering
        del methylation_df
        
        # Read and prefilter parquet data for each DMR
        dmr_reads_dict = read_and_prefilter_parquet(input, max_coverage_df, chunk_size)
        
        # Extract reads using parallel processing with prefiltered data
        selected_reads = extract_reads_parallel(
            max_coverage_df,
            dmr_reads_dict,
            dmr_methylation_dict,
            mode,
            ncpus
        )
        
        # Free up memory before writing output
        del dmr_reads_dict
        del dmr_methylation_dict
        
        # Write selected reads to TSV file
        console.print(f"[bold blue]Writing {len(selected_reads)} selected reads to[/bold blue] {output_file}")
        num_written = write_tsv(selected_reads, output_file)
        
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
