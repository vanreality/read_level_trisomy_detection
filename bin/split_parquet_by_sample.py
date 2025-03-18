import os
import click
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
import time
from tqdm import tqdm
from tabulate import tabulate

def get_parquet_columns(input_file):
    """
    Get the list of column names in a parquet file.
    
    Args:
        input_file: Path to the input parquet file
        
    Returns:
        List of column names
    """
    parquet_file = pq.ParquetFile(input_file)
    schema = parquet_file.schema
    return schema.names

def get_sample_column(input_file):
    """
    Determine which sample column to use in the parquet file.
    Checks for 'sampledown' first, then 'sample'.
    
    Args:
        input_file: Path to the input parquet file
        
    Returns:
        Column name to use for sample identification
        
    Raises:
        ValueError: If neither 'sampledown' nor 'sample' columns exist
    """
    columns = get_parquet_columns(input_file)
    
    if 'sampledown' in columns:
        return 'sampledown'
    elif 'sample' in columns:
        return 'sample'
    else:
        raise ValueError("Neither 'sampledown' nor 'sample' columns found in the parquet file")

def get_unique_samples(input_file, sample_column):
    """
    Efficiently extract unique sample names from a parquet file
    without loading the entire dataset into memory.
    
    Args:
        input_file: Path to the input parquet file
        sample_column: Column name to use for sample identification
        
    Returns:
        List of unique sample names
    """
    parquet_file = pq.ParquetFile(input_file)
    # Only read the sample column
    sample_table = parquet_file.read([sample_column])
    # Convert to pandas, get unique values and convert to list
    unique_samples = sample_table.to_pandas()[sample_column].unique().tolist()
    return unique_samples

def process_sample(sample_name, label, input_file, output_dir, sample_column, verbose=False):
    """
    Process a single sample from the parquet file and save it to a separate file.
    
    Args:
        sample_name: Name of the sample
        label: Label of the sample
        input_file: Path to the input parquet file
        output_dir: Directory to save output files
        sample_column: Column name to use for sample identification
        verbose: Whether to print additional information
        
    Returns:
        Dict with sample information and file path
    """
    output_filename = f"{sample_name}.parquet"
    output_file = os.path.join(output_dir, output_filename)
    
    # Use pyarrow.parquet.read_table with filters parameter for better memory efficiency
    filters = [(sample_column, '=', sample_name)]
    sample_table = pq.read_table(input_file, filters=filters)
    
    # Write the subset to a new parquet file
    pq.write_table(sample_table, output_file)
    
    # Return sample information with absolute file path
    return {
        'sample': sample_name,
        'label': label,
        'parquet': os.path.abspath(output_file)
    }

@click.command()
@click.option('--input', required=True, help='Path to the input parquet file')
@click.option('--output-dir', default='.', help='Directory to save output files')
@click.option('--samplesheet', default='samplesheet.csv', help='Path to output samplesheet CSV')
@click.option('--verbose', is_flag=True, help='Print verbose output')
def main(input, output_dir, samplesheet, verbose):
    """
    Split a parquet file into multiple files based on the sample column.
    Each output file will be named {sample}.parquet.
    Generates a samplesheet.csv with sample, label, and parquet file paths.
    """
    start_time = time.time()
    
    # Check if the input file exists
    if not os.path.exists(input):
        click.echo(f"Error: Input file '{input}' does not exist", err=True)
        return
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        click.echo(f"Created output directory: {output_dir}")
    
    try:
        # Determine which sample column to use
        sample_column = get_sample_column(input)
        click.echo(f"Using '{sample_column}' column for sample identification")
        
        # Get the unique sample names
        click.echo(f"Reading sample information from '{input}'...")
        sample_names = get_unique_samples(input, sample_column)
        
        click.echo(f"Found {len(sample_names)} unique samples. Starting processing...")
        
        # Need to read labels for each sample
        parquet_file = pq.ParquetFile(input)
        sample_label_table = parquet_file.read([sample_column, 'label'])
        sample_label_df = sample_label_table.to_pandas().drop_duplicates()
        sample_to_label = dict(zip(sample_label_df[sample_column], sample_label_df['label']))
        
        # Process the samples sequentially with a progress bar
        results = []
        for sample_name in tqdm(sample_names, desc="Processing samples"):
            label = sample_to_label.get(sample_name, None)
            result = process_sample(sample_name, label, input, output_dir, sample_column, verbose)
            results.append(result)
            # Explicitly call garbage collection to free memory if needed
            if verbose:
                click.echo(f"Processed sample: {sample_name} with label: {label}")
                
        # Create a DataFrame from the results
        samplesheet_df = pd.DataFrame(results)
        
        # Write the samplesheet to a CSV file
        samplesheet_path = os.path.join(output_dir, samplesheet)
        samplesheet_df.to_csv(samplesheet_path, index=False)
        
        elapsed_time = time.time() - start_time
        click.echo(f"Processing completed in {elapsed_time:.2f} seconds")
        click.echo(f"Wrote {len(results)} samples to individual parquet files")
        click.echo(f"Generated samplesheet: {os.path.abspath(samplesheet_path)}")
        
        # Display sample of the samplesheet
        if verbose and not samplesheet_df.empty:
            sample_rows = min(5, len(samplesheet_df))
            click.echo("\nSample of samplesheet.csv:")
            click.echo(tabulate(samplesheet_df.head(sample_rows), headers='keys', tablefmt='pretty'))
            
    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        return

if __name__ == "__main__":
    main()