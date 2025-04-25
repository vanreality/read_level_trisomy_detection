import os
import click
import polars as pl
import time
from tqdm import tqdm
from rich.console import Console
from rich.table import Table

console = Console()

def get_parquet_columns(input_file):
    """
    Get the list of column names in a parquet file.
    
    Args:
        input_file: Path to the input parquet file
        
    Returns:
        List of column names
    """
    try:
        schema = pl.read_parquet_schema(input_file)
        return list(schema.keys())
    except Exception as e:
        raise ValueError(f"Error reading parquet schema: {e}")

def get_unique_samples(input_file):
    """
    Efficiently extract unique sample names from a parquet file
    without loading the entire dataset into memory.
    
    Args:
        input_file: Path to the input parquet file
        
    Returns:
        List of unique sample names
        
    Raises:
        ValueError: If 'sample' column not found in the parquet file
    """
    try:
        columns = get_parquet_columns(input_file)
        if 'sample' not in columns:
            raise ValueError("'sample' column not found in the parquet file")
            
        # Only read the sample column
        sample_column = pl.scan_parquet(input_file).select(['sample'])
        unique_samples = sample_column.unique().collect()['sample'].to_list()
        return unique_samples
    except Exception as e:
        raise ValueError(f"Error getting unique samples: {e}")

def process_sample(sample_name, input_file, output_dir, verbose=False):
    """
    Process a single sample from the parquet file and save it to a separate file.
    
    Args:
        sample_name: Name of the sample
        input_file: Path to the input parquet file
        output_dir: Directory to save output files
        verbose: Whether to print additional information
        
    Returns:
        Dict with sample information and file path
    """
    output_filename = f"{sample_name}.parquet"
    output_file = os.path.join(output_dir, output_filename)
    
    # Use polars for better memory efficiency
    sample_df = pl.scan_parquet(input_file).filter(pl.col('sample') == sample_name).collect()
    
    # Write the subset to a new parquet file
    sample_df.write_parquet(output_file)
    
    # Get the label for the sample (first occurrence)
    label = None
    if 'label' in sample_df.columns:
        label_values = sample_df.select('label').unique().to_series()
        if len(label_values) > 0:
            label = label_values[0]
    
    # Return sample information with absolute file path
    return {
        'sample': sample_name,
        'label': label,
        'parquet': os.path.abspath(output_file)
    }

def extract_metadata(input_file, output_dir, output_filename='metadata.csv'):
    """
    Extract metadata from the parquet file and save it to a CSV file.
    
    Args:
        input_file: Path to the input parquet file
        output_dir: Directory to save output files
        output_filename: Name of the output metadata file
        
    Returns:
        Path to the metadata file
    """
    metadata_columns = ['sample', 'label', 'state', 'age', 'panel', 'MEAN_TARGET_COVERAGE', 'week']
    
    # Check which columns exist in the parquet file
    available_columns = get_parquet_columns(input_file)
    columns_to_select = [col for col in metadata_columns if col in available_columns]
    
    if 'sample' not in columns_to_select:
        raise ValueError("'sample' column not found in the parquet file")
    
    # Read only the needed columns from the parquet file
    metadata_df = pl.scan_parquet(input_file).select(columns_to_select).unique(subset=['sample']).collect()
    
    # Ensure columns are in the specified order
    metadata_df = metadata_df.select(metadata_columns)
    
    # Write metadata to CSV
    output_path = os.path.join(output_dir, output_filename)
    metadata_df.write_csv(output_path)
    
    return os.path.abspath(output_path)

@click.command()
@click.option('--input', required=True, help='Path to the input parquet file')
@click.option('--output-dir', default='.', help='Directory to save output files')
@click.option('--samplesheet', default='samplesheet.csv', help='Path to output samplesheet CSV')
@click.option('--metadata', default='metadata.csv', help='Path to output metadata CSV')
@click.option('--verbose', is_flag=True, help='Print verbose output')
def main(input, output_dir, samplesheet, metadata, verbose):
    """
    Split a parquet file into multiple files based on the sample column.
    Each output file will be named {sample}.parquet.
    Generates a samplesheet.csv with sample, label, and parquet file paths.
    Also creates a metadata.csv with sample information.
    """
    start_time = time.time()
    
    # Check if the input file exists
    if not os.path.exists(input):
        console.print(f"[bold red]Error:[/] Input file '{input}' does not exist")
        return
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        console.print(f"Created output directory: [cyan]{output_dir}[/]")
    
    try:
        # Get the unique sample names
        console.print(f"Reading sample information from '[cyan]{input}[/]'...")
        sample_names = get_unique_samples(input)
        
        console.print(f"Found [green]{len(sample_names)}[/] unique samples. Starting processing...")
        
        # Process the samples sequentially with a progress bar
        results = []
        for sample_name in tqdm(sample_names, desc="Processing samples"):
            result = process_sample(sample_name, input, output_dir, verbose)
            results.append(result)
            if verbose:
                console.print(f"Processed sample: [cyan]{sample_name}[/] with label: [yellow]{result['label']}[/]")
                
        # Create a DataFrame from the results and write to CSV
        samplesheet_df = pl.DataFrame(results)
        samplesheet_path = os.path.join(output_dir, samplesheet)
        samplesheet_df.write_csv(samplesheet_path)
        
        # Extract and save metadata
        console.print("Extracting metadata...")
        metadata_path = extract_metadata(input, output_dir, metadata)
        
        elapsed_time = time.time() - start_time
        console.print(f"Processing completed in [green]{elapsed_time:.2f}[/] seconds")
        console.print(f"Wrote [green]{len(results)}[/] samples to individual parquet files")
        console.print(f"Generated samplesheet: [cyan]{os.path.abspath(samplesheet_path)}[/]")
        console.print(f"Generated metadata: [cyan]{metadata_path}[/]")
        
        # Display sample of the samplesheet
        if verbose and len(results) > 0:
            sample_rows = min(5, len(results))
            table = Table(title="Sample of samplesheet.csv")
            
            # Add columns to the table
            for col in samplesheet_df.columns:
                table.add_column(col, style="cyan")
            
            # Add rows to the table
            for row in samplesheet_df.head(sample_rows).iter_rows(named=True):
                table.add_row(*[str(row[col]) for col in samplesheet_df.columns])
            
            console.print(table)
            
    except ValueError as e:
        console.print(f"[bold red]Error:[/] {e}")
        return
    except Exception as e:
        console.print(f"[bold red]Unexpected error:[/] {e}")
        return

if __name__ == "__main__":
    main()