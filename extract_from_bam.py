import click
import pandas as pd
import pysam
import random
import numpy as np
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO

@click.command()
@click.option('--samplesheet', default='samplesheet.csv', help='Path to the CSV sample sheet file')
@click.option('--bed', default='senddmr_igtc.bed', help='Path to the BED file with DMR regions')
@click.option('--threads', default=4, help='Number of threads to use for processing')
@click.option('--read_selection', type=click.Choice(['longest', 'shortest', 'random']), default='longest',
              help='Strategy for selecting reads at max coverage positions')
@click.option('--bam_column', default='bam', help='Column name in samplesheet that contains BAM file paths')
@click.option('--output_dir', default='output', help='Output directory')
def main(samplesheet, bed, threads, read_selection, bam_column, output_dir):
    """
    Process a CSV sample sheet and a BED file of DMR regions.
    For each DMR region:
    1. Calculate average coverage across all samples and find the position with maximum average coverage
    2. For each sample, select a read (longest/shortest/random) covering that position
    3. Output FASTA files with the selected reads
    """
    # Create output directory and subdirectories
    subdirs = [
        os.path.join(output_dir, "bam"),
        os.path.join(output_dir, "fasta"),
        os.path.join(output_dir, "fasta_pentabase"),
        os.path.join(output_dir, "merged_bam"),
        os.path.join(output_dir, "merged_bam_pentabase")
    ]
    
    for directory in subdirs:
        if not os.path.exists(directory):
            os.makedirs(directory)
            click.echo(f"Created directory: {directory}")
    
    # Load the sample sheet (must contain columns: sample, label, and the specified bam_column)
    try:
        samples_df = pd.read_csv(samplesheet)
    except Exception as e:
        click.echo(f"Error reading samplesheet: {e}")
        return

    # Create soft links and index BAM files
    bam_dir = os.path.join(output_dir, "bam")
    samples_df = setup_bam_files(samples_df, bam_dir, bam_column)
    
    # Load the bed file (assuming at least three columns: chrom, start, end)
    try:
        # Read bed file with no header; if your file contains a header, consider using skiprows=1
        dmr_df = pd.read_csv(bed, sep='\t', header=None, names=['chrom', 'start', 'end'], usecols=[0, 1, 2])
    except Exception as e:
        click.echo(f"Error reading bed file: {e}")
        return

    # Convert 'start' and 'end' columns to numeric, dropping any rows that fail conversion.
    dmr_df['start'] = pd.to_numeric(dmr_df['start'], errors='coerce')
    dmr_df['end'] = pd.to_numeric(dmr_df['end'], errors='coerce')
    if dmr_df[['start', 'end']].isnull().any().any():
        click.echo("Warning: Some rows in the bed file have non-numeric start or end values and will be skipped.")
    dmr_df = dmr_df.dropna(subset=['start', 'end'])
    dmr_df['start'] = dmr_df['start'].astype(int)
    dmr_df['end'] = dmr_df['end'].astype(int)
    
    # Initialize BED file for maximum coverage positions in output directory
    max_cov_pos_path = os.path.join(output_dir, 'max_cov_pos.bed')
    
    with open(max_cov_pos_path, 'w') as bed_file:
        bed_file.write('#chr\tstart\tend\tname\n')
    
    # Run the process
    click.echo(f"\n=== Processing BAM files from column '{bam_column}' ===")
    process_bams(samples_df, dmr_df, threads, read_selection, bam_column, output_dir)

def setup_bam_files(samples_df, bam_dir, bam_column):
    """
    Create soft links to BAM files in the bam directory and index them.
    Updates the DataFrame with new paths to the linked files.
    """
    new_df = samples_df.copy()
    
    for i, row in samples_df.iterrows():
        # Process BAM file
        if bam_column in row and pd.notna(row[bam_column]):
            original_bam = row[bam_column]
            filename = os.path.basename(original_bam)
            linked_bam = os.path.join(bam_dir, filename)
            
            # Create soft link if it doesn't exist
            if not os.path.exists(linked_bam):
                try:
                    os.symlink(os.path.abspath(original_bam), linked_bam)
                    click.echo(f"Created soft link for {filename} in {bam_dir}")
                except Exception as e:
                    click.echo(f"Error creating soft link for {filename}: {e}")
            
            # Create index if it doesn't exist
            if not os.path.exists(f"{linked_bam}.bai"):
                try:
                    subprocess.run(['samtools', 'index', linked_bam], check=True)
                    click.echo(f"Created index for {filename}")
                except Exception as e:
                    click.echo(f"Error creating index for {filename}: {e}")
            
            # Update path in DataFrame
            new_df.at[i, bam_column] = linked_bam
    
    return new_df

def process_bams(samples_df, dmr_df, threads, read_selection, bam_column, output_dir):
    """
    Run the complete process for BAM files.
    
    Args:
        samples_df: DataFrame containing sample information
        dmr_df: DataFrame containing DMR regions
        threads: Number of threads to use
        read_selection: Strategy for selecting reads
        bam_column: Column name in samples_df containing BAM file paths
        output_dir: Directory to store output files
    """
    click.echo(f"Step 1: Calculating average coverage for all DMRs across all samples using {bam_column}...")
    max_coverage_positions = calculate_max_average_coverage_positions(samples_df, dmr_df, threads, bam_column)
    
    # Write maximum coverage positions to BED file
    write_max_coverage_bed(max_coverage_positions, output_dir)
    
    click.echo(f"Step 2: Filtering DMRs and selecting reads for each sample from {bam_column}...")
    valid_dmrs_with_reads = select_reads_for_all_samples(samples_df, max_coverage_positions, read_selection, threads, bam_column)
    
    if valid_dmrs_with_reads:
        click.echo(f"Step 3: Writing FASTA files...")
        write_fasta_files(valid_dmrs_with_reads, read_selection, output_dir)
        
        click.echo(f"Step 4: Creating merged BAM files by label...")
        create_merged_bam_by_label(samples_df, valid_dmrs_with_reads, read_selection, bam_column, output_dir)
        
        click.echo(f"Process completed successfully. Selected {len(valid_dmrs_with_reads)} valid DMRs for {bam_column}.")
    else:
        click.echo(f"No valid DMRs found where all samples have reads covering the max coverage positions in {bam_column}.")

def calculate_max_average_coverage_positions(samples_df, dmr_df, threads, bam_column):
    """
    For each DMR, calculate the average coverage across all samples and find the position
    with maximum average coverage.
    
    Returns a dictionary mapping DMR indices to (chrom, start, end, max_pos, max_coverage) tuples.
    """
    # Initialize a dictionary to store coverage data for each DMR
    dmr_coverage_data = {}
    
    # Process each DMR
    for dmr_index, region in dmr_df.iterrows():
        chrom = region['chrom']
        start = int(region['start'])
        end = int(region['end'])
        region_length = end - start
        
        # Skip regions with no length
        if region_length <= 0:
            continue
            
        # Initialize coverage array for this region
        all_coverages = np.zeros((len(samples_df), region_length), dtype=int)
        
        # Process each sample to calculate coverage
        with ThreadPoolExecutor(max_workers=threads) as executor:
            futures = []
            
            for sample_idx, row in samples_df.iterrows():
                if bam_column in row and pd.notna(row[bam_column]):
                    bam_path = row[bam_column]
                    
                    # Add task for bam file
                    futures.append(executor.submit(get_coverage_for_region, bam_path, chrom, start, end, sample_idx))
            
            # Collect results
            for future in as_completed(futures):
                try:
                    sample_idx, coverage = future.result()
                    if coverage is not None and len(coverage) == region_length:
                        all_coverages[sample_idx] = coverage
                except Exception as e:
                    click.echo(f"Error calculating coverage: {e}")
        
        # Calculate average coverage across all samples
        avg_coverage = np.mean(all_coverages, axis=0)
        
        # Find position with maximum average coverage
        if len(avg_coverage) > 0:
            max_index = np.argmax(avg_coverage)
            max_pos = start + max_index
            max_coverage = avg_coverage[max_index]
            dmr_coverage_data[dmr_index] = (chrom, start, end, max_pos, max_coverage)
    
    return dmr_coverage_data

def write_max_coverage_bed(max_coverage_positions, output_dir):
    """
    Write the maximum coverage positions to a BED file.
    Creates a file in the output directory.
    """
    try:
        # Determine the file path
        bed_file_path = os.path.join(output_dir, "max_cov_pos.bed")
        
        with open(bed_file_path, 'a') as bed_file:
            for dmr_idx, (chrom, start, end, max_pos, max_coverage) in max_coverage_positions.items():
                # Format: chrom, max_pos, max_pos+1, name
                name = f"DMR,{chrom}:{start}-{end},max_coverage:{max_coverage:.2f}"
                bed_file.write(f"{chrom}\t{max_pos}\t{max_pos+1}\t{name}\n")
                
        click.echo(f"Added {len(max_coverage_positions)} regions to {bed_file_path}")
    except Exception as e:
        click.echo(f"Error writing max coverage positions to {bed_file_path}: {e}")

def get_coverage_for_region(bam_path, chrom, start, end, sample_index):
    """
    Calculate coverage for a specific region in a BAM file.
    Returns the sample index and an array of coverage values.
    """
    try:
        bamfile = pysam.AlignmentFile(bam_path, "rb")
        # Get coverage for all bases in the region, with quality_threshold=0 to handle "*" in basequality
        coverage_tuple = bamfile.count_coverage(chrom, start, end, quality_threshold=0)
        bamfile.close()
        
        # Sum coverage across all bases (A, C, G, T)
        total_coverage = [sum(bases) for bases in zip(*coverage_tuple)]
        
        return sample_index, total_coverage
    except Exception as e:
        click.echo(f"Error getting coverage from {bam_path} for region {chrom}:{start}-{end}: {e}")
        return sample_index, None

def select_reads_for_all_samples(samples_df, max_coverage_positions, read_selection, threads, bam_column='bam'):
    """
    For each DMR with a maximum coverage position, attempt to select reads
    from all samples. If any sample doesn't have a read at the position,
    the DMR is dropped.
    
    Returns a dictionary of valid DMRs with selected reads for all samples.
    """
    valid_dmrs_with_reads = {}
    # Count samples with valid BAM files for the specified column
    valid_sample_count = sum(1 for _, row in samples_df.iterrows() if bam_column in row and pd.notna(row[bam_column]))
    
    for dmr_index, (chrom, start, end, max_pos, max_coverage) in max_coverage_positions.items():
        all_sample_reads = {}
        
        with ThreadPoolExecutor(max_workers=threads) as executor:
            futures = []
            
            # Process each sample
            for _, row in samples_df.iterrows():
                if bam_column in row and pd.notna(row[bam_column]):
                    sample = row['sample']
                    label = row['label']
                    bam_path = row[bam_column]
                    
                    # Submit task to get reads from BAM file
                    futures.append(executor.submit(
                        get_reads_at_position, sample, label, bam_path, chrom, max_pos, read_selection
                    ))
            
            # Collect results
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result:
                        sample_key, read_record, original_read = result
                        all_sample_reads[sample_key] = (read_record, original_read)
                except Exception as e:
                    click.echo(f"Error selecting reads: {e}")
        
        # Check if all samples have reads for this DMR
        if len(all_sample_reads) == valid_sample_count:
            valid_dmrs_with_reads[dmr_index] = (chrom, start, end, all_sample_reads)
    
    return valid_dmrs_with_reads

def get_reads_at_position(sample, label, bam_path, chrom, position, read_selection):
    """
    For a given sample and position, select a read according to the specified strategy.
    Returns a tuple of (sample_key, SeqRecord, original_read) if a read is found, None otherwise.
    """
    try:
        bamfile = pysam.AlignmentFile(bam_path, "rb")
        candidate_reads = []
        
        # Use pileup to get reads covering the position
        for pileupcolumn in bamfile.pileup(chrom, position, position+1, truncate=True):
            if pileupcolumn.reference_pos == position:
                for pileupread in pileupcolumn.pileups:
                    if not pileupread.is_del and not pileupread.is_refskip:
                        candidate_reads.append(pileupread.alignment)
        
        bamfile.close()
        
        if not candidate_reads:
            return None
        
        # Select the appropriate read based on the strategy
        selected_read = None
        if read_selection == 'longest':
            selected_read = max(candidate_reads, key=lambda r: r.query_length if r.query_length else 0)
        elif read_selection == 'shortest':
            selected_read = min(candidate_reads, key=lambda r: r.query_length if r.query_length else float('inf'))
        else:  # random
            selected_read = random.choice(candidate_reads)
        
        if selected_read and selected_read.query_sequence:
            sample_key = f"{sample}_{label}"
            
            # Get sequence mapping coordinates
            seq_start = selected_read.reference_start
            seq_end = selected_read.reference_end if selected_read.reference_end else (seq_start + selected_read.query_length)
            
            # Create a SeqRecord from the selected read
            record = SeqRecord(
                seq=selected_read.query_sequence,
                id=selected_read.query_name,
                description="",  # Will be updated later with DMR info
                annotations={
                    'seq_start': seq_start,
                    'seq_end': seq_end
                }
            )
            
            return (sample_key, record, selected_read)
        
        return None
    except Exception as e:
        click.echo(f"Error getting reads from {bam_path} at position {chrom}:{position}: {e}")
        return None

def write_fasta_files(valid_dmrs_with_reads, read_selection, output_dir):
    """
    Write the selected reads to FASTA files, one file per sample.
    The files will be placed in the output_dir/fasta directory.
    """
    # Group reads by sample
    sample_reads = {}
    
    # Create fasta directory path
    fasta_dir = os.path.join(output_dir, "fasta")
    
    # Renumber DMRs after filtering
    dmr_renumbering = {old_idx: new_idx for new_idx, old_idx in enumerate(sorted(valid_dmrs_with_reads.keys()))}
    
    # Organize reads by sample
    for dmr_idx, (chrom, start, end, all_sample_reads) in valid_dmrs_with_reads.items():
        new_dmr_idx = dmr_renumbering[dmr_idx]
        
        for sample_key, (record, _) in all_sample_reads.items():
            if sample_key not in sample_reads:
                sample_reads[sample_key] = []
            
            # Split the sample_key to get the sample name
            parts = sample_key.split('_')
            sample = parts[0]
            
            # Update description with the new DMR index, seq_mapping information, and read_name
            record.description = f"dmr_index:{new_dmr_idx},dmr_coordiante:{chrom}:{start}-{end},seq_mapping:{chrom}:{record.annotations['seq_start']}-{record.annotations['seq_end']},read_name:{sample}_{record.id}"
            sample_reads[sample_key].append(record)
    
    # Write FASTA files for each sample
    for sample_key, records in sample_reads.items():
        parts = sample_key.split('_')
        sample = parts[0]
        label = parts[1]
        
        # Create the output filename
        out_filename = os.path.join(fasta_dir, f"{sample}_{label}_{read_selection}.fa")
        
        # Write the FASTA file
        try:
            with open(out_filename, "w") as f_out:
                SeqIO.write(records, f_out, "fasta")
            click.echo(f"Written {out_filename} with {len(records)} reads")
        except Exception as e:
            click.echo(f"Error writing output {out_filename}: {e}")

def create_merged_bam_by_label(samples_df, valid_dmrs_with_reads, read_selection, bam_column, output_dir):
    """
    Create merged BAM files for each label, containing all reads from samples with that label.
    Reads are renamed with sample prefix, and output BAMs are sorted and indexed.
    All output files are placed in the output_dir/merged_bam directory.
    """
    # Create merged_bam directory path
    merged_bam_dir = os.path.join(output_dir, "merged_bam")
    
    # Group reads by label
    label_reads = {}
    label_header = {}
    
    # First, get all labels and create structures
    for _, row in samples_df.iterrows():
        label = row['label']
        if label not in label_reads:
            label_reads[label] = []
            label_header[label] = None
    
    # Collect reads for each label
    for dmr_idx, (chrom, start, end, all_sample_reads) in valid_dmrs_with_reads.items():
        for sample_key, (_, original_read) in all_sample_reads.items():
            parts = sample_key.split('_')
            sample = parts[0]
            label = parts[1]
            
            # Create a copy of the read to modify its name
            modified_read = original_read
            modified_read.query_name = f"{sample}_{original_read.query_name}"
            
            # Store the modified read for this label
            label_reads[label].append(modified_read)
            
            # If we don't have a header for this label yet, use the one from this read's BAM file
            if label_header[label] is None:
                # Find the sample's BAM file to get the header
                for _, row in samples_df.iterrows():
                    if row['label'] == label:
                        if bam_column in row and pd.notna(row[bam_column]):
                            bam_path = row[bam_column]
                            try:
                                bamfile = pysam.AlignmentFile(bam_path, "rb")
                                label_header[label] = bamfile.header
                                bamfile.close()
                                break
                            except Exception as e:
                                click.echo(f"Error getting header from {bam_path}: {e}")
    
    # Write BAM files for each label
    for label, reads in label_reads.items():
        if reads and label_header[label]:
            # Create unsorted BAM first
            out_filename = os.path.join(merged_bam_dir, f"{label}_{read_selection}.bam")
            sorted_filename = os.path.join(merged_bam_dir, f"{label}_{read_selection}.sorted.bam")
            
            try:
                # Create a BAM file with the collected reads
                out_bam = pysam.AlignmentFile(out_filename, "wb", header=label_header[label])
                for read in reads:
                    out_bam.write(read)
                out_bam.close()
                
                click.echo(f"Written {out_filename} with {len(reads)} reads")
                
                # Sort the BAM file
                pysam.sort("-o", sorted_filename, out_filename)
                click.echo(f"Sorted BAM file written to {sorted_filename}")
                
                # Remove the unsorted BAM
                os.remove(out_filename)
                
                # Index the sorted BAM file
                pysam.index(sorted_filename)
                click.echo(f"Indexed sorted BAM file {sorted_filename}")
                
                # Run pentabase_conv with outputs in the output directory
                bam_prefix = os.path.join(output_dir, f"{label}_{read_selection}")
                bed_gz_path = f"{bam_prefix}.bed.gz"
                cmd = f"bash ./pentabase_conv --bam-file {sorted_filename} --ref-file hg38_only_chromsomes.fa --output-bed {bed_gz_path}"
                try:
                    os.system(cmd)
                    click.echo(f"Ran pentabase_conv on {sorted_filename}")
                    
                    # Process the pentabase bed.gz output to create new files with modified sequences
                    process_pentabase_output(bed_gz_path, sorted_filename, label, read_selection, output_dir)
                    
                except Exception as e:
                    click.echo(f"Error running pentabase_conv on {sorted_filename}: {e}")
                
            except Exception as e:
                click.echo(f"Error processing BAM file {out_filename}: {e}")

def process_pentabase_output(bed_gz_path, bam_path, label, read_selection, output_dir):
    """
    Process the pentabase_conv output file to replace sequences in FASTA and BAM files.
    
    Args:
        bed_gz_path: Path to the bed.gz file produced by pentabase_conv
        bam_path: Path to the BAM file that was used for pentabase_conv
        label: Label for the current group
        read_selection: Strategy used for selecting reads
        output_dir: Directory to store output files
    """
    try:
        click.echo(f"Processing pentabase output from {bed_gz_path}")
        
        # Maps to store read name to new sequence mappings
        read_to_sequence = {}
        
        # Parse the bed.gz file to extract read names and new sequences
        import gzip
        
        # Check if the file exists
        if not os.path.exists(bed_gz_path):
            click.echo(f"Warning: Pentabase output file {bed_gz_path} does not exist. Skipping processing.")
            return
        
        # Open the gzipped bed file and read the contents
        with gzip.open(bed_gz_path, 'rt') as bed_file:
            for line in bed_file:
                if line.startswith('#'):
                    continue
                
                # Split the line into columns
                cols = line.strip().split('\t')
                if len(cols) >= 7:
                    read_name = cols[3]
                    new_sequence = cols[6]
                    
                    # Store the mapping
                    read_to_sequence[read_name] = new_sequence
        
        click.echo(f"Extracted {len(read_to_sequence)} read-to-sequence mappings from {bed_gz_path}")
        
        # Create new FASTA files with pentabase sequences
        create_pentabase_fasta_files(read_to_sequence, label, read_selection, output_dir)
        
        # Create new BAM files with pentabase sequences
        create_pentabase_bam_file(read_to_sequence, bam_path, label, read_selection, output_dir)
        
    except Exception as e:
        click.echo(f"Error processing pentabase output: {e}")

def create_pentabase_fasta_files(read_to_sequence, label, read_selection, output_dir):
    """
    Create new FASTA files with sequences replaced by pentabase sequences.
    
    Args:
        read_to_sequence: Dictionary mapping read names to new sequences
        label: Label for the current group
        read_selection: Strategy used for selecting reads
        output_dir: Directory to store output files
    """
    try:
        # Define input and output directories
        fasta_dir = os.path.join(output_dir, "fasta")
        fasta_pentabase_dir = os.path.join(output_dir, "fasta_pentabase")
        
        # Find all FASTA files in the fasta directory that match the pattern
        fasta_pattern = os.path.join(fasta_dir, f"*_{label}_{read_selection}.fa")
        import glob
        
        for fasta_path in glob.glob(fasta_pattern):
            # Create a new filename in the pentabase directory
            base_name = os.path.basename(fasta_path)
            base_name_no_ext = os.path.splitext(base_name)[0]
            pentabase_fasta_path = os.path.join(fasta_pentabase_dir, f"{base_name_no_ext}_pentabase.fa")
            
            # Read the original FASTA file
            records = []
            for record in SeqIO.parse(fasta_path, "fasta"):
                # Extract the original read name from the description
                description_parts = record.description.split(',')
                read_name = None
                for part in description_parts:
                    if part.startswith("read_name:"):
                        read_name = part.split(':', 1)[1]
                        break
                
                if read_name and read_name in read_to_sequence:
                    # Create a new record with the pentabase sequence
                    from Bio.Seq import Seq
                    new_record = record
                    new_record.seq = Seq(read_to_sequence[read_name])
                    records.append(new_record)
                else:
                    # Keep the original record if no replacement found
                    records.append(record)
            
            # Write the new FASTA file
            with open(pentabase_fasta_path, "w") as f_out:
                SeqIO.write(records, f_out, "fasta")
            
            click.echo(f"Written pentabase FASTA file {pentabase_fasta_path} with {len(records)} reads")
    
    except Exception as e:
        click.echo(f"Error creating pentabase FASTA files: {e}")

def create_pentabase_bam_file(read_to_sequence, bam_path, label, read_selection, output_dir):
    """
    Create a new BAM file with sequences replaced by pentabase sequences.
    
    Args:
        read_to_sequence: Dictionary mapping read names to new sequences
        bam_path: Path to the original BAM file
        label: Label for the current group
        read_selection: Strategy used for selecting reads
        output_dir: Directory to store output files
    """
    try:
        # Define input and output directories
        merged_bam_pentabase_dir = os.path.join(output_dir, "merged_bam_pentabase")
        
        # Create a new filename in the pentabase BAM directory
        base_name = os.path.basename(bam_path)
        base_name_no_ext = os.path.splitext(base_name)[0]
        pentabase_bam_path = os.path.join(merged_bam_pentabase_dir, f"{base_name_no_ext}_pentabase.bam")
        
        # Open the original BAM file
        with pysam.AlignmentFile(bam_path, "rb") as input_bam:
            # Create a new BAM file with the same header
            with pysam.AlignmentFile(pentabase_bam_path, "wb", header=input_bam.header) as output_bam:
                # Process each read in the input BAM
                for read in input_bam:
                    read_name = read.query_name
                    if read_name in read_to_sequence:
                        # Replace the sequence with the pentabase sequence
                        new_sequence = read_to_sequence[read_name]
                        
                        # Create a copy of the read
                        read.query_sequence = new_sequence
                        
                        # Adjust qualities if necessary
                        if read.query_qualities is not None and len(read.query_qualities) != len(new_sequence):
                            # Set qualities to a default value (30 is a good quality score)
                            read.query_qualities = pysam.array.array('B', [30] * len(new_sequence))
                    
                    # Write the read to the output BAM
                    output_bam.write(read)
        
        # Sort and index the new BAM file
        sorted_pentabase_bam_path = os.path.join(merged_bam_pentabase_dir, f"{base_name_no_ext}_pentabase.sorted.bam")
        pysam.sort("-o", sorted_pentabase_bam_path, pentabase_bam_path)
        
        # Remove the unsorted BAM
        os.remove(pentabase_bam_path)
        
        # Index the sorted BAM file
        pysam.index(sorted_pentabase_bam_path)
        
        click.echo(f"Created pentabase BAM file: {sorted_pentabase_bam_path}")
    
    except Exception as e:
        click.echo(f"Error creating pentabase BAM file: {e}")

if __name__ == '__main__':
    main()