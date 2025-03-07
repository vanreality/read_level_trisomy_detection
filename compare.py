import pandas as pd
import numpy as np
import os
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.Align import PairwiseAligner
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import re
import seaborn as sns
from tqdm import tqdm
import click
import warnings
import umap
import multiprocessing
from functools import partial
from itertools import combinations_with_replacement
warnings.filterwarnings('ignore')

# Function to calculate sequence similarity (using global alignment)
def calculate_similarity(seq1, seq2):
    aligner = PairwiseAligner()
    aligner.mode = 'global'
    aligner.match_score = 1.0
    aligner.mismatch_score = -1.0
    aligner.open_gap_score = -2.0
    aligner.extend_gap_score = -1.0
    
    # Get the best alignment score
    score = aligner.score(seq1, seq2)
    max_len = max(len(seq1), len(seq2))
    if max_len == 0:
        return 0
    return score / max_len

# Function to calculate similarity between two samples
def calculate_sample_similarity(sample_pair, all_sequences, dmr_indices):
    sample1, sample2 = sample_pair
    
    # Calculate similarity across all DMRs
    dmr_similarities = []
    for dmr_idx in dmr_indices:
        if dmr_idx in all_sequences[sample1] and dmr_idx in all_sequences[sample2]:
            seq1 = all_sequences[sample1][dmr_idx]
            seq2 = all_sequences[sample2][dmr_idx]
            similarity = calculate_similarity(seq1, seq2)
            dmr_similarities.append(similarity)
    
    # Average similarity across all DMRs
    if dmr_similarities:
        similarity = np.mean(dmr_similarities)
    else:
        similarity = 0
        
    return (sample1, sample2, similarity)

@click.command()
@click.option('--samplesheet', required=True, help='Path to the samplesheet CSV file')
@click.option('--fasta-dir', required=True, help='Directory containing FASTA files')
@click.option('--output-dir', default='./plot', help='Output directory for plots (default: ./plot)')
@click.option('--fasta-pattern', default='{sample}_{label}.fa', 
              help='Pattern for FASTA filenames, must include {sample} and {label} placeholders (default: {sample}_{label}_longest_pentabase.fa)')
@click.option('--ncpus', default=1, type=int, help='Number of CPU cores to use for parallel processing (default: 1)')
def main(samplesheet, fasta_dir, output_dir, fasta_pattern, ncpus):
    """
    DMR sequence similarity analysis tool.
    
    This tool analyzes sequence similarity across DMRs from multiple samples,
    performs PCA and UMAP, and generates visualization plots.
    """
    # Validate fasta pattern
    if '{sample}' not in fasta_pattern or '{label}' not in fasta_pattern:
        raise ValueError("FASTA pattern must include both {sample} and {label} placeholders")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Read sample sheet
    df = pd.read_csv(samplesheet)
    
    # Construct FASTA file paths based on sample and label using the pattern
    df['fasta'] = df.apply(
        lambda row: os.path.join(fasta_dir, fasta_pattern.format(sample=row['sample'], label=row['label'])), 
        axis=1
    )
    
    df = df[['sample', 'label', 'fasta']]
    
    # Read all FASTA files and organize sequences by DMR index
    all_sequences = {}
    dmr_indices = set()
    
    # First pass: collect all DMR indices and sequences
    for _, row in tqdm(df.iterrows(), desc="Reading FASTA files", total=len(df)):
        sample = row['sample']
        fasta_file = row['fasta']
        all_sequences[sample] = {}
        
        try:
            for record in SeqIO.parse(fasta_file, "fasta"):
                # Extract DMR index from header
                match = re.search(r'dmr_index:(\d+)', record.description)
                if match:
                    dmr_index = int(match.group(1))
                    dmr_indices.add(dmr_index)
                    all_sequences[sample][dmr_index] = str(record.seq)
        except Exception as e:
            print(f"Error reading {fasta_file}: {e}")
    
    # Second pass: find DMRs missing from any sample
    samples = df['sample'].unique()
    dmr_indices = sorted(dmr_indices)
    dropped_dmrs = set()
    
    for dmr_idx in dmr_indices:
        for sample in samples:
            if dmr_idx not in all_sequences[sample]:
                dropped_dmrs.add(dmr_idx)
                break
    
    # Remove dropped DMRs from all_sequences and dmr_indices
    for dmr_idx in dropped_dmrs:
        for sample in samples:
            if dmr_idx in all_sequences[sample]:
                del all_sequences[sample][dmr_idx]
    
    dmr_indices = sorted(set(dmr_indices) - dropped_dmrs)
    
    print(f"Found {len(dmr_indices)} DMRs across {len(samples)} samples")
    print(f"Dropped {len(dropped_dmrs)} DMRs that were missing from some samples: {sorted(dropped_dmrs)}")
    
    # Create a map from label to numeric value for coloring
    unique_labels = df['label'].unique()
    label_to_num = {label: i for i, label in enumerate(unique_labels)}
    num_to_label = {i: label for i, label in enumerate(unique_labels)}
    
    # Generate n*n similarity matrix
    print(f"Generating full similarity matrix using {ncpus} CPU cores...")
    similarity_matrix = np.zeros((len(samples), len(samples)))
    
    # Set up multiprocessing pool
    pool = multiprocessing.Pool(processes=ncpus)
    
    # Generate all pairs of samples to process
    sample_pairs = list(combinations_with_replacement(samples, 2))
    
    # Create a partial function with fixed parameters
    calc_func = partial(calculate_sample_similarity, 
                         all_sequences=all_sequences, 
                         dmr_indices=dmr_indices)
    
    # Calculate similarities in parallel
    results = list(tqdm(
        pool.imap(calc_func, sample_pairs),
        total=len(sample_pairs),
        desc="Calculating similarity matrix"
    ))
    
    # Close the pool
    pool.close()
    pool.join()
    
    # Fill in the similarity matrix
    sample_to_idx = {sample: i for i, sample in enumerate(samples)}
    for sample1, sample2, similarity in results:
        i = sample_to_idx[sample1]
        j = sample_to_idx[sample2]
        similarity_matrix[i, j] = similarity
        # Fill in the symmetric part
        if i != j:
            similarity_matrix[j, i] = similarity
    
    # Save similarity matrix to CSV
    sim_df = pd.DataFrame(similarity_matrix, index=samples, columns=samples)
    sim_df.to_csv(f"{output_dir}/similarity.csv")
    print(f"Similarity matrix saved to {output_dir}/similarity.csv")
    
    # Get labels for each sample
    sample_labels = [df[df['sample'] == sample]['label'].values[0] for sample in samples]
    numeric_labels = [label_to_num[label] for label in sample_labels]
    
    # Calculate silhouette score if there are at least 2 clusters and more than 2 samples
    sil_score = "N/A"
    if len(set(numeric_labels)) > 1 and len(samples) > 2:
        try:
            sil_score = silhouette_score(similarity_matrix, numeric_labels)
            sil_score = f"{sil_score:.4f}"
        except Exception as e:
            print(f"Error calculating silhouette score: {e}")
    
    # Create colormap
    cmap = ListedColormap(sns.color_palette("husl", len(unique_labels)))
    
    # Apply PCA for dimension reduction
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(similarity_matrix)
    
    # PCA Plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1], 
                          c=numeric_labels, cmap=cmap, s=100, alpha=0.8)
    
    # Add labels to points
    for i, sample in enumerate(samples):
        plt.annotate(sample, (pca_result[i, 0], pca_result[i, 1]), 
                     fontsize=8, alpha=0.8)
    
    # Add legend
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                               label=label, markerfacecolor=cmap(label_to_num[label]), 
                               markersize=10) for label in unique_labels]
    plt.legend(handles=legend_elements, title="Labels")
    
    # Set title and labels
    plt.title(f"DMR Similarity Analysis - PCA\nSilhouette Score: {sil_score}")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)")
    
    # Save figure
    plt.tight_layout()
    plt.savefig(f"{output_dir}/similarity_pca.pdf")
    plt.close()
    
    # Apply UMAP for dimension reduction
    reducer = umap.UMAP()
    try:
        umap_result = reducer.fit_transform(similarity_matrix)
        
        # UMAP Plot
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(umap_result[:, 0], umap_result[:, 1], 
                              c=numeric_labels, cmap=cmap, s=100, alpha=0.8)
        
        # Add labels to points
        for i, sample in enumerate(samples):
            plt.annotate(sample, (umap_result[i, 0], umap_result[i, 1]), 
                         fontsize=8, alpha=0.8)
        
        # Add legend
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                   label=label, markerfacecolor=cmap(label_to_num[label]), 
                                   markersize=10) for label in unique_labels]
        plt.legend(handles=legend_elements, title="Labels")
        
        # Set title and labels
        plt.title(f"DMR Similarity Analysis - UMAP\nSilhouette Score: {sil_score}")
        plt.xlabel("UMAP Dimension 1")
        plt.ylabel("UMAP Dimension 2")
        
        # Save figure
        plt.tight_layout()
        plt.savefig(f"{output_dir}/similarity_umap.pdf")
        plt.close()
    except Exception as e:
        print(f"Error generating UMAP plot: {e}")
    
    print(f"Analysis complete. Plots saved to {output_dir}/ directory.")

if __name__ == '__main__':
    main()



