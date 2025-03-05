import pandas as pd
import numpy as np
import os
from Bio import SeqIO
from Bio.Seq import Seq
from Bio import pairwise2
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import re
import seaborn as sns
from tqdm import tqdm
import click
import warnings
warnings.filterwarnings('ignore')

# Function to calculate sequence similarity (using global alignment)
def calculate_similarity(seq1, seq2):
    alignment = pairwise2.align.globalxx(seq1, seq2, one_alignment_only=True)[0]
    score = alignment.score
    max_len = max(len(seq1), len(seq2))
    if max_len == 0:
        return 0
    return score / max_len

@click.command()
@click.option('--samplesheet', required=True, help='Path to the samplesheet CSV file')
@click.option('--fasta-dir', required=True, help='Directory containing FASTA files')
@click.option('--output-dir', default='./plot', help='Output directory for plots (default: ./plot)')
@click.option('--fasta-pattern', default='{sample}_{label}_longest_pentabase.fa', 
              help='Pattern for FASTA filenames, must include {sample} and {label} placeholders (default: {sample}_{label}_longest_pentabase.fa)')
def main(samplesheet, fasta_dir, output_dir, fasta_pattern):
    """
    DMR sequence similarity analysis tool.
    
    This tool analyzes sequence similarity across DMRs from multiple samples,
    performs PCA, and generates visualization plots.
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
    
    dmr_indices = sorted(dmr_indices)
    samples = df['sample'].unique()
    print(f"Found {len(dmr_indices)} DMRs across {len(samples)} samples")
    
    # Create a map from label to numeric value for coloring
    unique_labels = df['label'].unique()
    label_to_num = {label: i for i, label in enumerate(unique_labels)}
    num_to_label = {i: label for i, label in enumerate(unique_labels)}
    
    # For each sample as center, calculate similarity matrix and visualize
    for center_sample in tqdm(samples, desc="Processing samples"):
        # Create feature matrix [n_samples * n_dmrs]
        feature_matrix = np.zeros((len(samples), len(dmr_indices)))
        
        for i, sample in enumerate(samples):
            for j, dmr_idx in enumerate(dmr_indices):
                # Check if both samples have this DMR
                if dmr_idx in all_sequences[center_sample] and dmr_idx in all_sequences[sample]:
                    center_seq = all_sequences[center_sample][dmr_idx]
                    sample_seq = all_sequences[sample][dmr_idx]
                    similarity = calculate_similarity(center_seq, sample_seq)
                    feature_matrix[i, j] = similarity
                else:
                    feature_matrix[i, j] = 0
        
        # Apply PCA for dimension reduction
        pca = PCA(n_components=2)
        reduced_features = pca.fit_transform(feature_matrix)
        
        # Get labels for each sample
        sample_labels = [df[df['sample'] == sample]['label'].values[0] for sample in samples]
        numeric_labels = [label_to_num[label] for label in sample_labels]
        
        # Calculate silhouette score if there are at least 2 clusters and more than 2 samples
        sil_score = "N/A"
        if len(set(numeric_labels)) > 1 and len(samples) > 2:
            try:
                sil_score = silhouette_score(reduced_features, numeric_labels)
                sil_score = f"{sil_score:.4f}"
            except Exception as e:
                print(f"Error calculating silhouette score: {e}")
        
        # Create colormap
        cmap = ListedColormap(sns.color_palette("husl", len(unique_labels)))
        
        # Plot
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(reduced_features[:, 0], reduced_features[:, 1], 
                              c=numeric_labels, cmap=cmap, s=100, alpha=0.8)
        
        # Add labels to points
        for i, sample in enumerate(samples):
            plt.annotate(sample, (reduced_features[i, 0], reduced_features[i, 1]), 
                         fontsize=8, alpha=0.8)
        
        # Add legend
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                   label=label, markerfacecolor=cmap(label_to_num[label]), 
                                   markersize=10) for label in unique_labels]
        plt.legend(handles=legend_elements, title="Labels")
        
        # Set title and labels
        plt.title(f"DMR Similarity Analysis - Center: {center_sample}\nSilhouette Score: {sil_score}")
        plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)")
        plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)")
        
        # Save figure
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{center_sample}.pdf")
        plt.close()
    
    print(f"Analysis complete. Plots saved to {output_dir}/ directory.")

if __name__ == '__main__':
    main()



