#!/usr/bin/bash
#SBATCH -J extract_memory_efficient
#SBATCH --partition=cn-long
#SBATCH --cpus-per-task=128

singularity exec -B /lustre1,/lustre2,/appsnew /lustre1/cqyi/syfan/images/common_tools.sif \
python3 extract_from_parquet.py \
--input $1 \
--output $2 \
--ncpus 128 \
--sample_col $3 \
--labels_to_keep Normal \
--labels_to_keep T16 \
--threshold 0 \
--mode first