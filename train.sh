#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --partition=gpumedium
#SBATCH --gpus-per-node=a100.20gb:1
#SBATCH --job-name=versi_controiality
#SBATCH --mem=20G
module purge
module load CUDA/11.7.0
module load cuDNN/8.4.1.50-CUDA-11.7.0
module load NCCL/2.12.12-GCCcore-11.3.0-CUDA-11.7.0
module load Python/3.10.4-GCCcore-11.3.0
module load GCC/11.3.0
export
HF_DATASETS_CACHE="/scratch/$USER/.cache/huggingface/datasets"
export
TRANSFORMERS_CACHE="/scratch/$USER/.cache/huggingface/transformer
s"
python main.py