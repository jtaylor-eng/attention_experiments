#!/bin/bash
#SBATCH -A PAS2119
#SBATCH --partition=batch
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --job-name=safe-test
#SBATCH --output=std_out_err/safe_test.%j.out
#SBATCH --error=std_out_err/safe_test.%j.err

module load miniconda3/24.1.2-py310
module load cuda/12.4.1

source $(conda info --base)/etc/profile.d/conda.sh
conda activate /users/PAS3150/jacktaylor/miniconda3/envs/test_3.11

nvidia-smi

torchrun --nproc_per_node=1 --nnodes=1 train.py