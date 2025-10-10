#!/bin/bash
#SBATCH -A PAS2119
#SBATCH --partition=batch
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --job-name=single-gpu-test
#SBATCH --output=std_out_err/single_gpu_test.%j.out
#SBATCH --error=std_out_err/single_gpu_test.%j.err

module load miniconda3/24.1.2-py310
module load cuda/12.4.1

source $(conda info --base)/etc/profile.d/conda.sh
conda activate /users/PAS3150/jacktaylor/miniconda3/envs/test_3.11

echo "Running on a single GPU on node: $(hostname)"
nvidia-smi

python train.py