#!/bin/bash
#SBATCH -A PAS2836
#SBATCH --partition=nextgen
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=lm_train
#SBATCH --output=myjob.%j.out
#SBATCH --error=myjob.%j.err

module load miniconda3/24.1.2-py310
module load cuda/12.4.1

source $(conda info --base)/etc/profile.d/conda.sh
conda activate /users/PAS3150/jacktaylor/miniconda3/envs/test_3.11

nvidia-smi

python decoder_only_transformer.py