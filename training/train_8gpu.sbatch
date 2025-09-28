#!/bin/bash
#SBATCH -A PAS2119
#SBATCH --partition=batch
#SBATCH --time=02:00:00
#SBATCH --nodes=2
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --job-name=safe-test-8gpu
#SBATCH --output=std_out_err/safe_test_8gpu.%j.out
#SBATCH --error=std_out_err/safe_test_8gpu.%j.err

module load miniconda3/24.1.2-py310
module load cuda/12.4.1

source $(conda info --base)/etc/profile.d/conda.sh
conda activate /users/PAS3150/jacktaylor/miniconda3/envs/test_3.11

nvidia-smi

# Multi-node training with 8 GPUs (4 per node, 2 nodes)
# Get the first node in the job as master
MASTER_NODE=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_ADDR=$(scontrol show hostname $MASTER_NODE | head -n 1)

srun torchrun --nproc_per_node=4 --nnodes=2 --node_rank=$SLURM_PROCID --master_addr=$MASTER_ADDR --master_port=12345 train.py
