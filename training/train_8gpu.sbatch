#!/bin/bash
#SBATCH -A PAS2119
#SBATCH --partition=batch
#SBATCH --time=02:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1   
#SBATCH --cpus-per-task=8
#SBATCH --job-name=safe-test-8gpu
#SBATCH --output=std_out_err/safe_test_8gpu.%j-task%t.out
#SBATCH --error=std_out_err/safe_test_8gpu.%j-task%t.err

module load miniconda3/24.1.2-py310
module load cuda/12.4.1

source $(conda info --base)/etc/profile.d/conda.sh
conda activate /users/PAS3150/jacktaylor/miniconda3/envs/test_3.11

#set environment variables for torchrun to discover nodes
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
#ask OS for free port
export MASTER_PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')

echo "Job ID: $SLURM_JOB_ID"
echo "Master Node: $MASTER_ADDR"
echo "Using Free Port: $MASTER_PORT"

nvidia-smi

torchrun \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=4 \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    train.py