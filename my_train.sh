#!/bin/bash
#PBS -N pytorch_dp_job

#PBS -l select=1:ncpus=112:ngpus=1:mem=320gb:container_engine=enroot
#PBS -l walltime=24:00:00
#PBS -q normal
#PBS -P 12002486
#PBS -j oe

#!/bin/sh
export LOGLEVEL=INFO
export HYDRA_FULL_ERROR=1
export TF_CPP_MIN_LOG_LEVEL=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source "/home/users/ntu/lyuchen/miniconda3/bin/activate"
conda activate catk
cd /home/users/ntu/lyuchen/scratch/keguo_projects/ntu/sim

MASTER_PORT=29501 torchrun -m src.run > gail_nohist.log  2>&1


