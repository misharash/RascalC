#!/bin/bash
#SBATCH --account=desi
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=RascalC-Y1-blinded-LRGxELG-recon-batch
#SBATCH --array=0-1

# load cosmodesi environment
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
# load GSL for the C++ code
module load gsl

# OpenMP settings
export OMP_PROC_BIND=spread
export OMP_PLACES=threads
export OMP_NUM_THREADS=256 # should match what is set in python script

# Hopefully let numpy use all threads
export NUMEXPR_MAX_THREADS=256

srun python -u run_cov.py $SLURM_ARRAY_TASK_ID