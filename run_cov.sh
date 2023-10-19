#!/bin/bash
#SBATCH --account=desi
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=RascalC-BOSS-CMASS-N

# load cosmodesi environment
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main

# OpenMP settings
export OMP_PROC_BIND=spread
export OMP_PLACES=threads
export OMP_NUM_THREADS=256 # should match what is set in python script

# Hopefully let numpy use all threads
export NUMEXPR_MAX_THREADS=256
# Limit OpenBLAS thread usage (for jackknife assignment, error otherwise)
export OPENBLAS_NUM_THREADS=1

srun python -u run_cov.py