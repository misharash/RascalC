#!/bin/bash
#SBATCH --account=desi
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=RascalC-Y1-v0.6.1-double-blinded-test
#SBATCH --array=4 # only LRG 0.8-1.1 SGC for testing, not to waste allocated time

# load cosmodesi environment
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
# GSL needed by the C++ code should already be loaded in cosmodesi
# module load gsl

# OpenMP settings
export OMP_PROC_BIND=spread
export OMP_PLACES=threads
export OMP_NUM_THREADS=256 # should match what is set in python script

# Hopefully let numpy use all threads
export NUMEXPR_MAX_THREADS=32
# Limit OpenBLAS thread usage (for jackknife assignment, error otherwise)
export OPENBLAS_NUM_THREADS=1

python -u run_cov.py $SLURM_ARRAY_TASK_ID