#!/bin/bash
#SBATCH --account=desi
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH --time=0:03:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=RascalC-openmp-issue
#SBATCH --array=4 # only LRG 0.8-1.1 SGC for testing, not to waste allocated time

# load cosmodesi environment - already loaded for my user?
# source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
# GSL needed by the C++ code should already be loaded in cosmodesi
# module load gsl

# OpenMP settings
export OMP_PROC_BIND=spread
export OMP_PLACES=threads
export OMP_NUM_THREADS=256 # should match what is set in python script

# Hopefully let numpy use all threads
export NUMEXPR_MAX_THREADS=256

python -u run_cov.py $SLURM_ARRAY_TASK_ID