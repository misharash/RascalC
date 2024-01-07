#!/bin/bash

# load cosmodesi environment
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
# GSL needed by the C++ code should already be loaded in cosmodesi
# module load gsl

# OpenMP settings
export OMP_PROC_BIND=spread
export OMP_PLACES=threads
export OMP_NUM_THREADS=256 # should match what is set in python script

# Hopefully let numpy use all threads
export NUMEXPR_MAX_THREADS=256
# Limit OpenBLAS thread usage (for jackknife assignment, error otherwise)
export OPENBLAS_NUM_THREADS=1

python run_cov.py 4