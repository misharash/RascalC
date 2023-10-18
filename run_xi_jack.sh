#!/bin/bash
#SBATCH --account=desi_g
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --job-name=BOSS-CMASS-N-xi-jack

# load cosmodesi environment
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main

srun python -u run_xi_jack.py
