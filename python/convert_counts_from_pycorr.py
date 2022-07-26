"This reads a cosmodesi/pycorr .npy file and generates binned pair counts text file for RascalC to use"

from pycorr import TwoPointCorrelationFunction
import numpy as np
import os

## PARAMETERS
if len(sys.argv) not in (5, 6):
    print("Usage: python convert_counts_from_pycorr.py {INPUT_NPY_FILE} {OUTPUT_PAIRCOUNTS_TEXT_FILE} {R_STEP} {N_MU} [{COUNTS_FACTOR}].")
    sys.exit()
infile_name = str(sys.argv[1])
outfile_name = str(sys.argv[2])
r_step = int(sys.argv[3])
n_mu = int(sys.argv[4])
counts_factor = 1
if len(sys.argv) == 6: counts_factor = float(sys.argv[8])

result_orig = TwoPointCorrelationFunction.load(infile_name)
n_mu_orig = result_orig.shape[1]
assert n_mu_orig % (2 * n_mu) == 0, "Angular rebinning not possible"
mu_factor = n_mu_orig // 2 // n_mu

result = result_orig[::r_step, ::mu_factor] # rebin
paircounts = (result.R1R2.wcounts[:, n_mu:] + result.R1R2.wcounts[:, n_mu-1::-1]) / counts_factor # wrap around zero

## Write to file using numpy funs
np.savetxt(outfile_name, paircounts.reshape(-1, 1)) # the file always has 1 column