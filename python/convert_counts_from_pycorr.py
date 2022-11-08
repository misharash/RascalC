"This reads a cosmodesi/pycorr .npy file and generates binned pair counts text file for RascalC to use"

from pycorr import TwoPointCorrelationFunction
import numpy as np
import sys

## PARAMETERS
if len(sys.argv) not in (5, 6, 7, 8):
    print("Usage: python convert_counts_from_pycorr.py {INPUT_NPY_FILE} {OUTPUT_PAIRCOUNTS_TEXT_FILE} {R_STEP} {N_MU} [{COUNTS_FACTOR} [{SPLIT_ABOVE}] [{R_MAX_BIN}]]].")
    sys.exit()
infile_name = str(sys.argv[1])
outfile_name = str(sys.argv[2])
r_step = int(sys.argv[3])
n_mu = int(sys.argv[4])
counts_factor = float(sys.argv[5]) if len(sys.argv) >= 6 else 1 # basically number of randoms used for these counts, used to convert from total to 1 catalog count estimate
split_above = float(sys.argv[6]) if len(sys.argv) >= 7 else 0 # divide weighted RR counts by counts_factor**2 below this and by counts_factor above
r_max_bin = int(sys.argv[7]) if len(sys.argv) >= 8 else None # if given, limit used r_bins at that

result_orig = TwoPointCorrelationFunction.load(infile_name)
n_mu_orig = result_orig.shape[1]
assert n_mu_orig % (2 * n_mu) == 0, "Angular rebinning not possible"
mu_factor = n_mu_orig // 2 // n_mu

if r_max_bin: result_orig = result_orig[:r_max_bin] # cut to max bin

def fold_counts(counts): # utility function for correct folding, used in several places
    return counts[:, n_mu:] + counts[:, n_mu-1::-1] # first term is positive mu bins, second is negative mu bins in reversed order

result = result_orig[::r_step, ::mu_factor] # rebin
paircounts = fold_counts(result.R1R2.wcounts) / counts_factor # wrap around zero
nonsplit_mask = (result.sepavg(axis=0) < split_above)
if split_above > 0: paircounts[nonsplit_mask] /= counts_factor # divide once more below the splitting scale

## Write to file using numpy funs
np.savetxt(outfile_name, paircounts.reshape(-1, 1)) # the file always has 1 column