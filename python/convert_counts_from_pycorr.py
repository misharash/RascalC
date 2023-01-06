"This reads a cosmodesi/pycorr .npy file and generates binned pair counts text file for RascalC to use"

from pycorr import TwoPointCorrelationFunction
import numpy as np
import sys

## PARAMETERS
if len(sys.argv) not in (5, 6, 7, 8):
    print("Usage: python convert_counts_from_pycorr.py {INPUT_NPY_FILE} {OUTPUT_PAIRCOUNTS_TEXT_FILE} {R_STEP} {N_MU} [{COUNTS_FACTOR} [{SPLIT_ABOVE}] [{R_MAX}]]].")
    sys.exit(1)
infile_name = str(sys.argv[1])
outfile_name = str(sys.argv[2])
r_step = int(sys.argv[3])
n_mu = int(sys.argv[4])
counts_factor = float(sys.argv[5]) if len(sys.argv) >= 6 else 1 # basically number of randoms used for these counts, used to convert from total to 1 catalog count estimate
split_above = float(sys.argv[6]) if len(sys.argv) >= 7 else 0 # divide weighted RR counts by counts_factor**2 below this and by counts_factor above
r_max = int(sys.argv[7]) if len(sys.argv) >= 8 else None # if given, limit used r_bins at that
if r_max: assert r_max % r_step == 0, "Radial rebinning impossible after max radial bin cut"

result_orig = TwoPointCorrelationFunction.load(infile_name)
n_mu_orig = result_orig.shape[1]
assert n_mu_orig % (2 * n_mu) == 0, "Angular rebinning not possible"
mu_factor = n_mu_orig // 2 // n_mu

# determine the radius step in pycorr
r_steps_orig = np.diff(result_orig.edges[0])
r_step_orig = int(np.around(np.mean(r_steps_orig)))
assert np.allclose(r_steps_orig, r_step_orig, rtol=5e-3, atol=5e-3), "Binnings other than linear with integer step are not supported"
assert r_step % r_step_orig == 0, "Radial rebinning not possible"
r_step //= r_step_orig

if r_max:
    assert r_max % r_step_orig == 0, "Max radial bin cut incompatible with original radial binning"
    r_max //= r_step_orig
    result_orig = result_orig[:r_max] # cut to max bin

result = result_orig[::r_step, ::mu_factor].wrap() # rebin and wrap to positive mu
paircounts = result.R1R2.wcounts / counts_factor
nonsplit_mask = (result.sepavg(axis=0) < split_above)
if split_above > 0: paircounts[nonsplit_mask] /= counts_factor # divide once more below the splitting scale

## Write to file using numpy funs
np.savetxt(outfile_name, paircounts.reshape(-1, 1)) # the file always has 1 column