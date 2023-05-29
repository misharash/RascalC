"This reads cosmodesi/pycorr .npy file(s) and generates sample covariance of xi_l(s) in text format"

from pycorr import TwoPointCorrelationFunction
import numpy as np
import sys

## PARAMETERS
if len(sys.argv) < 8:
    print("Usage: python sample_cov_multipoles_from_pycorr.py {INPUT_NPY_FILE1} {INPUT_NPY_FILE2} [{INPUT_NPY_FILE3} ...] {OUTPUT_COV_FILE} {R_STEP} {MAX_L} {R_MAX} {N_CORRELATIONS}.")
    sys.exit(1)
infile_names = sys.argv[1:-5]
outfile_name = str(sys.argv[-5])
r_step = int(sys.argv[-4])
max_l = int(sys.argv[-3])
r_max = int(sys.argv[-2])
n_corr = int(sys.argv[-1])
assert r_max % r_step == 0, "Radial rebinning impossible after max radial bin cut"
assert max_l >= 0, "Maximal multipole is negative"
assert max_l % 2 == 0, "Only even multipoles supported"
ells = np.arange(0, max_l+1, 2)
n_l = len(ells)

n_files = len(infile_names)
assert len(infile_names) > 1, "Can not compute covariance with 1 sample or less"
assert n_corr > 0, "Need at least one correlation"
assert n_files % n_corr == 0, "Need the same number of files for each correlation"
n_samples = n_files // n_corr

# load first input file
result_orig = TwoPointCorrelationFunction.load(infile_names[0])
print("Read 2PCF shaped", result_orig.shape)

# determine the radius step in pycorr
r_steps_orig = np.diff(result_orig.edges[0])
r_step_orig = int(np.around(np.mean(r_steps_orig)))
assert np.allclose(r_steps_orig, r_step_orig, rtol=5e-3, atol=5e-3), "Binnings other than linear with integer step are not supported"
assert r_step % r_step_orig == 0, "Radial rebinning not possible"
r_step //= r_step_orig
assert r_max % r_step_orig == 0, "Max radial bin cut incompatible with original radial binning"
r_max //= r_step_orig

result = result_orig[:r_max:r_step] # rebin radially
n_r_bins = result.corr.shape[0]
n_bins = n_r_bins * n_l

# start xi array, ditch the weights
xi = np.zeros((len(infile_names), n_corr, n_bins))
xi[0, 0] = result.get_corr(mode='poles', ells=ells)

# load remaining input files
for i in range(1, n_files):
    infile_name = infile_names[i]
    result_tmp = TwoPointCorrelationFunction.load(infile_name)
    assert result_tmp.shape == result_orig.shape, "Different shape in file %s" % infile_name
    result = result_tmp[:r_max:r_step] # rebin radially
    i_sample = i % n_samples # sample index
    i_corr = i // n_samples # correlation function index
    xi[i_sample, i_corr] = result.get_corr(mode='poles', ells=ells)

xi = xi.reshape(len(infile_names), n_corr * n_bins) # convert array to 2D

cov = np.cov(xi.T) # xi has to be transposed, because variables (bins) are in columns (2nd index) of it and cov expects otherwise. Weights are collapsed across the bins; the proper expression for covariance with weights changing for different variables within one sample has not been found yet; the jackknife expression is short by ~n_samples.
np.savetxt(outfile_name, cov)