"This reads cosmodesi/pycorr .npy file(s) and generates sample covariance of xi(s,mu) in text format"

from pycorr import TwoPointCorrelationFunction
import numpy as np
import sys

## PARAMETERS
if len(sys.argv) < 8:
    print("Usage: python compute_cov_from_pycorr.py {INPUT_NPY_FILE1} {INPUT_NPY_FILE2} [{INPUT_NPY_FILE3} ...] {OUTPUT_COV_FILE} {R_STEP} {N_MU} {R_MAX_BIN} {N_CORRELATIONS}.")
    sys.exit()
infile_names = sys.argv[1:-5]
outfile_name = str(sys.argv[-5])
r_step = int(sys.argv[-4])
n_mu = int(sys.argv[-3])
r_max_bin = int(sys.argv[-2])
n_corr = int(sys.argv[-1])

n_files = len(infile_names)
assert len(infile_names) > 1, "Can not compute covariance with 1 sample or less"
assert n_corr > 0, "Need at least one correlation"
assert n_files % n_corr == 0, "Need the same number of files for each correlation"
n_samples = n_files // n_corr

def fold_counts(counts): # utility function for correct folding, used in several places
    return counts[:, n_mu:] + counts[:, n_mu-1::-1] # first term is positive mu bins, second is negative mu bins in reversed order

def fold_xi(xi, RR): # proper folding of correlation function around mu=0: average weighted by RR counts
    return fold_counts(xi*RR) / fold_counts(RR)

# load first input file
result_orig = TwoPointCorrelationFunction.load(infile_names[0])
print("Read 2PCF shaped", result_orig.shape)
n_mu_orig = result_orig.shape[1]
assert n_mu_orig % (2 * n_mu) == 0, "Angular rebinning not possible"
mu_factor = n_mu_orig // 2 // n_mu
result = result_orig[:r_max_bin:r_step, ::mu_factor] # rebin
n_bins = np.prod(result.corr.shape) // 2 # number of (s, mu) bins, factor of 2 from folding about mu=0

# start xi and weights arrays
xi, weights = np.zeros((2, len(infile_names), n_bins * n_corr))
weights[0, :n_bins] = fold_counts(result.R1R2.wcounts).ravel()
xi[0, :n_bins] = fold_xi(result.corr, result.R1R2.wcounts).ravel()

# load remaining input files if any
for i in range(1, n_files):
    infile_name = infile_names[i]
    result_tmp = TwoPointCorrelationFunction.load(infile_name)
    assert result_tmp.shape == result_orig.shape, "Different shape in file %s" % infile_name
    result = result_tmp[:r_max_bin:r_step, ::mu_factor] # rebin and accumulate
    i_sample = i % n_samples # sample index
    i_corr = i // n_samples # correlation function index
    weights[i_sample, i_corr*n_bins:(i_corr+1)*n_bins] = fold_counts(result.R1R2.wcounts).ravel()
    xi[i_sample, i_corr*n_bins:(i_corr+1)*n_bins] = fold_xi(result.corr, result.R1R2.wcounts).ravel()

# compute cov
def sample_cov(x, weights):
    weights /= np.sum(weights, axis=0)
    mean_x = np.average(x, weights=weights, axis=0)
    tmp = weights * (x - mean_x)
    weight_prod = np.matmul(weights.T, weights)
    return np.matmul(tmp.T, tmp) / (np.ones_like(weight_prod) - weight_prod)

np.savetxt(outfile_name, sample_cov(xi, weights))