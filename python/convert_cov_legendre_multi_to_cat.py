"This reads a .npy file of RascalC Legendre results (or txt covariance converted previously) and a triplet of cosmodesi/pycorr .npy files to produce a covariance for a catalog of these two tracers concatenated."

from pycorr import TwoPointCorrelationFunction
from scipy.special import legendre
import numpy as np
import sys

## PARAMETERS
if len(sys.argv) not in (9, 11):
    print("Usage: python combine_cov_multi_to_cat.py {RASCALC_RESULTS} {PYCORR_FILE_11} {PYCORR_FILE_12} {PYCORR_FILE_22} {N_R_BINS} {MAX_L} {R_BINS_SKIP} {OUTPUT_COV_FILE} [{BIAS1} {BIAS2}].")
    sys.exit(1)
rascalc_results = str(sys.argv[1])
pycorr_files = [str(sys.argv[i]) for i in range(2, 5)]
n_r_bins = int(sys.argv[5])
max_l = int(sys.argv[6])
assert max_l % 2 == 0, "Odd multipoles not supported"
n_l = max_l // 2 + 1
r_bins_skip = int(sys.argv[7])
output_cov_file = str(sys.argv[8])
bias1 = str(sys.argv[9]) if len(sys.argv) >= 10 else 1
bias2 = str(sys.argv[10]) if len(sys.argv) >= 11 else 1

# Read RascalC results
if any(rascalc_results.endswith(ext) for ext in (".npy", ".npz")):
    # read numpy file
    with np.load(rascalc_results) as f:
        cov_in = f['full_theory_covariance']
        n_bins = len(cov_in)
        assert n_bins % (3*n_l) == 0, "Number of bins mismatch"
        n = n_bins // (3*n_l)
        cov_in = cov_in.reshape(3, n, n_l, 3, n, n_l) # convert to 6D from 2D with [t, r, l] ordering for both rows and columns
        cov_in = cov_in.transpose(0, 2, 1, 3, 5, 4) # change ordering to [t, l, r] for both rows and columns
        cov_in = cov_in.reshape(n_bins, n_bins) # convert back from 6D to 2D
        print(f"Max abs eigenvalue of bias correction matrix in 1st results is {np.max(np.abs(np.linalg.eigvals(f['full_theory_D_matrix']))):.2e}")
else:
    # read text file
    cov_in = np.loadtxt(rascalc_results)
    n_bins = len(cov_in)
    assert n_bins % (3*n_l) == 0, "Number of bins mismatch"
    n = n_bins // (3*n_l)
    # assume it has been transposed

# Read pycorr files to figure out weights
weights = []
for pycorr_file in pycorr_files:
    result = TwoPointCorrelationFunction.load(pycorr_file)
    result = result[::result.shape[0]//n_r_bins].wrap().normalize()
    result = result[r_bins_skip:]
    weights.append(result.R1R2.wcounts)
weights = np.array(weights)
assert weights.shape[:-1] == (3, n), "Wrong shape of weights"

n_mu_bins = result.shape[1]
mu_edges = result.edges[1]

# Add weighting by bias for each tracer
bias_weights = np.array((bias1**2, 2*bias1*bias2, bias2**2)) # auto1, cross12, auto2 are multiplied by product of biases of tracers involved in each. Moreover, cross12 enters twice because wrapped cross21 is the same.
weights *= bias_weights[:, None]

# Normalize weights across the correlation type axis
weights /= np.sum(weights, axis=0)[None, :]

ells = np.arange(0, max_l+1, 2)
# Legendre multipoles integrated over mu bins, do not depend on radial binning
leg_mu_ints = np.zeros((n_l, n_mu_bins))

for i, ell in enumerate(ells):
    leg_pol = legendre(ell) # Legendre polynomial
    leg_pol_int = np.polyint(leg_pol) # its indefinite integral (analytic)
    leg_mu_ints[i] = np.diff(leg_pol_int(mu_edges)) # differences of indefinite integral between edges of mu bins = integral of Legendre polynomial over each mu bin

leg_mu_avg = leg_mu_ints / np.diff(mu_edges) # average value of Legendre polynomial in each bin

# Derivatives of angularly binned 2PCF wrt Legendre are leg_mu_avg[ell//2, mu_bin]
# Angularly binned 2PCF are added with weights (normalized) weights[tracer, r_bin, mu_bin]
# Derivatives of Legendre wrt binned 2PCF are leg_mu_ints[ell//2, mu_bin] * (2*ell+1)
# So we need to sum such product over mu bins, while tracers and radial bins stay independent, and the partial derivative of combined 2PCF wrt the 2PCFs 1/2 will be
pd = np.einsum('il,tkl,jl,km->tikjm', leg_mu_avg, weights, (2*ells[:, None]+1) * leg_mu_ints, np.eye(n)).reshape(n_bins, n_l*n)
# We have correct [t_in, l_in, r_in, l_out, r_out] ordering and want to make these matrices in the end thus the reshape.
# The output cov is single-tracer (for the combined catalog) so there is no t_out.

# Produce and save combined cov
cov_out = pd.T.dot(cov_in).dot(pd)
np.savetxt(output_cov_file, cov_out)
