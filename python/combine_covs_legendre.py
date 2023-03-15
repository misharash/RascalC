"This reads two sets of RascalC results and two cosmodesi/pycorr .npy files to combine two covs following NScomb procedure in Legendre mode. Covariance of N and S 2PCF is neglected."

from pycorr import TwoPointCorrelationFunction
from scipy.special import legendre
import numpy as np
import sys

## PARAMETERS
if len(sys.argv) not in (9, 11):
    print("Usage: python combine_covs_legendre.py {RASCALC_RESULTS1} {RASCALC_RESULTS2} {PYCORR_FILE1} {PYCORR_FILE2} {N_R_BINS} {MAX_L} {R_BINS_SKIP} {OUTPUT_COV_FILE} [{OUTPUT_COV_FILE1} {OUTPUT_COV_FILE2}].")
    sys.exit(1)
rascalc_results1 = str(sys.argv[1])
rascalc_results2 = str(sys.argv[2])
pycorr_file1 = str(sys.argv[3])
pycorr_file2 = str(sys.argv[4])
n_r_bins = int(sys.argv[5])
max_l = int(sys.argv[6])
assert max_l % 2 == 0, "Odd multipoles not supported"
n_l = max_l // 2 + 1
r_bins_skip = int(sys.argv[7])
output_cov_file = str(sys.argv[8])
if len(sys.argv) >= 11:
    output_cov_file1 = str(sys.argv[9])
    output_cov_file2 = str(sys.argv[10])

# Read RascalC results
with np.load(rascalc_results1) as f:
    cov1 = f['full_theory_covariance']
    n_bins = len(cov1)
    assert n_bins % n_l == 0, "Number of bins mismatch"
    n = n_bins // n_l
    cov1 = cov1.reshape(n, n_l, n, n_l) # convert to 4D from 2D with [r, l] ordering for both rows and columns
    cov1 = cov1.transpose(1, 0, 3, 2) # change orderng to [l, r] for both rows and columns
    cov1 = cov1.reshape(n_bins, n_bins) # convert back from 4D to 2D
    print(f"Max abs eigenvalue of bias correction matrix in 1st results is {np.max(np.abs(np.linalg.eigvals(f['full_theory_D_matrix']))):.2e}")
with np.load(rascalc_results2) as f:
    cov2 = f['full_theory_covariance']
    assert n_bins == len(cov2), "Number of bins mismatch"
    cov2 = cov2.reshape(n, n_l, n, n_l) # convert to 4D from 2D with [r, l] ordering for both rows and columns
    cov2 = cov2.transpose(1, 0, 3, 2) # change orderng to [l, r] for both rows and columns
    cov2 = cov2.reshape(n_bins, n_bins) # convert back from 4D to 2D
    print(f"Max abs eigenvalue of bias correction matrix in 2nd results is {np.max(np.abs(np.linalg.eigvals(f['full_theory_D_matrix']))):.2e}")
# Save to their files if any
if len(sys.argv) >= 11:
    np.savetxt(output_cov_file1, cov1)
    np.savetxt(output_cov_file2, cov2)

# Read pycorr files to figure out weights of s, mu binned 2PCF
result = TwoPointCorrelationFunction.load(pycorr_file1)
result = result[::result.shape[0]//n_r_bins].wrap().normalize()
result = result[r_bins_skip:]
weight1 = result.R1R2.wcounts

n_mu_bins = result.shape[1]
mu_edges = result.edges[1]

result = TwoPointCorrelationFunction.load(pycorr_file2)
result = result[::result.shape[0]//n_r_bins].wrap().normalize()
result = result[r_bins_skip:]
weight2 = result.R1R2.wcounts

# Normalize weights
sum_weight = weight1 + weight2
weight1 /= sum_weight
weight2 /= sum_weight

ells = np.arange(0, max_l+1, 2)
# Legendre multipoles integrated over mu bins, do not depend on radial binning
leg_mu_ints = np.zeros((n_l, n_mu_bins))

for i, ell in enumerate(ells):
    leg_pol = legendre(ell) # Legendre polynomial
    leg_pol_int = np.polyint(leg_pol) # its indefinite integral (analytic)
    leg_mu_ints[i] = np.diff(leg_pol_int(mu_edges)) # differences of indefinite integral between edges of mu bins = integral of Legendre polynomial over each mu bin

leg_mu_avg = leg_mu_ints / np.diff(mu_edges) # average value of Legendre polynomial in each bin

# Derivatives of angularly binned 2PCF wrt Legendre are leg_mu_avg[ell//2, mu_bin]
# Angularly binned 2PCF are added with weights (normalized) weight1/2[r_bin, mu_bin]
# Derivatives of Legendre wrt binned 2PCF are leg_mu_ints[ell//2, mu_bin] * (2*ell+1)
# So we need to sum such product over mu bins, while radial bins stay independent, and the partial derivative of combined 2PCF wrt the 2PCFs 1/2 will be
pd1 = np.einsum('il,kl,jl->ikjk', leg_mu_avg, weight1, (2*ells[:, None]+1) * leg_mu_ints).reshape(n_bins, n_bins)
pd2 = np.einsum('il,kl,jl->ikjk', leg_mu_avg, weight2, (2*ells[:, None]+1) * leg_mu_ints).reshape(n_bins, n_bins)
# We have correct [l_in, r_in, l_out, r_out] ordering and want to make these matrices in the end thus the reshape

# Produce and save combined cov
cov = pd1.T.dot(cov1).dot(pd1) + pd2.T.dot(cov2).dot(pd2)
np.savetxt(output_cov_file, cov)
