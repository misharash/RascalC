"This reads two sets of RascalC results and two triplets of cosmodesi/pycorr .npy files to combine two full 2-tracer covs following NS/GCcomb procedure for 2 tracers in Legendre mode and convert them . Covariance of N(GC) and S(GC) 2PCF is neglected."

from pycorr import TwoPointCorrelationFunction
from scipy.special import legendre
import numpy as np
import sys

## PARAMETERS
if len(sys.argv) not in (13, 15):
    print("Usage: python combine_covs_multi_to_cat.py {RASCALC_RESULTS1} {RASCALC_RESULTS2} {PYCORR_FILE1_11} {PYCORR_FILE2_11} {PYCORR_FILE1_12} {PYCORR_FILE2_12} {PYCORR_FILE1_22} {PYCORR_FILE2_22} {N_R_BINS} {MAX_L} {R_BINS_SKIP} {OUTPUT_COV_FILE} [{BIAS1} {BIAS2} [{OUTPUT_COV_FILE1} {OUTPUT_COV_FILE2}]].")
    sys.exit(1)
rascalc_results1 = str(sys.argv[1])
rascalc_results2 = str(sys.argv[2])
pycorr_files1 = [str(sys.argv[i]) for i in (3, 5, 7)]
pycorr_files2 = [str(sys.argv[i]) for i in (4, 6, 8)]
n_r_bins = int(sys.argv[9])
max_l = int(sys.argv[10])
assert max_l % 2 == 0, "Odd multipoles not supported"
n_l = max_l // 2 + 1
r_bins_skip = int(sys.argv[11])
output_cov_file = str(sys.argv[12])
bias1 = str(sys.argv[13]) if len(sys.argv) >= 14 else 1
bias2 = str(sys.argv[14]) if len(sys.argv) >= 15 else 1
if len(sys.argv) >= 17:
    output_cov_file1 = str(sys.argv[15])
    output_cov_file2 = str(sys.argv[16])

# Read RascalC results
with np.load(rascalc_results1) as f:
    cov1 = f['full_theory_covariance']
    n_bins = len(cov1)
    assert n_bins % (3*n_l) == 0, "Number of bins mismatch"
    n = n_bins // (3*n_l)
    cov1 = cov1.reshape(3, n, n_l, 3, n, n_l) # convert to 6D from 2D with [t, r, l] ordering for both rows and columns
    cov1 = cov1.transpose(0, 2, 1, 3, 5, 4) # change ordering to [t, l, r] for both rows and columns
    cov1 = cov1.reshape(n_bins, n_bins) # convert back from 6D to 2D
    print(f"Max abs eigenvalue of bias correction matrix in 1st results is {np.max(np.abs(np.linalg.eigvals(f['full_theory_D_matrix']))):.2e}")
with np.load(rascalc_results2) as f:
    cov2 = f['full_theory_covariance']
    assert n_bins == len(cov2), "Number of bins mismatch"
    cov2 = cov2.reshape(3, n, n_l, 3, n, n_l) # convert to 6D from 2D with [t, r, l] ordering for both rows and columns
    cov2 = cov2.transpose(0, 2, 1, 3, 5, 4) # change ordering to [t, l, r] for both rows and columns
    cov2 = cov2.reshape(n_bins, n_bins) # convert back from 6D to 2D
    print(f"Max abs eigenvalue of bias correction matrix in 2nd results is {np.max(np.abs(np.linalg.eigvals(f['full_theory_D_matrix']))):.2e}")

# Read pycorr files to figure out weights
weights1, results1 = [], []
for pycorr_file1 in pycorr_files1:
    result = TwoPointCorrelationFunction.load(pycorr_file1)
    result = result[::result.shape[0]//n_r_bins].wrap()
    result = result[r_bins_skip:]
    results1.append(result)
    weights1.append(result.R1R2.wcounts)
weights1 = np.array(weights1)
assert weights1.shape[:-1] == (3, n), "Wrong shape of weights 1"

n_mu_bins = result.shape[1]
mu_edges = result.edges[1]

weights2, results2 = [], []
for pycorr_file2 in pycorr_files2:
    result = TwoPointCorrelationFunction.load(pycorr_file2)
    result = result[::result.shape[0]//n_r_bins].wrap()
    result = result[r_bins_skip:]
    results2.append(result)
    weights2.append(result.R1R2.wcounts)
weights2 = np.array(weights2)
assert weights2.shape == (3, n, n_mu_bins), "Wrong shape of weights 2"

# Add weighting by bias for each tracer
bias_weights = np.array((bias1**2, 2*bias1*bias2, bias2**2)) # auto1, cross12, auto2 are multiplied by product of biases of tracers involved in each. Moreover, cross12 enters twice because wrapped cross21 is the same.
weights1 *= bias_weights[:, None, None]
weights2 *= bias_weights[:, None, None]

# Function for multiplying all the counts by a factor
def multiply_counts_pycorr(pycorr_result, factor):
    new = pycorr_result.copy()
    if isinstance(wnorm, str):
        wnorm = getattr(pycorr_result, wnorm).wnorm
    for name in new.count_names:
        counts = getattr(new, name)
        setattr(new, name, counts.normalize(wnorm = counts.wnorm * factor))
    return new

# Now multiply ALL the counts
results1 = [multiply_counts_pycorr(result, factor) for result, factor in zip(results1, bias_weights)]
results2 = [multiply_counts_pycorr(result, factor) for result, factor in zip(results2, bias_weights)]
# Sum and normalize the counts
result1 = sum(results1).normalize()
result2 = sum(results2).normalize()
# Weights will be counts
weight1 = result1.R1R2.wcounts
weight2 = result2.R1R2.wcounts

# Normalize weights
sum_weight = weight1 + weight2
weight1 /= sum_weight
weight2 /= sum_weight
# Normalize the full weights across correlation function labels
weights1 /= np.sum(weights1, axis=0)[None, :, :]
weights2 /= np.sum(weights2, axis=0)[None, :, :]

ells = np.arange(0, max_l+1, 2)
# Legendre multipoles integrated over mu bins, do not depend on radial binning and tracers
leg_mu_ints = np.zeros((n_l, n_mu_bins))

for i, ell in enumerate(ells):
    leg_pol = legendre(ell) # Legendre polynomial
    leg_pol_int = np.polyint(leg_pol) # its indefinite integral (analytic)
    leg_mu_ints[i] = np.diff(leg_pol_int(mu_edges)) # differences of indefinite integral between edges of mu bins = integral of Legendre polynomial over each mu bin

leg_mu_avg = leg_mu_ints / np.diff(mu_edges) # average value of Legendre polynomial in each bin

# First, convert multi-tracer cov to single-tracer in each region

# Derivatives of angularly binned 2PCF wrt Legendre are leg_mu_avg[ell//2, mu_bin]
# Angularly binned 2PCF are added with weights (normalized) weights1/2[tracer, r_bin, mu_bin]
# Derivatives of Legendre wrt binned 2PCF are leg_mu_ints[ell//2, mu_bin] * (2*ell+1)
# So we need to sum such product over mu bins, while tracers and radial bins stay independent, and the partial derivative of combined 2PCF wrt the 2PCFs 1/2 will be
pd1 = np.einsum('il,tkl,jl,km->tikjm', leg_mu_avg, weights1, (2*ells[:, None]+1) * leg_mu_ints, np.eye(n)).reshape(n_bins, n_l*n)
pd2 = np.einsum('il,tkl,jl,km->tikjm', leg_mu_avg, weights2, (2*ells[:, None]+1) * leg_mu_ints, np.eye(n)).reshape(n_bins, n_l*n)
# We have correct [t_in, l_in, r_in, l_out, r_out] ordering and want to make these matrices in the end thus the reshape.
# The resulting covs are single-tracer (for the combined catalogs) so there is no t_out.

# Produce single-tracer covs for each region
cov1 = pd1.T.dot(cov1).dot(pd1)
cov2 = pd2.T.dot(cov2).dot(pd2)
# Save to their files if any
if len(sys.argv) >= 17:
    np.savetxt(output_cov_file1, cov1)
    np.savetxt(output_cov_file2, cov2)

n_bins = n_l*n # all covariances are single tracer now

# Now, combine single-tracer covs

# Derivatives of angularly binned 2PCF wrt Legendre are leg_mu_avg[ell//2, mu_bin]
# Angularly binned 2PCF are added with weights (normalized) weight1/2[r_bin, mu_bin]
# Derivatives of Legendre wrt binned 2PCF are leg_mu_ints[ell//2, mu_bin] * (2*ell+1)
# So we need to sum such product over mu bins, while radial bins stay independent, and the partial derivative of combined 2PCF wrt the 2PCFs 1/2 will be
pd1 = np.einsum('il,kl,jl,km->ikjm', leg_mu_avg, weight1, (2*ells[:, None]+1) * leg_mu_ints, np.eye(n)).reshape(n_bins, n_bins)
pd2 = np.einsum('il,kl,jl,km->ikjm', leg_mu_avg, weight2, (2*ells[:, None]+1) * leg_mu_ints, np.eye(n)).reshape(n_bins, n_bins)
# We have correct [l_in, r_in, l_out, r_out] ordering and want to make these matrices in the end thus the reshape

# Produce and save combined cov
cov = pd1.T.dot(cov1).dot(pd1) + pd2.T.dot(cov2).dot(pd2)
np.savetxt(output_cov_file, cov)
