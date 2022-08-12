"This reads two sets of RascalC results and two cosmodesi/pycorr .npy files to combine two covs following NScomb procedure"

from pycorr import TwoPointCorrelationFunction
import numpy as np
import sys

## PARAMETERS
if len(sys.argv) not in (9, 11):
    print("Usage: python combine_covs.py {RASCALC_RESULTS1} {RASCALC_RESULTS2} {PYCORR_FILE1} {PYCORR_FILE2} {N_R_BINS} {N_MU_BINS} {R_BINS_SKIP} {OUTPUT_COV_FILE} [{OUTPUT_COV_FILE1} {OUTPUT_COV_FILE2}].")
    sys.exit()
rascalc_results1 = str(sys.argv[1])
rascalc_results2 = str(sys.argv[2])
pycorr_file1 = str(sys.argv[3])
pycorr_file2 = str(sys.argv[4])
n_r_bins = int(sys.argv[5])
n_mu_bins = int(sys.argv[6])
r_bins_skip = int(sys.argv[7])
output_cov_file = str(sys.argv[8])
if len(sys.argv) >= 11:
    output_cov_file1 = str(sys.argv[9])
    output_cov_file2 = str(sys.argv[10])

# Read RascalC results
with np.load(rascalc_results1) as f:
    cov1 = f['full_theory_covariance']
    print(f"Max eigenvalue of bias correction matrix in 1st results is {np.max(np.linalg.eigvals(f['full_theory_D_matrix'])):.2e}")
with np.load(rascalc_results2) as f:
    cov2 = f['full_theory_covariance']
    print(f"Max eigenvalue of bias correction matrix in 2nd results is {np.max(np.linalg.eigvals(f['full_theory_D_matrix'])):.2e}")
# Save to their files if any
if len(sys.argv) >= 11:
    np.savetxt(output_cov_file1, cov1)
    np.savetxt(output_cov_file2, cov2)

# Read pycorr files to figure out weights
result = TwoPointCorrelationFunction.load(pycorr_file1)
result = result[::result.shape[0]//n_r_bins, ::result.shape[1]//n_mu_bins].normalize()
result = result[r_bins_skip:]
weight1 = (result.R1R2.wcounts[n_mu_bins:] + result.R1R2.wcounts[n_mu_bins-1::-1]).ravel()
result = TwoPointCorrelationFunction.load(pycorr_file2)
result = result[::result.shape[0]//n_r_bins, ::result.shape[1]//n_mu_bins].normalize()
result = result[r_bins_skip:]
weight2 = (result.R1R2.wcounts[n_mu_bins:] + result.R1R2.wcounts[n_mu_bins-1::-1]).ravel()

# Produce and save combined cov
# following xi = (xi1 * weight1 + xi2 * weight2) / (weight1 + weight2)
cov = (cov1 * weight1[None, :] * weight1[:, None] + cov2 * weight2[None, :] * weight2[:, None]) / (weight1 + weight2)[None, :] / (weight1 + weight2)[:, None]
np.savetxt(output_cov_file, cov)