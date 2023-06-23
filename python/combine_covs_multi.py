"This reads two sets of RascalC results and two triplets of cosmodesi/pycorr .npy files to combine two covs following NS/GCcomb procedure for 2 tracers. Covariance of N and S 2PCF is neglected."

from pycorr import TwoPointCorrelationFunction
import numpy as np
import sys

## PARAMETERS
if len(sys.argv) not in (13, 15):
    print("Usage: python combine_covs_multi.py {RASCALC_RESULTS1} {RASCALC_RESULTS2} {PYCORR_FILE1_11} {PYCORR_FILE2_11} {PYCORR_FILE1_12} {PYCORR_FILE2_12} {PYCORR_FILE1_22} {PYCORR_FILE2_22} {N_R_BINS} {N_MU_BINS} {R_BINS_SKIP} {OUTPUT_COV_FILE} [{OUTPUT_COV_FILE1} {OUTPUT_COV_FILE2}].")
    sys.exit(1)
rascalc_results1 = str(sys.argv[1])
rascalc_results2 = str(sys.argv[2])
pycorr_files1 = [str(sys.argv[i]) for i in (3, 5, 7)]
pycorr_files2 = [str(sys.argv[i]) for i in (4, 6, 8)]
n_r_bins = int(sys.argv[9])
n_mu_bins = int(sys.argv[10])
r_bins_skip = int(sys.argv[11])
output_cov_file = str(sys.argv[12])
if len(sys.argv) >= 15:
    output_cov_file1 = str(sys.argv[13])
    output_cov_file2 = str(sys.argv[14])

# Read RascalC results
with np.load(rascalc_results1) as f:
    cov1 = f['full_theory_covariance']
    print(f"Max abs eigenvalue of bias correction matrix in 1st results is {np.max(np.abs(np.linalg.eigvals(f['full_theory_D_matrix']))):.2e}")
with np.load(rascalc_results2) as f:
    cov2 = f['full_theory_covariance']
    print(f"Max abs eigenvalue of bias correction matrix in 2nd results is {np.max(np.abs(np.linalg.eigvals(f['full_theory_D_matrix']))):.2e}")
# Save to their files if any
if len(sys.argv) >= 15:
    np.savetxt(output_cov_file1, cov1)
    np.savetxt(output_cov_file2, cov2)

# Read pycorr files to figure out weights
weight1 = np.zeros(0)
for pycorr_file1 in pycorr_files1:
    result = TwoPointCorrelationFunction.load(pycorr_file1)
    result = result[::result.shape[0]//n_r_bins, ::result.shape[1]//2//n_mu_bins].wrap().normalize()
    result = result[r_bins_skip:]
    weight1 = np.append(weight1, result.R1R2.wcounts.ravel())
weight2 = np.zeros(0)
for pycorr_file2 in pycorr_files2:
    result = TwoPointCorrelationFunction.load(pycorr_file2)
    result = result[::result.shape[0]//n_r_bins, ::result.shape[1]//2//n_mu_bins].wrap().normalize()
    result = result[r_bins_skip:]
    weight2 = np.append(weight2, result.R1R2.wcounts.ravel())

# Produce and save combined cov
# following xi = (xi1 * weight1 + xi2 * weight2) / (weight1 + weight2)
cov = (cov1 * weight1[None, :] * weight1[:, None] + cov2 * weight2[None, :] * weight2[:, None]) / (weight1 + weight2)[None, :] / (weight1 + weight2)[:, None]
np.savetxt(output_cov_file, cov)
