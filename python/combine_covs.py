"This reads two sets of RascalC results and two cosmodesi/pycorr .npy files to combine two covs following NScomb procedure"

from pycorr import TwoPointCorrelationFunction
import numpy as np
import sys

## PARAMETERS
if len(sys.argv) not in (6, 8):
    print("Usage: python combine_covs.py {RASCALC_RESULTS1} {PYCORR_FILE1} {RASCALC_RESULTS2} {PYCORR_FILE2} {OUTPUT_COV_FILE} [{OUTPUT_COV_FILE1} {OUTPUT_COV_FILE2}].")
    sys.exit()
rascalc_results1 = str(sys.argv[1])
pycorr_file1 = str(sys.argv[2])
rascalc_results2 = str(sys.argv[3])
pycorr_file2 = str(sys.argv[4])
output_cov_file = str(sys.argv[5])
if len(sys.argv) >= 8:
    output_cov_file1 = str(sys.argv[6])
    output_cov_file2 = str(sys.argv[7])

# Read RascalC results
with np.load(rascalc_results1) as f:
    cov1 = f['full_theory_covariance']
    print(f"Max eigenvalue of bias correction matrix in 1st results is {np.max(np.linalg.eigvals(f['full_theory_D_matrix'])):.2e}")
with np.load(rascalc_results2) as f:
    cov2 = f['full_theory_covariance']
    print(f"Max eigenvalue of bias correction matrix in 2nd results is {np.max(np.linalg.eigvals(f['full_theory_D_matrix'])):.2e}")
# Save to their files if any
if len(sys.argv) >= 8:
    np.savetxt(output_cov_file1, cov1)
    np.savetxt(output_cov_file2, cov2)

# Read pycorr files to figure out weights
result = TwoPointCorrelationFunction.load(pycorr_file1).normalize()
weight1 = result.R1R2.wcounts
result = TwoPointCorrelationFunction.load(pycorr_file2).normalize()
weight2 = result.R1R2.wcounts

# Produce and save combined cov
# following xi = (xi1 * weight1 + xi2 * weight2) / (weight1 + weight2)
cov = (cov1 * weight1**2 + cov2 * weight2**2) / (weight1 + weight2)**2
np.savetxt(output_cov_file, cov)