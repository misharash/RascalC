## Script to post-process the single-field integrals computed by the C++ code.
## We output the theoretical covariance matrices, (quadratic-bias corrected) precision matrices and the effective number of samples, N_eff.

import numpy as np
import sys,os
from tqdm import trange

# PARAMETERS
if len(sys.argv) not in (6, 7, 8):
    print("Usage: python post_process_default.py {COVARIANCE_DIR} {N_R_BINS} {N_MU_BINS} {N_SUBSAMPLES} {OUTPUT_DIR} [{SHOT_NOISE_RESCALING} [{SKIP_R_BINS}]]")
    sys.exit(1)

file_root = str(sys.argv[1])
n = int(sys.argv[2])
m = int(sys.argv[3])
n_samples = int(sys.argv[4])
outdir = str(sys.argv[5])
alpha = float(sys.argv[6]) if len(sys.argv) >= 7 else 1
skip_bins = int(sys.argv[7]) * m if len(sys.argv) >= 8 else 0 # convert from radial to total number of bins right away

# Create output directory
if not os.path.exists(outdir):
    os.makedirs(outdir)

def load_matrices(index):
    """Load intermediate or full covariance matrices"""
    cov_root = os.path.join(file_root, 'CovMatricesAll/')
    c2 = np.diag(np.loadtxt(cov_root+'c2_n%d_m%d_11_%s.txt'%(n,m,index))[skip_bins:])
    c3 = np.loadtxt(cov_root+'c3_n%d_m%d_1,11_%s.txt'%(n,m,index))[skip_bins:, skip_bins:]
    c4 = np.loadtxt(cov_root+'c4_n%d_m%d_11,11_%s.txt'%(n,m,index))[skip_bins:, skip_bins:]

    # Now symmetrize and return matrices
    return c2,0.5*(c3+c3.T),0.5*(c4+c4.T)

# Load in full theoretical matrices
print("Loading best estimate of covariance matrix")
c2,c3,c4=load_matrices('full')

# Compute full covariance matrices and precision
full_cov = c4+c3*alpha+c2*alpha**2.
n_bins = len(c4)

# Check positive definiteness
assert np.all(np.linalg.eigvalsh(full_cov) > 0), "The full covariance is not positive definite - insufficient convergence"

# Compute full precision matrix
print("Computing the full precision matrix estimate:")
# Load in partial theoretical matrices
c2s,c3s,c4s=[],[],[]
for i in trange(n_samples, desc="Loading full subsamples"):
    c2, c3, c4 = load_matrices(i)
    c2s.append(c2)
    c3s.append(c3)
    c4s.append(c4)
c2s, c3s, c4s = [np.array(a) for a in (c2s, c3s, c4s)]
partial_cov = alpha**2 * c2s + alpha * c3s + c4s
sum_partial_cov = np.sum(partial_cov, axis=0)
tmp=0.
for i in range(n_samples):
    c_excl_i = (sum_partial_cov - partial_cov[i]) / (n_samples - 1)
    tmp += np.matmul(np.linalg.inv(c_excl_i), partial_cov[i])
full_D_est=(n_samples-1.)/n_samples * (-1.*np.eye(n_bins) + tmp/n_samples)
full_prec = np.matmul(np.eye(n_bins)-full_D_est,np.linalg.inv(full_cov))
print("Full precision matrix estimate computed")

# Now compute effective N:
slogdetD=np.linalg.slogdet(full_D_est)
D_value = slogdetD[0]*np.exp(slogdetD[1]/n_bins)
if slogdetD[0]<0:
    print("N_eff is negative! Setting to zero")
    N_eff_D = 0.
else:
    N_eff_D = (n_bins+1.)/D_value+1.
    print("Total N_eff Estimate: %.4e"%N_eff_D)

output_name = os.path.join(outdir, 'Rescaled_Covariance_Matrices_Default_n%d_m%d.npz'%(n,m))
np.savez(output_name,full_theory_covariance=full_cov,
         shot_noise_rescaling=alpha,full_theory_precision=full_prec,
         N_eff=N_eff_D,full_theory_D_matrix=full_D_est,
         individual_theory_covariances=partial_cov)

print("Saved output covariance matrices as %s"%output_name)
