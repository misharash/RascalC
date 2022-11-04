"This reads cosmodesi/pycorr .npy file(s) and generates input xi text file for RascalC to use"

from pycorr import TwoPointCorrelationFunction
import numpy as np
import sys

## PARAMETERS
if len(sys.argv) < 5:
    print("Usage: python convert_xi_from_pycorr.py {INPUT_NPY_FILE1} [{INPUT_NPY_FILE2} ...] {OUTPUT_XI_DAT_FILE} {R_STEP} {N_MU}.")
    sys.exit()
infile_names = sys.argv[1:-3]
outfile_name = str(sys.argv[-3])
r_step = int(sys.argv[-2])
n_mu = int(sys.argv[-1])

nfiles = len(infile_names)

# load first input file
result_orig = TwoPointCorrelationFunction.load(infile_names[0])
print("Read 2PCF shaped", result_orig.shape)
n_mu_orig = result_orig.shape[1]
assert n_mu_orig % (2 * n_mu) == 0, "Angular rebinning not possible"
mu_factor = n_mu_orig // 2 // n_mu
result = result_orig[::r_step, ::mu_factor] # rebin
# retrieve data sizes
data_size1_sum = result_orig.D1D2.size1
data_size2_sum = result_orig.D1D2.size2

def fold_counts(counts): # utility function for correct folding, used in several places
    return counts[:, n_mu:] + counts[:, n_mu-1::-1] # first term is positive mu bins, second is negative mu bins in reversed order

def fold_xi(xi, RR): # proper folding of correlation function around mu=0: average weighted by RR counts
    return fold_counts(xi*RR) / fold_counts(RR)

# start arrays of individual xi and their weights for cov computation, pointless if less than 2 files
if nfiles > 1:
    all_xi, all_weights = np.zeros((2, nfiles, np.prod(result.shape) // 2)) # the last is number of bins, division by 2 coming from folding about mu=0
    all_xi[0] = fold_xi(result.corr, result.R1R2.wcounts).ravel()
    all_weights[0] = fold_counts(result.R1R2.wcounts).ravel()

# load remaining input files if any
for i in range(1, nfiles): # number 0 done above
    infile_name = infile_names[i]
    result_tmp = TwoPointCorrelationFunction.load(infile_name)
    assert result_tmp.shape == result_orig.shape, "Different shape in file %s" % infile_name
    result += result_tmp[::r_step, ::mu_factor] # rebin and accumulate
    # accumulate data sizes
    data_size1_sum += result_tmp.D1D2.size1
    data_size2_sum += result_tmp.D1D2.size2
    # save individual xi and its weight
    all_xi[i] = fold_xi(result_tmp.corr, result_tmp.R1R2.wcounts).ravel()
    all_weights[i] = fold_counts(result_tmp.R1R2.wcounts).ravel()

print(f"Mean size of data 1 is {data_size1_sum/len(infile_names):.6e}")
print(f"Mean size of data 2 is {data_size2_sum/len(infile_names):.6e}")
np.savetxt(outfile_name + ".ndata", np.array((data_size1_sum, data_size2_sum)) / len(infile_names)) # save them for later

xi = fold_xi(result.corr, result.R1R2.wcounts) # wrap around zero

## Custom array to string function
def my_a2s(a, fmt='%.18e'):
    return ' '.join([fmt % e for e in a])

## Write to file using numpy funs
header = my_a2s(result.sepavg(axis=0))+'\n'+my_a2s(result.sepavg(axis=1)[n_mu:])
np.savetxt(outfile_name, xi, header=header, comments='')

# covariance computation, not possible if less than 2 files
if nfiles > 1:
    all_weights /= np.sum(all_weights, axis=0)[None, :] # normalize
    mean_xi = np.average(all_xi, weights=all_weights, axis=0) # recompute the mean just in case, might have subtle differences from xi above
    delta_xi = all_weights * (all_xi - mean_xi[None, :])
    weight_prod = np.matmul(all_weights.T, all_weights)
    cov = np.matmul(delta_xi.T, delta_xi) / (np.ones_like(weight_prod) - weight_prod)
    np.savetxt(outfile_name + ".cov", cov)