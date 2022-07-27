"This reads cosmodesi/pycorr .npy files and generates sample covariance and mean xi text file (not necessary for RascalC)"

from pycorr import TwoPointCorrelationFunction
import numpy as np
import sys

## PARAMETERS
if len(sys.argv) < 7:
    print("Usage: python sample_cov_from_pycorr.py {INPUT_NPY_FILE1} {INPUT_NPY_FILE2} [{INPUT_NPY_FILE3} ...] {OUTPUT_COV_FILE} {OUTPUT_XI_FILE} {R_STEP} {R_SKIP} {N_MU}.")
    sys.exit()
infile_names = sys.argv[1:-4]
covfile_name = str(sys.argv[-4])
xifile_name = str(sys.argv[-3])
r_step = int(sys.argv[-2])
n_mu = int(sys.argv[-1])

def formatted_xi(result): # proper folding and raveling of correlation function around mu=0: average weighted by RR counts
    xi, RR = result.corr, result.R1R2.wcounts
    xi_RR = xi*RR
    folded_xi = (xi_RR[:, n_mu:] + xi_RR[:, n_mu-1::-1]) / (RR[:, n_mu:] + RR[:, n_mu-1::-1])
    return folded_xi.ravel()

# load first input file
result_orig = TwoPointCorrelationFunction.load(infile_names[0])
n_mu_orig = result_orig.shape[1]
assert n_mu_orig % (2 * n_mu) == 0, "Angular rebinning not possible"
mu_factor = n_mu_orig // 2 // n_mu
result_orig_rebinned = result_orig[::r_step, ::mu_factor] # rebin
all_xis = [formatted_xi(result_orig_rebinned)] # start list of xi's

# load remaining input files
for infile_name in infile_names[1:]:
    result_tmp = TwoPointCorrelationFunction.load(infile_name)
    assert result_tmp.shape == result_orig.shape, "Different shape in file %s" % infile_name
    all_xis.append(formatted_xi(result_tmp[::r_step, ::mu_factor])) # rebin and save to list

all_xis = np.array(all_xis)
xi_avg = np.mean(all_xis, axis=0)
tmp = all_xis - xi_avg[None, :]
cov_xi = np.matmul(tmp.T, tmp) / (len(all_xis) - 1)

## Custom array to string function
def my_a2s(a, fmt='%.18e'):
    return ' '.join([fmt % e for e in a])

## Write to file using numpy funs
header = my_a2s(result_orig_rebinned.sepavg(axis=0))+'\n'+my_a2s(result_orig_rebinned.sepavg(axis=1)[n_mu:])
np.savetxt(xifile_name, xi_avg.reshape(-1, n_mu), header=header, comments='')
np.savetxt(covfile_name, cov_xi)