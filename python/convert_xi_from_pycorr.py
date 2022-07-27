"This reads cosmodesi/pycorr .npy file(s) and generates input xi text file for RascalC to use"

from pycorr import TwoPointCorrelationFunction
import numpy as np
import os

## PARAMETERS
if len(sys.argv) < 5:
    print("Usage: python convert_xi_from_pycorr.py {INPUT_NPY_FILE1} [{INPUT_NPY_FILE2} ...] {OUTPUT_XI_DAT_FILE} {R_STEP} {N_MU}.")
    sys.exit()
infile_names = sys.argv[1:-3]
outfile_name = str(sys.argv[-3])
r_step = int(sys.argv[-2])
n_mu = int(sys.argv[-1])

# load first input file
result_orig = TwoPointCorrelationFunction.load(infile_names[0])
n_mu_orig = result_orig.shape[1]
assert n_mu_orig % (2 * n_mu) == 0, "Angular rebinning not possible"
mu_factor = n_mu_orig // 2 // n_mu
result = result_orig[::r_step, ::mu_factor] # rebin
# retrieve data sizes
data_size1_sum = result_orig.D1D2.size1
data_size2_sum = result_orig.D1D2.size2

# load remaining input files if any
for infile_name in infile_names[1:]:
    result_tmp = TwoPointCorrelationFunction.load(infile_name)
    assert result_tmp.shape == result_orig.shape, "Different shape in file %s" % infile_name
    result += result_tmp[::r_step, ::mu_factor] # rebin and accumulate
    # accumulate data sizes
    data_size1_sum += result_tmp.D1D2.size1
    data_size2_sum += result_tmp.D1D2.size2

print(f"Mean size of data 1 is {data_size1_sum/len(infile_names):.6f}")
print(f"Mean size of data 2 is {data_size2_sum/len(infile_names):.6f}")

def fold_xi(xi, RR): # proper folding of correlation function around mu=0: average weighted by RR counts
    xi_RR = xi*RR
    return (xi_RR[:, n_mu:] + xi_RR[:, n_mu-1::-1]) / (RR[:, n_mu:] + RR[:, n_mu-1::-1])

xi = fold_xi(result.corr, result.R1R2.wcounts) # wrap around zero

## Custom array to string function
def my_a2s(a, fmt='%.18e'):
    return ' '.join([fmt % e for e in a])

## Write to file using numpy funs
header = my_a2s(result.sepavg(axis=0))+'\n'+my_a2s(result.sepavg(axis=1))
np.savetxt(outfile_name, xi, header=header, comments='')