"This reads a cosmodesi/pycorr .npy file and generates input xi text file for RascalC to use"

from pycorr import TwoPointCorrelationFunction
import numpy as np
import os

## PARAMETERS
if len(sys.argv) != 5:
    print("Usage: python convert_xi_from_pycorr.py {INPUT_NPY_FILE} {OUTPUT_DAT_FILE} {R_STEP} {N_MU}.")
    sys.exit()
infile_name = str(sys.argv[1])
outfile_name = str(sys.argv[2])
r_step = int(sys.argv[3])
n_mu = int(sys.argv[4])

result_orig = TwoPointCorrelationFunction.load(infile_name)
n_mu_orig = result_orig.shape[1]
assert n_mu_orig % (2 * n_mu) == 0, "Angular rebinning not possible"
mu_factor = n_mu_orig // 2 // n_mu

result = result_orig[::r_step, ::mu_factor] # rebin
xi = (result.corr[:, n_mu:] + result.corr[:, n_mu-1::-1])/2 # wrap around zero

## Custom array to string function
def my_a2s(a, fmt='%.18e'):
    return ' '.join([fmt % e for e in a])

## Write to file using numpy funs
header = my_a2s(result.sepavg(axis=0))+'\n'+my_a2s(result.sepavg(axis=1))
np.savetxt(outfile_name, xi, header=header, comments='')