"This reads a cosmodesi/pycorr .npy file and generates jackknife xi, weight, paircounts, and total paircounts text files for RascalC to use"

from pycorr import TwoPointCorrelationFunction
import numpy as np
import sys

## PARAMETERS
if len(sys.argv) not in (8, 9, 10):
    print("Usage: python convert_xi_jack_from_pycorr.py {INPUT_NPY_FILE} {OUTPUT_XI_JACK_FILE} {JACKKNIFE_WEIGHTS_FILE} {JACKKNIFE_PAIRCOUNTS_FILE} {BINNED_PAIRCOUNTS_FILE} {R_STEP} {N_MU} [{COUNTS_FACTOR} [{SPLIT_ABOVE}]].")
    sys.exit()
infile_name = str(sys.argv[1])
xi_name = str(sys.argv[2])
jackweights_name = str(sys.argv[3])
jackpairs_name = str(sys.argv[4])
binpairs_name = str(sys.argv[5])
r_step = int(sys.argv[6])
n_mu = int(sys.argv[7])
counts_factor = 1
if len(sys.argv) >= 9: counts_factor = float(sys.argv[8])
split_above = 0
if len(sys.argv) == 10: counts_factor = float(sys.argv[9])

result_orig = TwoPointCorrelationFunction.load(infile_name)
n_mu_orig = result_orig.shape[1]
assert n_mu_orig % (2 * n_mu) == 0, "Angular rebinning not possible"
mu_factor = n_mu_orig // 2 // n_mu

result = result_orig[::r_step, ::mu_factor] # rebin
binpairs = (result.R1R2.wcounts[:, n_mu:] + result.R1R2.wcounts[:, n_mu-1::-1]).ravel() / counts_factor # total counts, just wrap around mu=0 and make 1D
nonsplit_mask = (result.sepavg(axis=0) < split_above)
if split_above > 0: binpairs[nonsplit_mask] /= counts_factor # divide once more below the splitting scale

def jack_realization_rascalc(jack_estimator, i):
    # returns RascalC-framed jackknife realization, different from implemented in pycorr
    cls = jack_estimator.__class__.__bases__[0]
    kw = {}
    for name in jack_estimator.count_names:
        counts = getattr(jack_estimator, name)
        kw[name] = counts.auto[i] + counts.auto[i] + counts.cross12[i] + counts.cross21[i] # j1 x all2 + all1 x j2 = 2 x j1 x j2 + j1 x (all2 - j2) + j2 x (all1 - j1)
        kw[name] = kw[name].normalize(kw[name].wnorm / 2.) # now divide by 2 to match conventions from jackknife_weights{,_cross}.py
    return cls(**kw)

results = [jack_realization_rascalc(result, i) for i in result.realizations]

def fold_xi(xi, RR): # proper folding of correlation function around mu=0: average weighted by RR counts
    xi_RR = xi*RR
    return (xi_RR[:, n_mu:] + xi_RR[:, n_mu-1::-1]) / (RR[:, n_mu:] + RR[:, n_mu-1::-1])

jack_xi = np.array([fold_xi(jack.corr, jack.R1R2.wcounts).ravel() for jack in results]) # wrap around mu=0
jack_pairs = np.array([(jack.R1R2.wcounts[:, n_mu:] + jack.R1R2.wcounts[:, n_mu-1::-1]).ravel() for jack in results]) / counts_factor # wrap around mu=0
if split_above > 0: jack_pairs[:, nonsplit_mask] /= counts_factor # divide once more below the splitting scale
jack_pairs_sum = np.sum(jack_pairs, axis=0)
assert np.allclose(jack_pairs_sum, binpairs), "Total counts mismatch"
jack_weights = jack_pairs / binpairs[None, :]
full_xi = fold_xi(result.corr, result.R1R2.wcounts).ravel()
jack_xi_avg = np.average(jack_xi, weights=jack_weights, axis=0)
assert np.allclose(full_xi, jack_xi_avg), "Total xi mismatch"

## Custom array to string function
def my_a2s(a, fmt='%.18e'):
    return ' '.join([fmt % e for e in a])

## Write to file using numpy funs
np.savetxt(binpairs_name, binpairs.reshape(-1, 1)) # this file must have 1 column
header = my_a2s(result.sepavg(axis=0))+'\n'+my_a2s(result.sepavg(axis=1)[n_mu:])
np.savetxt(xi_name, jack_xi, header=header, comments='')
jack_numbers = np.array(result.realizations).reshape(-1, 1) # column of jackknife numbers, may be useless but needed for format compatibility
np.savetxt(jackweights_name, np.hstack((jack_numbers, jack_weights)))
np.savetxt(jackpairs_name, np.hstack((jack_numbers, jack_pairs)))