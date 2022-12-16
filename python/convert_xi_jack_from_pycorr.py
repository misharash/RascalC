"This reads a cosmodesi/pycorr .npy file and generates jackknife xi, weight, paircounts, and total paircounts text files for RascalC to use"

from pycorr import TwoPointCorrelationFunction
import numpy as np
import sys

## PARAMETERS
if len(sys.argv) not in (8, 9, 10, 11):
    print("Usage: python convert_xi_jack_from_pycorr.py {INPUT_NPY_FILE} {OUTPUT_XI_JACK_FILE} {JACKKNIFE_WEIGHTS_FILE} {JACKKNIFE_PAIRCOUNTS_FILE} {BINNED_PAIRCOUNTS_FILE} {R_STEP} {N_MU} [{COUNTS_FACTOR} [{SPLIT_ABOVE} [{R_MAX}]]].")
    sys.exit(1)
infile_name = str(sys.argv[1])
xi_name = str(sys.argv[2])
jackweights_name = str(sys.argv[3])
jackpairs_name = str(sys.argv[4])
binpairs_name = str(sys.argv[5])
r_step = int(sys.argv[6])
n_mu = int(sys.argv[7])
counts_factor = float(sys.argv[8]) if len(sys.argv) >= 9 else 1 # basically number of randoms used for these counts
split_above = float(sys.argv[9]) if len(sys.argv) >= 10 else 0 # divide weighted RR counts by counts_factor**2 below this and by counts_factor above
r_max = int(sys.argv[10]) if len(sys.argv) >= 11 else None # if given, limit used r_bins at that
if r_max: assert r_max % r_step == 0, "Radial rebinning impossible after max radial bin cut"

result_orig = TwoPointCorrelationFunction.load(infile_name)
n_mu_orig = result_orig.shape[1]
assert n_mu_orig % (2 * n_mu) == 0, "Angular rebinning not possible"
mu_factor = n_mu_orig // 2 // n_mu

# determine the radius step in pycorr
r_steps_orig = np.diff(result_orig.edges[0])
r_step_orig = int(np.around(np.mean(r_steps_orig)))
assert np.allclose(r_steps_orig, r_step_orig, rtol=5e-3, atol=5e-3), "Binnings other than linear with integer step are not supported"
assert r_step % r_step_orig == 0, "Radial rebinning not possible"
r_step //= r_step_orig

if r_max:
    assert r_max % r_step_orig == 0, "Max radial bin cut incompatible with original radial binning"
    r_max //= r_step_orig
    result_orig = result_orig[:r_max] # cut to max bin

result = result_orig[::r_step, ::mu_factor].wrap() # rebin and wrap to positive mu
binpairs = result.R1R2.wcounts.ravel() / counts_factor # total counts, made 1D
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

jack_xi = np.array([jack.corr.ravel() for jack in results]) # already wrapped
jack_pairs = np.array([jack.R1R2.wcounts.ravel() for jack in results]) / counts_factor # already wrapped
if split_above > 0: jack_pairs[:, nonsplit_mask] /= counts_factor # divide once more below the splitting scale
jack_pairs_sum = np.sum(jack_pairs, axis=0)
assert np.allclose(jack_pairs_sum, binpairs), "Total counts mismatch"
jack_weights = jack_pairs / binpairs[None, :]

## Custom array to string function
def my_a2s(a, fmt='%.18e'):
    return ' '.join([fmt % e for e in a])

## Write to file using numpy funs
np.savetxt(binpairs_name, binpairs.reshape(-1, 1)) # this file must have 1 column
header = my_a2s(result.sepavg(axis=0))+'\n'+my_a2s(result.sepavg(axis=1))
np.savetxt(xi_name, jack_xi, header=header, comments='')
jack_numbers = np.array(result.realizations).reshape(-1, 1) # column of jackknife numbers, may be useless but needed for format compatibility
np.savetxt(jackweights_name, np.hstack((jack_numbers, jack_weights)))
np.savetxt(jackpairs_name, np.hstack((jack_numbers, jack_pairs)))