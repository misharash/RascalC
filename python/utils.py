# Contains some utility functions widely used in other scripts
# Not intended for execution from command line
import sys, os
import numpy as np
import pycorr
from warnings import warn
from astropy.io import fits

def get_arg_safe(index: int, type = str, default: object = None) -> object:
    # get argument by index from sys.argv and convert it to the requested type if there are enough elements there
    # otherwise return the default value
    return type(sys.argv[index]) if len(sys.argv) > index else default

def blank_function(*args, **kwargs) -> None:
    # function that accepts anything and does nothing
    # mostly intended for skipping optional printing
    pass

def my_a2s(a, fmt='%.18e'):
    # custom array to string function
    return ' '.join([fmt % e for e in a])

def my_str_to_bool(s: str) -> bool:
    # naive conversion to bool for all non-empty strings is True, and one can't give an empty string as a command line argument, so need to make it more explicit
    return s not in ("0", "false")

def symmetrized(A):
    # symmetrize a 2D matrix
    return 0.5 * (A + A.T)

def parse_FKP_arg(FKP_weights: str) -> bool | tuple[float, str]:
    if not my_str_to_bool(FKP_weights): return False
    # determine if it actually has P0,NZ_name format. Such strings should convert to True
    arg_FKP_split = FKP_weights.split(",")
    if len(arg_FKP_split) == 2:
        return (float(arg_FKP_split[0]), arg_FKP_split[1])
    if len(arg_FKP_split) == 1: return True
    raise ValueError("FKP parameter matched neither USE_FKP_WEIGHTS (true/false in any register or 0/1) nor P0,NZ_name (float and string without space).")

def read_particles_fits_file(input_file: str, FKP_weights: bool | (float, str) = False, mask: int = 0, use_weights: bool = True):
    # Read FITS file with particles. Can apply mask filtering and compute FKP weights in different ways. Works for DESI setups
    filt = True # default pre-filter is true
    with fits.open(input_file) as f:
        data = f[1].data
        all_ra = data["RA"]
        all_dec = data["DEC"]
        all_z = data["Z"]
        colnames = data.columns.names
        all_w = data["WEIGHT"] if "WEIGHT" in colnames and use_weights else np.ones_like(all_z)
        if FKP_weights:
            all_w *= 1/(1+FKP_weights[0]*data[FKP_weights[1]]) if FKP_weights != True else data["WEIGHT_FKP"]
        if "WEIGHT" not in colnames and not FKP_weights: warn("No weights found, assigned unit weight to each particle.")
        if mask: filt = (data["STATUS"] & mask == mask) # all 1-bits from mask have to be set in STATUS; skip if mask=0
    return np.array((all_ra, all_dec, all_z, all_w)).T[filt]

def read_xi_file(xi_file: str):
    # Interpret RascalC text format using numpy functions
    if not os.path.isfile(xi_file): raise FileNotFoundError('Could not find input file %s' % xi_file)
    r_vals = np.genfromtxt(xi_file, max_rows=1)
    mu_vals = np.genfromtxt(xi_file, max_rows=1, skip_header=1)
    xi_vals = np.genfromtxt(xi_file, skip_header=2)
    return r_vals, mu_vals, xi_vals

def write_xi_file(xi_file: str, r_vals: np.ndarray[float], mu_vals: np.ndarray[float], xi_vals: np.ndarray[float]):
    # Reproduce RascalC text format using numpy functions
    header = my_a2s(r_vals) + '\n' + my_a2s(mu_vals)
    np.savetxt(xi_file, xi_vals, header=header, comments='')

def write_binning_file(out_file: str, r_edges: np.ndarray[float], print_function = blank_function):
    # Save bin edges array into a Corrfunc (and RascalC) radial binning file format
    np.savetxt(out_file, np.array((r_edges[:-1], r_edges[1:])).T)
    print_function("Binning file '%s' written successfully." % out_file)

def fix_bad_bins_pycorr(xi_estimator: pycorr.twopoint_estimator.BaseTwoPointEstimator) -> pycorr.twopoint_estimator.BaseTwoPointEstimator:
    # fixes bins with negative wcounts by overwriting their content by reflection
    # only known cause for now is self-counts (DD, RR) in bin 0, n_mu_orig/2-1 – subtraction is sometimes not precise enough, especially with float32
    cls = xi_estimator.__class__
    kw = {}
    for name in xi_estimator.count_names:
        counts = getattr(xi_estimator, name)
        bad_bins_mask = counts.wcounts < 0
        for s_bin, mu_bin in zip(*np.nonzero(bad_bins_mask)):
            warn(f"Negative {name}.wcounts ({counts.wcounts[s_bin, mu_bin]:.2e}) found in bin {s_bin}, {mu_bin}; replacing them with reflected bin ({counts.wcounts[s_bin, -1-mu_bin]:.2e})")
            counts.wcounts[s_bin, mu_bin] = counts.wcounts[s_bin, -1-mu_bin]
        kw[name] = counts
    return cls(**kw)

def reshape_pycorr(xi_estimator: pycorr.TwoPointEstimator, n_mu: int | None = None, r_step: float = 1, r_max: float = np.inf, skip_r_bins: int = 0) -> pycorr.TwoPointEstimator:
    n_mu_orig = xi_estimator.shape[1]
    if n_mu_orig % 2 != 0: raise ValueError("Wrapping not possible")
    if n_mu:
        if n_mu_orig % (2 * n_mu) != 0: raise ValueError("Angular rebinning not possible")
        mu_factor = n_mu_orig // 2 // n_mu
    else: mu_factor = 1 # leave the original number of mu bins

    # determine the radius step in pycorr
    r_steps_orig = np.diff(xi_estimator.edges[0])
    r_step_orig = np.mean(r_steps_orig)
    if not np.allclose(r_steps_orig, r_step_orig, rtol=5e-3, atol=5e-3): raise ValueError("Binning appears not linear with integer step; such case is not supported")
    r_factor_exact = r_step_orig / r_step
    r_factor = int(np.rint(r_factor_exact))
    if not np.allclose(r_factor, r_factor_exact, rtol=5e-3): raise ValueError("Radial rebinning seems impossible")

    # Apply r_max cut
    r_values = xi_estimator.sepavg(axis = 0)
    xi_estimator = xi_estimator[r_values <= r_max]

    return fix_bad_bins_pycorr(xi_estimator[skip_r_bins * r_factor:])[::r_factor, ::mu_factor].wrap() # first skip bins, then fix bad bins, then rebin and wrap to positive mu