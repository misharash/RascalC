"""
This module contains convenience functions that

- read RascalC results (checking eigenvalues of the inversion bias matrix. If they are much smaller than 1, it is safe to simply invert the covariance matrix. Otherwise, a correction factor is necessary. Caveat: may need to cut the inversion bias matrix for 3PCF covariances, which is not yet implemented)
- convert (reorder) 3PCF covariance matrices (as is often needed for Legendre moments)
- and/or export (save) the full 3PCF covariance matrix to a text file.
"""

import numpy as np
from typing import Callable
from .get_shot_noise_rescaling import get_shot_noise_rescaling # for convenience
from .cov_utils import get_cov_header, load_cov


def convert_cov_3pcf_legendre(cov: np.ndarray[float], max_l: int, exclude_samebins: bool = True, apply_scaling: bool = True) -> np.ndarray[float]:
    """
    Change the bin ordering of the 3PCF covariance matrix in Legendre mode for a single tracer.
    The original bin ordering (in ``RascalC`` ``.npy`` files) is by radial bin pairs (top-level, going over all possible first and second bins, resulting in repetitions) and then by multipoles.
    The resulting bin ordering (for text files) is by multipoles (top-level) and then by radial bin pairs (unique only).
    By default, the same-bin pairs are excluded to match the ENCORE output convention, this can be disabled with ``exclude_samebins=False``.
    By default, the ell-dependent scaling factor is applied to convert to ENCORE basis functions from RascalC Legendre polynomial basis, this can be disabled with ``apply_scaling=False``.
    """
    n_l = max_l + 1
    n_bins = len(cov)
    if n_bins % n_l != 0: raise ValueError("Number of bins in the Legendre covariance must be divisible by the number of multipoles (even and odd)")
    n_r_bins2 = n_bins // n_l
    n_r_bins = int(np.rint(np.sqrt(n_r_bins2)))
    if n_r_bins2 != n_r_bins ** 2: raise ValueError("Number of radial bin pairs does not appear to be a square")
    n_bin_pairs = n_r_bins * (n_r_bins + 1) // 2 - exclude_samebins * n_r_bins # number of unique bin pairs for the output covariance
    n_bins_new = n_bin_pairs * n_l # number of bins for the output covariance
    cov = cov.reshape(n_r_bins, n_r_bins, n_l, n_r_bins, n_r_bins, n_l) # convert to 6D from 2D with [r1, r2, l] ordering for both rows and columns
    # prepare to exclude repeated bin pairs, and same-bin pairs if requested
    bin_indices = np.arange(n_r_bins)
    bin_index1 = np.repeat(bin_indices, n_r_bins-exclude_samebins-bin_indices)
    bin_index2 = np.concatenate([bin_indices[i+exclude_samebins:] for i in range(n_r_bins)])
    # with exclude_samebins=True (=1), bin_index1 and bin_index2 cover all the bin pairs under the condition bin_index1 < bin_index2, the order follows the ENCORE format
    # with exclude_samebins=False (=0), bin_index1 and bin_index2 cover all the bin pairs under the condition bin_index1 <= bin_index2
    cov = cov[bin_index1, bin_index2][:, :, bin_index1, bin_index2] # cut to the bin pairs specified above, also making the matrix 4D with [r_pair, l] indices for both rows and columns
    if apply_scaling: # prepare to apply the ell-dependent factor between RascalC and ENCORE basis functions
        ells = np.arange(n_l)
        ell_factor = ((-1)**ells * np.sqrt(2 * ells + 1) / (4 * np.pi)) # the ell-dependent factor between the ENCORE 3-point basis functions and Legendre polynomials given by Equation (16) in https://arxiv.org/pdf/2105.08722
        cov /= ell_factor[:, None, None] * ell_factor[None, None, :] # apply the factor, NumPy broadcasting matches trailing dimensions. need to check if it is not multiplication; there might also be a factor of 2 or something similar
    cov = cov.transpose(1, 0, 3, 2) # change ordering to [l, r_pair] for both rows and columns
    cov = cov.reshape(n_bins_new, n_bins_new) # convert back from 4D to 2D
    return cov


def load_cov_3pcf_legendre(rascalc_results_file: str, max_l: int, exclude_samebins: bool = True, apply_scaling: bool = True, print_function: Callable[[str], None] = print) -> np.ndarray[float]:
    "Load the theoretical covariance matrix from RascalC results file, change the bin ordering and apply scaling as in :func:`convert_cov_3pcf_legendre`; intended for 3PCF Legendre single-tracer mode."
    return convert_cov_3pcf_legendre(load_cov(rascalc_results_file, print_function), max_l, exclude_samebins=exclude_samebins, apply_scaling=apply_scaling)


def export_cov_3pcf_legendre(rascalc_results_file: str, max_l: int, output_cov_file: str, exclude_samebins: bool = True, apply_scaling: bool = True, print_function: Callable[[str], None] = print) -> None:
    "Export the theoretical covariance matrix from RascalC results file to a text file with conversion appropriate for 3PCF single-tracer Legendre mode (see :func:`convert_cov_3pcf_legendre`)."
    np.savetxt(output_cov_file, load_cov_3pcf_legendre(rascalc_results_file, max_l, exclude_samebins=exclude_samebins, apply_scaling=apply_scaling, print_function = print_function), header = get_cov_header(rascalc_results_file))