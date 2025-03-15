from pycorr import TwoPointCorrelationFunction
import numpy as np
from ..pycorr_utils.utils import reshape_pycorr
from ..cov_utils import get_cov_header, load_cov_legendre
from ..pycorr_utils.counts import get_counts_from_pycorr
from ..mu_bin_legendre_factors import compute_mu_bin_legendre_factors
from typing import Callable


def combine_covs_legendre(rascalc_results1: str, rascalc_results2: str, pycorr_file1: str, pycorr_file2: str, output_cov_file: str, max_l: int, r_step: float = 1, skip_r_bins: int | tuple[int, int] = 0, output_cov_file1: str | None = None, output_cov_file2: str | None = None, print_function: Callable[[str], None] = print) -> np.ndarray[float]:
    """
    Produce Legendre mode single-tracer covariance matrix for the region/footprint that is a combination of two regions/footprints neglecting the correlations between the clustering statistics in the different regions.
    For additional details, see Appendix B.2 of `Rashkovetskyi et al 2025 <https://arxiv.org/abs/2404.03007>`_.

    Parameters
    ----------
    rascalc_results1, rascalc_results2 : string
        Filenames for the RascalC (post-processing) results for the two regions in NumPy format.
    
    pycorr_file1, pycorr_file2 : string
        Filenames for the ``pycorr`` (https://github.com/cosmodesi/pycorr) ``.npy`` files with the correlation functions and pair counts for the two regions.
        The order of regions must be the same as in RascalC results.
    
    output_cov_file : string
        Filename for the output text file, in which the covariance matrix will be saved.

    max_l : integer
        The highest (even) multipole index, must match the RascalC results.

    r_step : float
        The width of the radial (separation) bins, must match the RascalC results.
    
    skip_r_bins : integer or tuple of two integers
        (Optional) removal of some radial bins from the loaded ``pycorr`` counts before adjusting the radial (separation) bin width to match the covariance settings.
        First (or the only) number sets the number of radial/separation bins to skip from the beginning.
        Second number (if provided) sets the number of radial/separation bins to skip from the end.
        By default, no bins are skipped.
        E.g. if the ``pycorr`` counts are in 1 Mpc/h bins from 0 to 200 Mpc/h and the RascalC covariances are computed only between 20 and 200 Mpc/h, ``skip_r_bins`` should be ``20``.
    
    output_cov_file1, output_cov_file2 : string or None
        (Optional) if provided, the text covariance matrices for the corresponding region will be saved in this file.
    
    print_function : Callable[[str], None]
        (Optional) custom function to use for printing. Needs to take string arguments and not return anything. Default is ``print``.

    Returns
    -------
    combined_cov : np.ndarray[float]
        The resulting covariance matrix for the combined region.
    """
    # Read RascalC results
    header1 = get_cov_header(rascalc_results1)
    cov1 = load_cov_legendre(rascalc_results1, max_l, print_function)
    n_bins = len(cov1)
    header2 = get_cov_header(rascalc_results2)
    cov2 = load_cov_legendre(rascalc_results2, max_l, print_function)
    # Save to their files if any
    if output_cov_file1: np.savetxt(output_cov_file1, cov1, header = header1)
    if output_cov_file2: np.savetxt(output_cov_file2, cov2, header = header2)
    header = f"combined from {rascalc_results1} with {header1} and {rascalc_results2} with {header2}" # form the final header to include both

    # Read pycorr files to figure out weights of s, mu binned 2PCF
    xi_estimator1 = reshape_pycorr(TwoPointCorrelationFunction.load(pycorr_file1), n_mu = None, r_step = r_step, skip_r_bins = skip_r_bins).normalize()
    n_r_bins = xi_estimator1.shape[0]
    mu_edges = xi_estimator1.edges[1]
    weight1 = get_counts_from_pycorr(xi_estimator1, counts_factor = 1)
    weight2 = get_counts_from_pycorr(reshape_pycorr(TwoPointCorrelationFunction.load(pycorr_file2).normalize(), n_mu = None, r_step = r_step, skip_r_bins = skip_r_bins), counts_factor = 1)

    # Normalize weights
    sum_weight = weight1 + weight2
    weight1 /= sum_weight
    weight2 /= sum_weight

    mu_leg_factors, leg_mu_factors = compute_mu_bin_legendre_factors(mu_edges, max_l, do_inverse = True)

    # Derivatives of angularly binned 2PCF wrt Legendre are leg_mu_factors[ell//2, mu_bin]
    # Angularly binned 2PCF are added with weights (normalized) weight1/2[r_bin, mu_bin]
    # Derivatives of Legendre wrt binned 2PCF are mu_leg_factors[mu_bin, ell//2]
    # So we need to sum such product over mu bins, while radial bins stay independent, and the partial derivative of combined 2PCF wrt the 2PCFs 1/2 will be
    pd1 = np.einsum('il,kl,lj,km->ikjm', leg_mu_factors, weight1, mu_leg_factors, np.eye(n_r_bins)).reshape(n_bins, n_bins)
    pd2 = np.einsum('il,kl,lj,km->ikjm', leg_mu_factors, weight2, mu_leg_factors, np.eye(n_r_bins)).reshape(n_bins, n_bins)
    # We have correct [l_in, r_in, l_out, r_out] ordering and want to make these matrices in the end thus the reshape

    # Produce and save combined cov
    cov = pd1.T.dot(cov1).dot(pd1) + pd2.T.dot(cov2).dot(pd2)
    np.savetxt(output_cov_file, cov, header=header) # includes source parts and their shot-noise rescaling values in the header
    return cov
