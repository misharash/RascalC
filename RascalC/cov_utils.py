"Simple convenience functions that read RascalC results and save full cov to text file, in addition checking eigenvalues of bias matrix"

import numpy as np
from typing import Callable
from .get_shot_noise_rescaling import get_shot_noise_rescaling


def get_cov_header(rascalc_results_file: str) -> str:
    return "shot_noise_rescaling = " + str(get_shot_noise_rescaling(rascalc_results_file))


def convert_cov_legendre(cov: np.ndarray[float], max_l: int) -> np.ndarray[float]:
    if max_l % 2 != 0: raise ValueError("Only even multipoles supported")
    n_l = max_l // 2 + 1
    n_bins = len(cov)
    if n_bins % n_l != 0: raise ValueError("Number of bins in the covariance must be divisible by the number of even multipoles")
    n_r_bins = n_bins // n_l
    cov = cov.reshape(n_r_bins, n_l, n_r_bins, n_l) # convert to 4D from 2D with [r, l] ordering for both rows and columns
    cov = cov.transpose(1, 0, 3, 2) # change orderng to [l, r] for both rows and columns
    cov = cov.reshape(n_bins, n_bins) # convert back from 4D to 2D
    return cov


def convert_cov_legendre_multi(cov: np.ndarray[float], max_l: int) -> np.ndarray[float]:
    if max_l % 2 != 0: raise ValueError("Only even multipoles supported")
    n_l = max_l // 2 + 1
    n_bins = len(cov)
    if n_bins % (3 * n_l) != 0: raise ValueError("Number of bins in the covariance must be divisible by thrice the number of even multipoles")
    n_r_bins = n_bins // (3 * n_l)
    cov = cov.reshape(3, n_r_bins, n_l, 3, n_r_bins, n_l) # convert to 6D from 2D with [t, r, l] ordering for both rows and columns
    cov = cov.transpose(0, 2, 1, 3, 5, 4) # change orderng to [t, l, r] for both rows and columns
    cov = cov.reshape(n_bins, n_bins) # convert back from 6D to 2D
    return cov


def load_cov(rascalc_results_file: str, print_function: Callable = print) -> np.ndarray[float]:
    with np.load(rascalc_results_file) as f:
        print_function(f"Max abs eigenvalue of bias correction matrix is {np.max(np.abs(np.linalg.eigvals(f['full_theory_D_matrix']))):.2e}")
        # if the printed value is small the cov matrix should be safe to invert as is
        return f['full_theory_covariance']


def load_cov_legendre(rascalc_results_file: str, max_l: int, print_function: Callable = print) -> np.ndarray[float]:
    return convert_cov_legendre(load_cov(rascalc_results_file, print_function), max_l)


def load_cov_legendre_multi(rascalc_results_file: str, max_l: int, print_function: Callable = print) -> np.ndarray[float]:
    return convert_cov_legendre_multi(load_cov(rascalc_results_file, print_function), max_l)


def export_cov(rascalc_results_file: str, output_cov_file: str, print_function: Callable = print) -> None:
    np.savetxt(output_cov_file, load_cov(rascalc_results_file, print_function = print_function), header = get_cov_header(rascalc_results_file))


def export_cov_legendre(rascalc_results_file: str, max_l: int, output_cov_file: str, print_function: Callable = print) -> None:
    np.savetxt(output_cov_file, load_cov_legendre(rascalc_results_file, max_l, print_function = print_function), header = get_cov_header(rascalc_results_file))


def export_cov_legendre_multi(rascalc_results_file: str, max_l: int, output_cov_file: str, print_function: Callable = print) -> None:
    np.savetxt(output_cov_file, load_cov_legendre_multi(rascalc_results_file, max_l, print_function = print_function), header = get_cov_header(rascalc_results_file))
