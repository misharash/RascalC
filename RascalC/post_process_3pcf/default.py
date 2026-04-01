"""
Function to post-process the single-field 3PCF Legendre binned integrals computed by the C++ code.
We output the theoretical covariance matrices, (quadratic-bias corrected) precision matrices and the effective number of samples, N_eff.
"""

import numpy as np
import os
from ..utils import symmetrized, format_skip_r_bins
from ..post_process.utils import apply_cov_filter, check_eigval_convergence, check_positive_definiteness, compute_D_precision_matrix, compute_N_eff_D
from ..raw_covariance_matrices import load_raw_covariances_3pcf_legendre
from typing import Callable, Iterable


def cov_filter_3pcf_legendre(n: int, max_l: int, skip_r_bins: int | tuple[int, int] = 0, skip_l: int = 0, exclude_samebins: bool = True, exclude_odd_l: bool = False):
    """Produce a 2D indexing array for 3PCF Legendre covariance matrices in RascalC convention (not yet fully compatible with ENCORE bin ordering)."""
    skip_r_bins_start, skip_r_bins_end = format_skip_r_bins(skip_r_bins)
    n_l = max_l + 1
    l_indices = np.arange(0, n_l, 1+exclude_odd_l)
    if skip_l > 0: l_indices = l_indices[:-skip_l] # without the condition, wouldn't work right for skip_l
    r_indices = np.arange(skip_r_bins_start, n - skip_r_bins_end)
    r_indices1, r_indices2 = [a.ravel() for a in np.meshgrid(r_indices, r_indices, indexing='ij')] # flattened array indices
    r_filter = (r_indices1 <= r_indices2 - exclude_samebins) # strictly less for exclude_samebin=True, less or equal otherwise
    r_indices1, r_indices2 = r_indices1[r_filter], r_indices2[r_filter] # apply filter to both index arrays
    indices_l_r = (n_l * (n * r_indices1 + r_indices2))[:, None] + l_indices[None, :]
    # indices_l_r = (n_l * (n * r_indices1 + r_indices2))[None, :] + l_indices[:, None] # could switch the l vs bin pair ordering right here easily but decided not to
    indices_1d = indices_l_r.ravel()
    return np.ix_(indices_1d, indices_1d)


def symmetrized_3pcf(A: np.typing.NDArray[np.float64], n: int, max_l: int) -> np.typing.NDArray[np.float64]:
    "Symmetrize a 3PCF covariance matrix (2D array), or an array of 3PCF covariance matrices (3D array)"
    if len(A.shape) not in (2, 3): raise ValueError("Dimension of the input array must be 2 or 3")
    m = max_l + 1
    if not np.array_equal(A.shape[-2:], [n * n * m] * 2): raise ValueError("Unexpected shape in the last 2 dimensions")
    leading_dims = list(A.shape[:-2]) # list containing a leading dimension for the array of covariance matrices, and empty for a single covariance matrix
    A1 = A.reshape(leading_dims + [n, n, m] * 2) # last 6 axes will be [r1, r2, l12, r3, r4, l34]
    A2 = (A1 + A1.swapaxes(-2, -3)) / 2 # symmetrize wrt swaps of r3 and r4. Create new array against the risk of A1 being a view of original A
    A2 = (A2 + A2.swapaxes(-5, -6)) / 2 # symmetrize wrt swaps of r1 and r2
    A2 = A2.reshape(leading_dims + [n * n * m] * 2) # back to the original shape
    return symmetrized(A2) # finally, symetrize wrt full covariance matrix bin swap


def load_matrices(input_data: dict[str], n: int, max_l: int, cov_filter: np.typing.NDArray[np.int_], full: bool = True, use_c6_0: bool = False) -> tuple[np.typing.NDArray[np.float64], np.typing.NDArray[np.float64], np.typing.NDArray[np.float64], np.typing.NDArray[np.float64]]:
    """Load the 3PCF single-tracer covariance matrix terms."""
    matrices = []
    for npoints in range(3, 7):
        these_matrices = [input_data[f"c{npoints}_{index}" + "_full" * full] for index in range(2)]
        this_matrix = these_matrices[0] * (use_c6_0 or npoints != 6) + these_matrices[1] # by default, exclude c6_0 term here (but why? need to test)
        matrices.append(apply_cov_filter(symmetrized_3pcf(this_matrix, n, max_l), cov_filter)) # symmetrize before filtering, because filtering removes repeating bin pairs
    return tuple(matrices)


def add_cov_terms(c3: np.typing.NDArray[np.float64], c4: np.typing.NDArray[np.float64], c5: np.typing.NDArray[np.float64], c6: np.typing.NDArray[np.float64], alpha: float = 1) -> np.typing.NDArray[np.float64]:
    """Add the 3PCF single-tracer covariance matrix terms with a given shot-noise rescaling value."""
    return c6 + c5 * alpha + c4 * alpha**2 + c3 * alpha**3


def post_process_3pcf(file_root: str, n: int, max_l: int, outdir: str | None = None, alpha: float = 1, skip_r_bins: int | tuple[int, int] = 0, skip_l: int = 0, n_samples: None | int | Iterable[int] | Iterable[bool] = None, exclude_samebins: bool = True, exclude_odd_l: bool = False, use_c6_0: bool = False, print_function: Callable[[str], None] = print, dry_run: bool = False) -> dict[str]:
    r"""
    3PCF post-processing for Legendre (accumulated) mode.

    Do not run this (or any other post-processing function/script) while the main RascalC computation is running — this may delete the output directory and cause the code to crash.

    Parameters
    ----------
    file_root : string
        Path to the RascalC (:func:`RascalC.run_cov_3pcf` or command-line) output directory.
    
    n : integer
        The number of radial bins used in the RascalC run (before applying ``skip_r_bins`` if it is provided).
    
    max_l : integer
        The maximum ell (Legendre moment index) used in the RascalC run (before applying ``skip_l`` if it is provided).

    outdir : string or None
        (Optional) path to the directory in which the post-processing results should be saved. If None (default), is set to ``file_root``. Empty string means the current working directory.
        We advise to use different output directories for different post-processing options.

    alpha : float
        Fixed shot-noise rescaling value to use. In principle optional, but the default value of 1 may not be particularly good.

    skip_r_bins : integer or tuple of two integers
        (Optional) removal of some radial bins.
        First (or the only) number sets the number of radial/separation bins to skip from the beginning.
        Second number (if provided) sets the number of radial/separation bins to skip from the end.
        By default, no bins are skipped.

    skip_l : integer
        (Optional) number of higher multipoles to skip (from the end, counting all multipoles by default and only even multipoles if exlude_odd_l is True).

    n_samples : None, integer, array/list/tuple/etc of integers or boolean values
        (Optional) selection of RascalC subsamples (independent realizations of Monte-Carlo integrals).
        
            - If None, use all (default).
            - If an integer, use the given number of samples from the beginning.
            - If an array/list/tuple/etc of integers, it will be used as a NumPy index array.
            - If an array/list/tuple/etc of boolean, it will be used as a NumPy boolean array mask.
    
    exclude_samebins : boolean
        (Optional) If False, the covariance will include the pairs of the same radial bins.
        The default behavior (for the True value) is to exclude them for compatibility with ENCORE.
        In either case, the post-processed covariances only include each pair of different radial bins in one ordering, ``bin1 < bin2``; the raw covariances also include ``bin1 > bin2`` pairs.
    
    exclude_odd_l : boolean
        (Optional) If True, the covariance will exclude the odd multipoles; note that then they will also not count in ``skip_l``. By default (False value), odd multipoles are kept and counted in ``skip_l``.
    
    print_function : Callable
        (Optional) custom function to use for printing. Default is ``print``.
    
    dry_run: boolean
        (Optional) If True, this will not run actual post-processing, only determine the filename and path (see below).

    Returns
    -------
    post_processing_results : dict[str, np.ndarray[float]]
        Post-processing results as a dictionary with string keys and Numpy array values. All this information is also saved in a ``Rescaled_Covariance_Matrices*.npz`` file in the ``out_dir`` (in ``file_root`` if the former is not provided).
        Selected common keys are: ``"full_theory_covariance"`` for the final covariance matrix and ``"shot_noise_rescaling"`` for the shot-noise rescaling value(s).
        For convenience, in the output dictionary only, ``"filename"`` contains the name of the file where the results were saved (which can be inconvenient to predict), and ``"path"`` contains its path (also obtainable by :func:`os.path.join`-ing ``out_dir`` with the filename)
    """
    # Set default output directory if not set
    if outdir is None: outdir = file_root

    output_name = os.path.join(outdir, 'Rescaled_Covariance_Matrices_3PCF_n%d_l%d.npz' % (n, max_l))
    name_dict = dict(path=output_name, filename=os.path.basename(output_name))
    if dry_run: return name_dict

    cov_filter = cov_filter_3pcf_legendre(n, max_l, skip_r_bins, skip_l, exclude_samebins, exclude_odd_l)
    
    input_file = load_raw_covariances_3pcf_legendre(file_root, n, max_l, n_samples, print_function)

    # Create output directory
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # Load in full theoretical matrices
    print_function("Loading best estimate of covariance matrix")
    c3, c4, c5, c6 = load_matrices(input_file, n, max_l, cov_filter, full=True, use_c6_0=use_c6_0)

    # Check matrix convergence by analogy with 2PCF, may be less helpful
    check_eigval_convergence(c3, c6, alpha, Npcf=3, print_function=print_function)

    # Compute full covariance matrix
    full_cov = add_cov_terms(c3, c4, c5, c6, alpha)

    # Check positive definiteness
    check_positive_definiteness(full_cov)

    # Compute full precision matrix
    print_function("Computing the full precision matrix estimate:")
    # Load in partial theoretical matrices
    c3s, c4s, c5s, c6s = load_matrices(input_file, n, max_l, cov_filter, full=False, use_c6_0=use_c6_0)
    partial_cov = add_cov_terms(c3s, c4s, c5s, c6s, alpha)
    full_D_est, full_prec = compute_D_precision_matrix(partial_cov, full_cov)
    print_function("Full precision matrix estimate computed")

    # Now compute effective N:
    N_eff_D = compute_N_eff_D(full_D_est, print_function)  

    output_dict = dict(full_theory_covariance=full_cov, shot_noise_rescaling=alpha,
                       full_theory_precision=full_prec, N_eff=N_eff_D,
                       full_theory_D_matrix=full_D_est, individual_theory_covariances=partial_cov)
    
    np.savez_compressed(output_name, **output_dict)
    print_function("Saved output covariance matrices as %s"%output_name)

    return output_dict | name_dict