## Script to post-process the single-field integrals computed by the C++ code in mixed Legendre (LEGENDRE_MIX) mode. This computes the shot-noise rescaling parameter, alpha, from a data derived covariance matrix.
## We output the data and theory jackknife covariance matrices, in addition to full theory covariance matrices and (quadratic-bias corrected) precision matrices. The effective number of samples, N_eff, is also computed.

import numpy as np
import os
from warnings import warn
from .utils import cov_filter_legendre, cov_filter_smu, load_matrices_single, check_eigval_convergence, add_cov_terms_single, check_positive_definiteness, compute_D_precision_matrix, compute_N_eff_D, fit_shot_noise_rescaling
from ..raw_covariance_matrices import load_raw_covariances_legendre, Iterable
from .jackknife import load_disconnected_term_single


def post_process_legendre_mix_jackknife(jackknife_file: str, weight_dir: str, file_root: str, m: int, max_l: int, outdir: str, skip_r_bins: int = 0, skip_l: int = 0, tracer: int = 1, n_samples: None | int | Iterable[int] | Iterable[bool] = None, load_disconnected_term: bool = False, print_function = print) -> dict[str]:
    # Load jackknife xi estimates from data
    print_function("Loading correlation function jackknife estimates from %s" % jackknife_file)
    xi_jack = np.loadtxt(jackknife_file, skiprows = 2)
    n_jack = xi_jack.shape[0] # total jackknives
    n = xi_jack.shape[1] // m # radial bins
    n_l = max_l // 2 + 1 # number of even multipoles
    n_bins = (n_l - skip_l) * (n - skip_r_bins) # total Legendre bins to work with

    weight_file = os.path.join(weight_dir, 'jackknife_weights_n%d_m%d_j%d_11.dat' % (n, m, n_jack))
    mu_bin_legendre_file = os.path.join(weight_dir, 'mu_bin_legendre_factors_m%d_l%d.txt' % (m, max_l))

    print_function("Loading jackknife weights from %s" % weight_file)
    weights = np.loadtxt(weight_file)[:, 1:]

    # First exclude any dodgy jackknife regions
    good_jk = np.where(np.all(np.isfinite(xi_jack), axis=1))[0] # all xi in jackknife have to be normal numbers
    if len(good_jk) < n_jack:
        warn("Using only %d out of %d jackknives - some xi values were not finite" % (len(good_jk), n_jack))
        xi_jack = xi_jack[good_jk]
        weights = weights[good_jk]
    weights /= np.sum(weights, axis=0) # renormalize weights after possibly discarding some jackknives

    # Compute data covariance matrix
    print_function("Computing data covariance matrix")
    mean_xi = np.sum(xi_jack * weights, axis = 0)
    tmp = weights * (xi_jack - mean_xi)
    data_cov = np.matmul(tmp.T, tmp)
    denom = np.matmul(weights.T, weights)
    data_cov /= (np.ones_like(denom) - denom)

    print("Loading mu bin Legendre factors from %s" % mu_bin_legendre_file)
    mu_bin_legendre_factors = np.loadtxt(mu_bin_legendre_file) # rows correspond to mu bins, columns to multipoles
    if skip_l > 0: mu_bin_legendre_factors = mu_bin_legendre_factors[:, :-skip_l] # discard unneeded l; the expression works wrong for skip_l=0

    # Project the data jackknife covariance from mu bins to Legendre multipoles
    data_cov = data_cov.reshape(n, m, n, m) # make the array 4D with [r_bin, mu_bin] indices for rows and columns
    data_cov = data_cov[skip_r_bins:, :, skip_r_bins:, :] # discard the extra radial bins now since it is convenient
    data_cov = np.einsum("imjn,mp,nq->ipjq", data_cov, mu_bin_legendre_factors, mu_bin_legendre_factors) # use mu bin Legendre factors to project mu bins into Legendre multipoles, staying within the same radial bins. The indices are now [r_bin, ell] for rows and columns
    data_cov = data_cov.reshape(n_bins, n_bins)

    cov_filter = cov_filter_legendre(n, max_l, skip_r_bins, skip_l)
    n_l = max_l // 2 + 1 # number of multipoles
    
    input_file = load_raw_covariances_legendre(file_root, n, max_l, n_samples, print_function)

    # Create output directory
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # Load in full jackknife theoretical matrices
    print_function("Loading best estimate of jackknife covariance matrix")
    c2j, c3j, c4j = load_matrices_single(input_file, cov_filter, tracer, full = True, jack = True)

    # Load the full disconnected term if present
    if load_disconnected_term:
        disconnected_loaded = False
        import traceback
        try:
            cov_filter_disconnected = cov_filter_smu(n, m, skip_r_bins)
            RR_file = os.path.join(weight_dir, f'binned_pair_counts_n{n}_m{m}_j{n_jack}_{tracer}{tracer}.dat')
            RR = np.loadtxt(RR_file)
            cjd = load_disconnected_term_single(input_file, cov_filter_disconnected, RR, weights, tracer, full = True) # in s,mu bins
            cjd = cjd.reshape(n - skip_r_bins, m, n - skip_r_bins, m) # make the array 4D with [r_bin, mu_bin] indices for rows and columns
            cjd = np.einsum("imjn,mp,nq->ipjq", cjd, mu_bin_legendre_factors, mu_bin_legendre_factors) # use mu bin Legendre factors to project mu bins into Legendre multipoles, staying within the same radial bins. The indices are now [r_bin, ell] for rows and columns
            cjd = cjd.reshape(n_bins, n_bins) # convert the array back into 2D covariance matrix
            c4j += cjd # add to the 4-point term - unaffected by rescaling
            disconnected_loaded = True
        except Exception:
            warn("Could not load the full jackknife disconnected term. This is not critical, but if you think it should have been computed, check the errors below:")
            traceback.print_exc()

    # Check matrix convergence
    check_eigval_convergence(c2j, c4j, "Jackknife")

    # Load in partial jackknife theoretical matrices
    c2s, c3s, c4s = load_matrices_single(input_file, cov_filter, tracer, full = False, jack = True)

    # Load the partial disconnected terms if present
    if load_disconnected_term:
        try:
            csd = load_disconnected_term_single(input_file, cov_filter_disconnected, RR, weights, tracer, full = False) # in s,mu bins
            csd = csd.reshape(len(c4s), n - skip_r_bins, m, n - skip_r_bins, m) # make the array 3D with [r_bin, mu_bin] indices for rows and columns, and subsample index in front
            csd = np.einsum("kimjn,mp,nq->kipjq", csd, mu_bin_legendre_factors, mu_bin_legendre_factors) # use mu bin Legendre factors to project mu bins into Legendre multipoles, staying within the same radial bins. The indices are now [r_bin, ell] for rows and columns. Additionally, the very first index is for the subsample.
            csd = csd.reshape(len(c4s), n_bins, n_bins) # convert back into an array of 2D covariance matrices, with subsample index in front
            c4s += csd # add to the 4-point term - unaffected by rescaling
        except Exception:
            if disconnected_loaded: # only warn if succeeded previously
                warn("Could not load the partial jackknife disconnected terms. This is not critical, but if you think they should have been computed, check the errors below:")
                traceback.print_exc()

    # Now optimize for shot-noise rescaling parameter alpha
    print_function("Optimizing for the shot-noise rescaling parameter")
    alpha_best = fit_shot_noise_rescaling(data_cov, c2j, c3j, c4j, c2s, c3s, c4s)
    print_function("Optimization complete - optimal rescaling parameter is %.6f" % alpha_best)

    # Compute jackknife and full covariance matrices
    jack_cov = add_cov_terms_single(c2j, c3j, c4j, alpha_best)
    partial_jack_cov = add_cov_terms_single(c2s, c3s, c4s, alpha_best)
    _, jack_prec = compute_D_precision_matrix(partial_jack_cov, jack_cov)

    c2f, c3f, c4f = load_matrices_single(input_file, cov_filter, tracer, full = True, jack = False)
    full_cov = add_cov_terms_single(c2f, c3f, c4f, alpha_best)

    # Check convergence
    check_eigval_convergence(c2f, c4f, "Full")

    # Check positive definiteness
    check_positive_definiteness(full_cov)

    # Compute full precision matrix
    print_function("Computing the full precision matrix estimate:")
    # Load in partial jackknife theoretical matrices
    c2fs, c3fs, c4fs = load_matrices_single(input_file, cov_filter, tracer, full = False, jack = False)
    partial_cov = add_cov_terms_single(c2fs, c3fs, c4fs, alpha_best)
    full_D_est, full_prec = compute_D_precision_matrix(partial_cov, full_cov)
    print_function("Full precision matrix estimate computed")    

    # Now compute effective N:
    N_eff_D = compute_N_eff_D(full_D_est, print_function)

    output_dict = {"jackknife_theory_covariance": jack_cov, "full_theory_covariance": full_cov, "jackknife_data_covariance": data_cov, "shot_noise_rescaling": alpha_best, "jackknife_theory_precision": jack_prec, "full_theory_precision": full_prec, "N_eff": N_eff_D, "full_theory_D_matrix": full_D_est, "individual_theory_covariances": partial_cov, "individual_theory_jackknife_covariances": partial_jack_cov}

    output_name = os.path.join(outdir, 'Rescaled_Covariance_Matrices_Legendre_Jackknife_n%d_l%d_j%d.npz' % (n, max_l, n_jack))
    np.savez_compressed(output_name, **output_dict)

    print_function("Saved output covariance matrices as %s"%output_name)

    return output_dict
