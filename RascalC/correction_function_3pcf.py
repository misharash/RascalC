import os
import numpy as np
from typing import Callable
from scipy.special import legendre
from .correction_function import compute_V_n_w_bar, load_randoms


def compute_inv_phi_periodic_3pcf(n: int, n_multipoles: int) -> np.ndarray[float]:
    "Compute the survey correction function coefficients for the periodic box geometry."
    ## Output periodic survey correction function
    phi_inv_mult = np.zeros([n, n, n_multipoles])
    
    ## Set to correct periodic survey values
    phi_inv_mult[:, :, 0] = 1

    return phi_inv_mult


def compute_inv_phi_aperiodic_3pcf(n: int, m: int, n_multipoles: int, r_bins: np.ndarray[float], triple_counts: np.ndarray[float], print_function: Callable[[str], None] = print) -> np.ndarray[float]:
    "Compute the survey correction function coefficients for the realistic survey geometry."

    mu_all = np.linspace(-1,1,m+1)
    mu_cen = 0.5*(mu_all[1:]+mu_all[:-1])
    
    ## reshape RRR counts and add symmetries
    RRR_true = triple_counts.reshape(n, n, m)
    RRR_true = (RRR_true + RRR_true.transpose(1, 0, 2)) / 2
        
    ## Now construct Legendre moments
    leg_triple = np.zeros([n, n, n_multipoles])
    for ell in range(n_multipoles):
        # (NB: we've absorbed a factor of delta_mu into RRR_true here)
        leg_triple[:, :, ell] += (2.*ell+1.) * np.sum(legendre(ell)(mu_cen)[None, None, :] * RRR_true, axis=-1)
    
    # as a precaution, check for negative counts, which should be problematic
    n_mu_check = 2001
    triple_counts_check = np.zeros([n, n, n_mu_check])
    mu_values_check = np.linspace(-1, 1, n_mu_check)
    for ell in range(n_multipoles):
        triple_counts_check += leg_triple[:, :, ell][:, :, None] * legendre(ell)(mu_values_check)[None, None, :]
    problem_indices = np.argwhere(triple_counts_check <= 0)
    if len(problem_indices) > 0:
        rbins, mu_counts = np.unique(problem_indices[:, :2], axis=0, return_counts=True)
        for ((rbin1, rbin2), mu_count) in zip(rbins, mu_counts):
            if rbin1 > rbin2: continue # this case can be skipped by symmetry
            print_function(f"WARNING: counts are not positive for radial bin pair {rbin1}, {rbin2} for {mu_count} mu values of {n_mu_check} checked")

    vol_r = 4 * np.pi / 3 * (r_bins[:, 1] **3 - r_bins[:, 0] ** 3)

    ## Construct inverse multipoles of Phi
    phi_inv_mult = leg_triple / (.5 * vol_r[:, None, None] * vol_r[None, :, None])
            
    ## Check all seems reasonable
    if np.mean(phi_inv_mult[:,:,0])<1e-3:
        print_function(phi_inv_mult[:,:,0])
        raise ValueError("Survey correction function seems too small - are the RRR counts normalized correctly?")
    if np.mean(phi_inv_mult[:,:,0])>1e3:
        raise ValueError("Survey correction function seems too large - are the RRR counts normalized correctly?")

    return phi_inv_mult


def compute_3pcf_correction_function(randoms_pos: np.ndarray[float], randoms_weights: np.ndarray[float], binfile: str, outdir: str, periodic: bool, RRR_file: str | None = None, print_function: Callable[[str], None] = print) -> str:
    """
    Function to compute the multipole decomposition of the 3PCF inverse survey correction function.
    The 3PCF survey correction function is defined as the ratio between idealistic and true RRR pair counts for a single survey.

    NB: Input RRR counts should be normalized by the cube of the sum of random weights here.
    NB: Assume mu is in [-1,1] limit here
    """

    n_multipoles = 7 # matches the value hard-coded in the C++ code

    if periodic:
        print_function("Assuming periodic boundary conditions - so Phi(r,mu) = 1 everywhere")
    elif RRR_file is None:
        raise TypeError("RRR counts file is required if aperiodic")

    V, n_bar, w_bar = compute_V_n_w_bar(randoms_pos, randoms_weights)

    # Load in binning files 
    r_bins = np.loadtxt(binfile)
    n=len(r_bins)

    ## Define normalization constant
    norm = 6. * V * n_bar**3 * w_bar**3 # I don't think there is an exactly right answer once number density or weights vary across the survey

    print_function("Normalizing output survey correction by %.2e"%norm)

    if periodic:
        phi_inv_mult = compute_inv_phi_periodic_3pcf(n, n_multipoles)

    else:
        triple_counts = np.loadtxt(RRR_file)*np.sum(randoms_weights)**3
    
        # Compute number of angular bins in data-set
        m = (len(triple_counts)//n)//n
        if len(triple_counts) % m != 0: raise ValueError("Incorrect RRR format")

        phi_inv_mult = compute_inv_phi_aperiodic_3pcf(n, m, n_multipoles, r_bins, triple_counts / norm, print_function=print_function)
        
    if periodic:
        outfile = os.path.join(outdir, 'BinCorrectionFactor3PCF_n%d_periodic.txt'%(n))
    else:
        outfile = os.path.join(outdir, 'BinCorrectionFactor3PCF_n%d_m%d.txt'%(n,m))
    
    np.savetxt(outfile, phi_inv_mult.reshape(n*n, n_multipoles) * norm)
    print_function("Saved (normalized) output to %s\n"%outfile)

    return outfile


def compute_3pcf_correction_function_from_files(random_filename: str, binfile: str, outdir: str, periodic: bool, RRR_file: str | None = None, print_function: Callable[[str], None] = print) -> str:
    print_function("Loading randoms")
    return compute_3pcf_correction_function(*load_randoms(random_filename), binfile, outdir, periodic, RRR_file, print_function = print_function)


def compute_3pcf_correction_function_from_encore(randoms_pos: np.ndarray[float], randoms_weights: np.ndarray[float], binfile: str, outdir: str, triple_counts: np.ndarray, print_function: Callable[[str], None] = print) -> str:
    """
    Function to compute the multipole decomposition of the 3PCF inverse survey correction function from ENCORE triple counts.
    The 3PCF survey correction function is defined as the ratio between idealistic and true RRR pair counts for a single survey.

    NB: Input RRR counts are not normalized here, and are already in multipole format.
    Caveat: ENCORE only computes the triple counts for pairs of different radial bins, whereas RascalC expects them for all pairs of radial bins, crucially including the pairs of identical bins. Here we try to fill the missing data for those identical-bin pairs using the neighboring bin pairs. These should only affect the covariance rows and columns corresponding to the identical-bin pairs, which should be removed from the covariance for use with ENCORE 3PCF measurements in the end. So, those missing bin pairs should not matter in the end, but it is nicer not to have complete nonsence in the intermediate products.
    """

    n_multipoles = 7 # matches the value hard-coded in the C++ code

    ells = np.arange(len(triple_counts)) # rows in triple_counts correspond to multipoles, columns - to radial bin pairs; the first column might just contain these ells

    if np.array_equal(triple_counts[:, 0], ells): triple_counts = triple_counts[:, 1:] # remove the first column if it is all ells

    # Load in binning files 
    r_bins = np.loadtxt(binfile)
    n = len(r_bins)

    if triple_counts.shape[1] != n*(n-1)//2: raise ValueError("The shape of RRR_counts is inconsistent with the radial bins provided")
    # check this after removing the ells column if present
    # the columns correspond to radial bins

    # change normalization from ENCORE to simple multipoles used in RascalC
    triple_counts *= ((-1)**ells * np.sqrt(2 * ells + 1) / (4 * np.pi))[:, None] # add the second dimension, corresponding to the radial bins, to avoid indexing errors
    # the ell-dependent factor between the ENCORE 3-point basis functions and Legendre polynomials given by Equation (16) in https://arxiv.org/pdf/2105.08722
    # need to check if it is not division; there might also be a factor of 2 or something similar

    # ensure the number of multipoles in triple_counts is right to avoid indexing errors
    if len(triple_counts) < n_multipoles: # this seems more likely
        print_function(f"INFO: ENCORE triple counts have {len(triple_counts)} multipoles, fewer than {n_multipoles} used for the survey correction function, extending by zeros")
        triple_counts = np.vstack([triple_counts, np.zeros([n_multipoles - len(triple_counts), triple_counts.shape[1]])])
    elif len(triple_counts) > n_multipoles: # this seems less likely
        print_function(f"INFO: ENCORE triple counts have {len(triple_counts)} multipoles, more than {n_multipoles} used for the survey correction function, discarding the higher multipoles")
        triple_counts = triple_counts[:n_multipoles]

    bin_indices = np.arange(n)
    bin_index1 = np.repeat(bin_indices, n-1-bin_indices)
    bin_index2 = np.concatenate([bin_indices[i+1:] for i in range(n)])
    # bin_index1 and bin_index2 cover all the bin pairs under the condition bin_index1 < bin_index2, the order follows the ENCORE format
    # they could be read from first two non-comment rows of the ENCORE file, but this seems unnecessary

    leg_triple = np.zeros([n, n, n_multipoles])
    leg_triple[bin_index1, bin_index2] = triple_counts.T # fill above the diagonal; transposition puts radial bin pair index first and multipole index last
    leg_triple[bin_index2, bin_index1] = triple_counts.T # fill below the diagonal symmetrically

    # fill the middle diagonal elements
    bin_indices_middle = bin_indices[1:-1]
    leg_triple[bin_indices_middle, bin_indices_middle] = (leg_triple[bin_indices_middle+1, bin_indices_middle] + leg_triple[bin_indices_middle-1, bin_indices_middle]) / 2 # average the neighboring elements along the column. the neighboring elements along the row are the same due to symmetry

    # fill the edge/corner diagonal elements which are more tricky
    leg_triple[0, 0] = (2 * (2 * leg_triple[1, 0] - leg_triple[2, 0]) + (2 * leg_triple[1, 1] - leg_triple[2, 2])) / 3
    leg_triple[-1, -1] = (2 * (2 * leg_triple[-2, -1] - leg_triple[-3, -1]) + (2 * leg_triple[-2, -2] - leg_triple[-3, -3])) / 3
    
    # check for negative counts, which should be problematic
    n_mu_check = 2001
    triple_counts_check = np.zeros([n, n, n_mu_check])
    mu_values_check = np.linspace(-1, 1, n_mu_check)
    for ell in range(n_multipoles):
        triple_counts_check += leg_triple[:, :, ell][:, :, None] * legendre(ell)(mu_values_check)[None, None, :]
    problem_indices = np.argwhere(triple_counts_check <= 0)
    if len(problem_indices) > 0:
        rbins, mu_counts = np.unique(problem_indices[:, :2], axis=0, return_counts=True)
        for ((rbin1, rbin2), mu_count) in zip(rbins, mu_counts):
            if rbin1 > rbin2: continue # this case can be skipped by symmetry
            print_function(("INFO" if rbin1 == rbin2 else "WARNING") + f": counts are not positive for radial bin pair {rbin1}, {rbin2} for {mu_count} mu values of {n_mu_check} checked")
            # the problem for same-bin pairs is less critical (and seems more likely), for different-bin pairs it is more critical

    vol_r = 4 * np.pi / 3 * (r_bins[:, 1] ** 3 - r_bins[:, 0] ** 3) # volume of radial/separation bins as 1D array

    V, n_bar, w_bar = compute_V_n_w_bar(randoms_pos, randoms_weights)

    ## Define normalization constant
    norm = 6. * V * n_bar**3 * w_bar**3 # I don't think there is an exactly right answer once number density or weights vary across the survey

    ## Construct inverse multipoles of Phi
    phi_inv_mult = leg_triple / (.5 * norm * vol_r[:, None, None] * vol_r[None, :, None])
            
    ## Check all seems reasonable
    if np.mean(phi_inv_mult[:,:,0])<1e-3:
        print_function(phi_inv_mult[:,:,0])
        raise ValueError("Survey correction function seems too small - are the RRR counts normalized correctly?")
    if np.mean(phi_inv_mult[:,:,0])>1e3:
        raise ValueError("Survey correction function seems too large - are the RRR counts normalized correctly?")
        
    outfile = os.path.join(outdir, 'BinCorrectionFactor3PCF_n%d.txt' % n)
    
    np.savetxt(outfile, phi_inv_mult.reshape(n*n, n_multipoles) * norm)
    print_function("Saved (normalized) output to %s\n"%outfile)

    return outfile