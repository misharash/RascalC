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