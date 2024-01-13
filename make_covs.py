# This script generates all covs

import os
from datetime import datetime
import pickle
import hashlib
from typing import Callable
from RascalC.raw_covariance_matrices import cat_raw_covariance_matrices, collect_raw_covariance_matrices
from RascalC.post_process.legendre_multi import post_process_legendre_multi
from RascalC.convergence_check_extra import convergence_check_extra
from RascalC.cov_utils import export_cov_legendre_multi
from RascalC.comb.combine_covs_legendre_multi import combine_covs_legendre_multi

max_l = 4
nbin = 50 # radial bins for output cov
rmax = 200 # maximum output cov radius in Mpc/h

jackknife = 0
njack = 60 if jackknife else 0
if jackknife: mbin = 100

version_label = "v0.6"

regs = ('SGC', 'NGC') # regions for filenames
reg_comb = "GCcomb"

tracers = [['LRG', 'ELG_LOPnotqso']]
zs = [[0.8, 1.1]]
alphas_ext = [[[0.97, 0.83], [0.87, 0.81]]] # from single-tracer jackknives, external to these runs. 

skip_r_bins = 5
skip_l = 0

nrandoms = 4

xilabel = "".join([str(i) for i in range(0, max_l+1, 2)])

r_step = rmax // nbin
rmin_real = r_step * skip_r_bins

hash_dict_file = "make_covs.hash_dict.pkl"
if os.path.isfile(hash_dict_file):
    # Load hash dictionary from file
    with open(hash_dict_file, "rb") as f:
        hash_dict = pickle.load(f)
else:
    # Initialize hash dictionary as empty
    hash_dict = {}
# Hash dict keys are goal filenames, the elements are also dictionaries with dependencies/sources filenames as keys

# Set up logging
logfile = "make_covs.log.txt"

def print_and_log(s: object = "") -> None:
    print(s)
    print_log(s)
print_log = lambda l: os.system(f"echo \"{l}\" >> {logfile}")

print_and_log(datetime.now())
print_and_log(f"Executing {__file__}")

def my_make(goal: str, deps: list[str], recipe: Callable, force: bool = False, verbose: bool = False) -> None:
    need_make, current_dep_hashes = hash_check(goal, deps, force=force, verbose=verbose)
    if need_make:
        print_and_log(f"Making {goal} from {deps}")
        try:
            recipe()
        except Exception as e:
            print_and_log(f"{goal} not built: {e}")
            return
        hash_dict[goal] = current_dep_hashes # update the dependency hashes only if the make was successfully performed
        print_and_log()

def hash_check(goal: str, srcs: list[str], force: bool = False, verbose: bool = False) -> tuple[bool, dict]:
    # First output indicates whether we need to/should execute the recipe to make goal from srcs
    # Also returns the src hashes in the dictionary current_src_hashes
    current_src_hashes = {}
    for src in srcs:
        if not os.path.exists(src):
            if verbose: print_and_log(f"Can not make {goal} from {srcs}: {src} missing\n") # and next operations can be omitted
            return False, current_src_hashes
        current_src_hashes[src] = sha256sum(src)
    if not os.path.exists(goal) or force: return True, current_src_hashes # need to make if goal is missing or we are forcing, but hashes needed to be collected beforehand, also ensuring the existence of sources
    try:
        if set(current_src_hashes.values()) == set(hash_dict[goal].values()): # comparing to hashes of sources used to build the goal last, regardless of order and names. Collisions seem unlikely
            if verbose: print_and_log(f"{goal} uses the same {srcs} as previously, no need to make\n")
            return False, current_src_hashes
    except KeyError: pass # if hash dict is empty need to make, just proceed
    return True, current_src_hashes

def sha256sum(filename: str, buffer_size: int = 128*1024) -> str: # from https://stackoverflow.com/a/44873382
    h = hashlib.sha256()
    b = bytearray(buffer_size)
    mv = memoryview(b)
    with open(filename, 'rb', buffering=0) as f:
        while n := f.readinto(mv):
            h.update(mv[:n])
    return h.hexdigest()

# Make steps for making covs
for tlabels, (z_min, z_max), these_alphas_ext in zip(tracers, zs, alphas_ext):
    corlabels = [tlabels[0], "_".join(tlabels), tlabels[1]]
    reg_results, reg_pycorr_names = [], []
    reg_results_rescaled = []
    for reg, alphas in zip(regs, these_alphas_ext):
        outdir = "_".join(tlabels + [reg]) + f"_z{z_min}-{z_max}" # output file directory
        if not os.path.isdir(outdir): continue # if doesn't exist can't really do anything else
        
        raw_name = os.path.join(outdir, f"Raw_Covariance_Matrices_n{nbin}_l{max_l}.npz")

        # detect the per-file dirs if any
        outdirs_perfile = [int(name) for name in os.listdir(outdir) if name.isdigit()] # per-file dir names are pure integers
        if len(outdirs_perfile) > 0: # if such dirs found, need to cat the raw covariance matrices
            outdirs_perfile = [os.path.join(outdir, str(index)) for index in sorted(outdirs_perfile)] # sort integers, transform back to strings and prepend the parent directory
            cat_raw_covariance_matrices(nbin, f"l{max_l}", outdirs_perfile, [None] * len(outdirs_perfile), outdir, print_function = print_and_log) # concatenate the subsamples
        else: # if no subdirs found, run the raw matrix collection just in case
            collect_raw_covariance_matrices(outdir, print_function = print_and_log)

        # Gaussian covariances

        results_name = os.path.join(outdir, 'Rescaled_Covariance_Matrices_Legendre_n%d_l%d.npz' % (nbin, max_l))
        reg_results.append(results_name)
        cov_name = "xi" + xilabel + "_" + "_".join(tlabels + [reg]) + f"_{z_min}_{z_max}_default_FKP_lin{r_step}_s{rmin_real}-{rmax}_cov_RascalC_Gaussian.txt"
        reg_pycorr_names.append([f"/global/cfs/cdirs/desi/users/dvalcin/EZMOCKS/Overlap/Y1/FOR_MISHA/{version_label}/allcounts_{corlabel}_{reg}_{z_min}_{z_max}_default_FKP_lin_njack{njack}_nran{nrandoms}.npy" for corlabel in corlabels])

        def make_gaussian_cov():
            results = post_process_legendre_multi(outdir, nbin, max_l, outdir, skip_r_bins = skip_r_bins, skip_l = skip_l, print_function = print_and_log)
            convergence_check_extra(results, print_function = print_and_log)

        # RascalC results depend on full output (most straightforwardly)
        my_make(results_name, [raw_name], make_gaussian_cov)
        # Recipe: run post-processing
        # Also perform convergence check (optional but nice)

        # Individual cov file depends on RascalC results
        my_make(cov_name, [results_name], lambda: export_cov_legendre_multi(results_name, max_l, cov_name))
        # Recipe: export cov

        # Post-processing with external alphas
        if alphas:
            results_name_rescaled = os.path.join(outdir, 'rescaled/Rescaled_Covariance_Legendre_n%d_l%d.npz' % (nbin, max_l))
            reg_results_rescaled.append(results_name_rescaled)

            def make_rescaled_cov():
                results = post_process_legendre_multi(outdir, nbin, max_l, os.path.dirname(results_name_rescaled), skip_r_bins = skip_r_bins, skip_l = skip_l, alpha_1 = alphas[0], alpha_2 = alphas[1], print_function = print_and_log)
                convergence_check_extra(results, print_function = print_and_log)

            # RascalC results depend on full output (most straightforwardly)
            my_make(results_name_rescaled, [raw_name], make_rescaled_cov)
            # Recipe: run post-processing
            # Also perform convergence check (optional but nice)

            # Load shot-noise rescaling and make name
            cov_name_rescaled = "xi" + xilabel + "_" + "_".join(tlabels + [reg]) + f"_{z_min}_{z_max}_default_FKP_lin{r_step}_s{rmin_real}-{rmax}_cov_RascalC_rescaled.txt"
            # Individual cov file depends on RascalC results
            my_make(cov_name_rescaled, [results_name_rescaled], lambda: export_cov_legendre_multi(results_name_rescaled, max_l, cov_name_rescaled))
            # Recipe: run convert cov

    if len(reg_pycorr_names) == len(regs): # if we have pycorr files for all regions
        if len(reg_results) == len(regs): # if we have RascalC results for all regions
            # Combined Gaussian cov

            cov_name = "xi" + xilabel + "_" + "_".join(tlabels + [reg_comb]) + f"_{z_min}_{z_max}_default_FKP_lin{r_step}_s{rmin_real}-{rmax}_cov_RascalC_Gaussian.txt" # combined cov name

            # Comb cov depends on the region RascalC results
            my_make(cov_name, reg_results, lambda: combine_covs_legendre_multi(*reg_results, *reg_pycorr_names, cov_name, max_l, r_step = r_step, skip_r_bins = skip_r_bins, print_function = print_and_log))
            # Recipe: run combine covs

        if len(reg_results_rescaled) == len(regs): # if we have RascalC rescaled results for all regions
            # Combined rescaled cov
            cov_name_rescaled = "xi" + xilabel + "_" + "_".join(tlabels + [reg_comb]) + f"_{z_min}_{z_max}_default_FKP_lin{r_step}_s{rmin_real}-{rmax}_cov_RascalC_rescaled.txt" # combined cov name

            # Comb cov depends on the region RascalC results
            my_make(cov_name_rescaled, reg_results_rescaled, lambda: combine_covs_legendre_multi(*reg_results_rescaled, *reg_pycorr_names, cov_name_rescaled, max_l, r_step = r_step, skip_r_bins = skip_r_bins, print_function = print_and_log))
            # Recipe: run combine covs

# Save the updated hash dictionary
with open(hash_dict_file, "wb") as f:
    pickle.dump(hash_dict, f)

print_and_log(datetime.now())
print_and_log("Finished execution.")
