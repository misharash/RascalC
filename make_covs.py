# This script generates all covs

import os, sys
from datetime import datetime
import pickle
import hashlib
import numpy as np
from glob import glob
import fnmatch

max_l = 4
nbin = 50 # radial bins for output cov
rmax = 200 # maximum output cov radius in Mpc/h

jackknife = 1
njack = 60 if jackknife else 0
if jackknife: mbin = 100

version_label = "v0.6"

regs = ('SGC', 'NGC') # regions for filenames
reg_comb = "GCcomb"

tracers = ['LRG'] * 4 + ['ELG_LOP'] * 3 + ['BGS_BRIGHT-21.5', 'QSO']
tracers = [tracer + "_ffa" for tracer in tracers]
zs = [[0.4, 0.6], [0.6, 0.8], [0.8, 1.1], [0.4, 1.1], [0.8, 1.1], [1.1, 1.6], [0.8, 1.6], [0.1, 0.4], [0.8, 2.1]]

skip_bins = 5
skip_l = 0

split_above = 20

xilabel = "".join([str(i) for i in range(0, max_l+1, 2)])

r_step = rmax // nbin
nbin_final = nbin - skip_bins
rmin_real = r_step * skip_bins

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

def exec_print_and_log(commandline: str, terminate_on_error: bool = False) -> None:
    print_and_log(f"Running command: {commandline}")
    if commandline.startswith("python"): # additional anti-buffering for python
        commandline = commandline.replace("python", "python -u", 1)
    status = os.system(f"bash -c 'set -o pipefail; stdbuf -oL -eL {commandline} 2>&1 | tee -a {logfile}'")
    # tee prints what it gets to stdout AND saves to file
    # stdbuf -oL -eL should solve the output delays due to buffering without hurting the performance too much
    # without pipefail, the exit_code would be of tee, not reflecting main command failures
    # feed the command to bash because on Ubuntu it was executed in sh (dash) where pipefail is not supported
    exit_code = os.waitstatus_to_exitcode(status) # assumes we are in Unix-based OS; on Windows status is the exit code
    if exit_code:
        print(f"{commandline} exited with error (code {exit_code}).")
        if terminate_on_error:
            print("Terminating the script execution due to this error.")
            sys.exit(1)

def my_make(goal: str, deps: list[str], *cmds: tuple[str], force: bool = False, verbose: bool = False) -> None:
    need_make, current_dep_hashes = hash_check(goal, deps, force=force, verbose=verbose)
    if need_make:
        print_and_log(f"Making {goal} from {deps}")
        for cmd in cmds:
            ret = exec_print_and_log(cmd)
            if ret:
                print_and_log(f"{cmd} exited with error (code {ret}). Aborting\n")
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
for tracer, (z_min, z_max) in zip(tracers, zs):
    nrandoms = 1 if tracer.startswith("BGS") else 4 # 1 random for BGS only
    tlabels = [tracer]
    reg_results, reg_pycorr_names = [], []
    if jackknife: reg_results_jack = []
    for reg in regs:
        outdir = "_".join(tlabels + [reg]) + f"_z{z_min}-{z_max}" # output file directory
        full_output_names = [os.path.join(outdir, "CovMatricesAll/c2_n%d_l%d_11_full.txt" % (nbin, max_l)), os.path.join(outdir, "CovMatricesAll/c3_n%d_l%d_1,11_full.txt" % (nbin, max_l)), os.path.join(outdir, "CovMatricesAll/c4_n%d_l%d_11,11_full.txt" % (nbin, max_l))]
        all_output_names = []
        # Generate complete list of output names
        # First, find number of subsamples per file
        no_subsamples_per_file = 0
        while True:
            ith_output_names = [os.path.join(outdir, "0/CovMatricesAll/c2_n%d_l%d_11_%d.txt" % (nbin, max_l, no_subsamples_per_file)), os.path.join(outdir, "0/CovMatricesAll/c3_n%d_l%d_1,11_%d.txt" % (nbin, max_l, no_subsamples_per_file)), os.path.join(outdir, "0/CovMatricesAll/c4_n%d_l%d_11,11_%d.txt" % (nbin, max_l, no_subsamples_per_file))]
            if all(os.path.isfile(fname) for fname in ith_output_names):
                all_output_names += ith_output_names
                no_subsamples_per_file += 1
            else: break
        # no_subsamples_per_file should be now accurate for this tracer, redshift range and region
        if no_subsamples_per_file > 0: # i.e., have samples in per-file subdirectories
            # Second, find number of sequential files that have all the samples
            nfiles = 1 # there is at least one if the above succeeded
            while True:
                these_output_names = [os.path.join(outdir, "%d/CovMatricesAll/c2_n%d_l%d_11_%d.txt" % (nfiles, nbin, max_l, i)) for i in range(no_subsamples_per_file)] + [os.path.join(outdir, "%d/CovMatricesAll/c3_n%d_l%d_1,11_%d.txt" % (nfiles, nbin, max_l, i)) for i in range(no_subsamples_per_file)] + [ os.path.join(outdir, "%d/CovMatricesAll/c4_n%d_l%d_11,11_%d.txt" % (nfiles, nbin, max_l, i)) for i in range(no_subsamples_per_file)] # filenames for all npoints and subsample indices
                if all(os.path.isfile(fname) for fname in these_output_names):
                    all_output_names += these_output_names
                    nfiles += 1
                else: break
            # nfiles should be now accurate for this tracer, redshift range and region
            n_subsamples = no_subsamples_per_file * nfiles # set the number of subsamples which can be used straightforwardly
            # Full output depends on all output names. Use only one name for goal
            my_make(full_output_names[-1], all_output_names, f"python python/cat_subsets_of_integrals.py {nbin} l{max_l} " + " ".join([f"{os.path.join(outdir, str(i))} {no_subsamples_per_file}" for i in range(nfiles)]) + f" {outdir}")
            # Recipe: run subsample catenation
        else: # if no subsamples found in per-file subdirectories, try to detect in the root of directory
            n_subsamples = 0
            while True:
                ith_output_names = [os.path.join(outdir, "CovMatricesAll/c2_n%d_l%d_11_%d.txt" % (nbin, max_l, n_subsamples)), os.path.join(outdir, "CovMatricesAll/c3_n%d_l%d_1,11_%d.txt" % (nbin, max_l, n_subsamples)), os.path.join(outdir, "CovMatricesAll/c4_n%d_l%d_11,11_%d.txt" % (nbin, max_l, n_subsamples))]
                if all(os.path.isfile(fname) for fname in ith_output_names):
                    all_output_names += ith_output_names
                    n_subsamples += 1
                else: break
            # no_subsamples_per_file should be now accurate for this tracer, redshift range and region
            if n_subsamples > 0:
                # Full output depends on all output names. Use only one name for goal
                my_make(full_output_names[-1], all_output_names, f"python python/cat_subsets_of_integrals.py {nbin} l{max_l} {outdir} {n_subsamples} {outdir}")
                # Recipe: run subsample catenation, somewhat redundant in this case, but assures that the full outputs exist and are averages of existing samples
            else: continue # no samples detected => skip the tracer, can not do anything

        # Gaussian covariances

        results_name = os.path.join(outdir, 'Rescaled_Covariance_Matrices_Legendre_n%d_l%d.npz' % (nbin, max_l))
        reg_results.append(results_name)
        cov_name = "xi" + xilabel + "_" + "_".join(tlabels + [reg]) + f"_{z_min}_{z_max}_default_FKP_lin{r_step}_s{rmin_real}-{rmax}_cov_RascalC_Gaussian.txt"
        reg_pycorr_names.append(f"/global/cfs/cdirs/desi/survey/catalogs/Y1/mocks/SecondGenMocks/AbacusSummit/mock{0}/xi/smu/allcounts_{tracer}_{reg}_{z_min}_{z_max}_default_FKP_lin_njack{njack}_nran{nrandoms}_split{split_above}.npy")

        # RascalC results depend on full output (most straightforwardly)
        my_make(results_name, full_output_names, f"python python/post_process_legendre.py {outdir} {nbin} {max_l} {n_subsamples} {outdir} {1} {skip_bins} {skip_l}", f"python python/convergence_check_extra.py {results_name}")
        # Recipe: run post-processing
        # Also perform convergence check (optional but nice)

        # Individual cov file depends on RascalC results
        my_make(cov_name, [results_name], f"python python/convert_cov_legendre.py {results_name} {nbin_final} {cov_name}")
        # Recipe: run convert cov

        # Jackknife post-processing
        if jackknife:
            results_name_jack = os.path.join(outdir, 'Rescaled_Covariance_Matrices_Legendre_Jackknife_n%d_l%d_j%d.npz' % (nbin, max_l, njack))
            reg_results_jack.append(results_name_jack)
            xi_jack_name = os.path.join(outdir, f"xi_jack/xi_jack_n{nbin}_m{mbin}_j{njack}_11.dat")

            # RascalC results depend on full output (most straightforwardly)
            my_make(results_name_jack, full_output_names, f"python python/post_process_legendre_mix_jackknife.py {xi_jack_name} {os.path.join(outdir, 'weights')} {outdir} {mbin} {max_l} {n_subsamples} {outdir} {skip_bins} {skip_l}", f"python python/convergence_check_extra.py {results_name_jack}")
            # Recipe: run post-processing
            # Also perform convergence check (optional but nice)

            # Load shot-noise rescaling and make name
            cov_name_jack = "xi" + xilabel + "_" + "_".join(tlabels + [reg]) + f"_{z_min}_{z_max}_default_FKP_lin{r_step}_s{rmin_real}-{rmax}_cov_RascalC_rescaled.txt"
            # Individual cov file depends on RascalC results
            my_make(cov_name_jack, [results_name_jack], f"python python/convert_cov_legendre.py {results_name_jack} {nbin_final} {cov_name_jack}")
            # Recipe: run convert cov

            # Here is a special case where the goal name could change (with shot-noise rescaling), so let us delete alternative versions from the directory and the hash dictionary if any
            # Change of filename does not break the general make logic – the same jack results file must yield the same shot-noise rescaling anyway
            cov_name_jack_pattern = "xi" + xilabel + "_" + "_".join(tlabels + [reg]) + f"_{z_min}_{z_max}_default_FKP_lin{r_step}_s{rmin_real}-{rmax}_cov_RascalC_rescaled*.txt"
            # Filenames
            for fname in glob(cov_name_jack_pattern): # all existing files matching the pattern
                if not os.path.samefile(fname, cov_name_jack): os.remove(fname) # if not our result file, delete it
            # Hash dictionary keys (goal names) - could be independent
            for key in fnmatch.filter(hash_dict.keys(), cov_name_jack_pattern): # all hash dictionary keys matching the pattern
                if key != cov_name_jack: hash_dict.pop(key) # if not our goal name, remove the key (and its value)

    if len(reg_pycorr_names) == len(regs): # if we have pycorr files for all regions
        if len(reg_results) == len(regs): # if we have RascalC results for all regions
            # Combined Gaussian cov

            cov_name = "xi" + xilabel + "_" + "_".join(tlabels + [reg_comb]) + f"_{z_min}_{z_max}_default_FKP_lin{r_step}_s{rmin_real}-{rmax}_cov_RascalC_Gaussian.txt" # combined cov name

            # Comb cov depends on the region RascalC results
            my_make(cov_name, reg_results, "python python/combine_covs_legendre.py " + " ".join(reg_results) + " " + " ".join(reg_pycorr_names) + f" {nbin} {max_l} {skip_bins} {cov_name}")
            # Recipe: run combine covs

        if jackknife and len(reg_results_jack) == len(regs): # if jackknife and we have RascalC jack results for all regions
            # Combined rescaled cov
            cov_name_jack = "xi" + xilabel + "_" + "_".join(tlabels + [reg_comb]) + f"_{z_min}_{z_max}_default_FKP_lin{r_step}_s{rmin_real}-{rmax}_cov_RascalC_rescaled.txt" # combined cov name

            # Comb cov depends on the region RascalC results
            my_make(cov_name_jack, reg_results_jack, "python python/combine_covs_legendre.py " + " ".join(reg_results_jack) + " " + " ".join(reg_pycorr_names) + f" {nbin} {max_l} {skip_bins} {cov_name_jack}")
            # Recipe: run combine covs

# Save the updated hash dictionary
with open(hash_dict_file, "wb") as f:
    pickle.dump(hash_dict, f)

print_and_log(datetime.now())
print_and_log("Finished execution.")
