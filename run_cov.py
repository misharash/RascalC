### Python script for running RascalC in DESI setup (Michael Rashkovetskyi, 2022).
# It is important to check the log file in the output directory to make sure all went well

import os, sys
from datetime import datetime
import numpy as np

def check_path(filename: str, fallback_dir: str | None = None) -> str:
    if fallback_dir is not None:
        if os.path.isfile(filename): return filename
        filename = os.path.join(fallback_dir, os.path.basename(filename))
    assert os.path.isfile(filename), f"{filename} missing"
    return filename

def prevent_override(filename: str, max_num: int = 10) -> str: # append _{number} to filename to prevent override
    for i in range(max_num+1):
        trial_name = filename + ("_" + str(i)) * bool(i) # will be filename for i=0
        if not os.path.exists(trial_name): return trial_name
    print(f"Could not prevent override of {filename}, aborting.")
    sys.exit(1)

##################### INPUT PARAMETERS ###################

terminate_on_error = 1 # whether to terminate if any of executed scripts returns nonzero

ntracers = 1 # number of tracers
if ntracers > 1:
    cycle_randoms = 1
periodic = 0 # whether to run with periodic boundary conditions (must also be set in Makefile)
make_randoms = 0 # whether to generate randoms, only works in periodic case (cubic box)
jackknife = 1 # whether to compute jackknife integrals (must also be set in Makefile)
njack = 60 if jackknife else 0 # number of jackknife regions; if jackknife flag is not set this is used for the pycorr filenames and should be 0
legendre_orig = 0 # original Legendre mode - when each pair's contribution is accumulated to multipoles of 2PCF directly
legendre_mix = 1 # mixed Legendre mode - when the s,mu-binned 2PCF is estimated and then projected into Legendre multipoles using integrals of Legendre polynomials 
legendre = legendre_orig or legendre_mix # any Legendre
if legendre:
    max_l = 4

assert ntracers in (1, 2), "Only single- and two-tracer modes are currently supported"
assert not (make_randoms and jackknife), "Jackknives with generated randoms not implemented"
assert not (make_randoms and not periodic), "Non-periodic random generation not supported"
assert not (jackknife and legendre_orig), "Jackknife and original Legendre modes are incompatible"

nthread = 256 # number of OMP threads to use

version_label = "v0.6"

id = int(sys.argv[1]) # SLURM_JOB_ID to decide what this one has to do
reg = "NGC" if id%2 else "SGC" # region for filenames

id //= 2 # extracted all needed info from parity, move on
tracers = ['LRG'] * 4 + ['ELG_LOPnotqso'] * 3 + ['BGS_BRIGHT-21.5', 'QSO']
zs = [[0.4, 0.6], [0.6, 0.8], [0.8, 1.1], [0.4, 1.1], [0.8, 1.1], [1.1, 1.6], [0.8, 1.6], [0.1, 0.4], [0.8, 2.1]]
# need 2 * 9 = 18 jobs in this array

tlabels = [tracers[id]] # tracer labels for filenames
assert len(tlabels) == ntracers, "Need label for each tracer"
nrandoms = 1 if tlabels[0].startswith("BGS") else 4 # 1 random for BGS only
if any(tlabels[0].startswith(t) for t in ("BGS", "LRG")): version_label = "v0.6.1" # newer version for BGS and LRG, older for ELG and QSO

# data processing steps
redshift_cut = 1
convert_to_xyz = 1
if redshift_cut or convert_to_xyz:
    # the following options are set for each tracer, possibly differently. Make sure that all the counts are compatible with the selected weighting and selection.
    use_weights = [1] * ntracers # For FITS files: 0 - do not use the WEIGHT column even if present. 1 - use WEIGHT column if present. Has no effect with plain text files
    FKP_weights = [1] * ntracers # For FITS files: 0 - do not use FKP weights. 1 - load them from WEIGHT_FKP column. "P0,NZ_name" - compute manually with given P0 and NZ from column "NZ_name". Has no effect with plain text files.
    masks = [0] * ntracers # default, basically no mask. All bits set to 1 in the mask have to be set in the FITS data STATUS. Does nothing with plain text files.
create_jackknives = jackknife and 1
normalize_weights = 1 # rescale weights in each catalog so that their sum is 1. Will also use normalized RR counts from pycorr
cat_randoms = 1 # concatenate random files for RascalC input
if cat_randoms:
    cat_randoms_files = [f"{tlabel}_{reg}_0-{nrandoms-1}_clustering.ran.xyzw" + ("j" if jackknife else "") for tlabel in tlabels]

z_min, z_max = zs[id] # for redshift cut and filenames

input_dir = f"/global/cfs/cdirs/desi/users/uendert/desi_blinding/LSScats/{version_label}/doubleblinded/"

# cosmology
if convert_to_xyz:
    Omega_m = 0.31519
    Omega_k = 0
    w_dark_energy = -1

# File names and directories
if jackknife:
    data_ref_filenames = [check_path(input_dir + f"{tlabel}_{reg}_clustering.dat.fits") for tlabel in tlabels] # only for jackknife reference or ndata backup, has to have rdz contents
    assert len(data_ref_filenames) == ntracers, "Need reference data for all tracers"
input_filenames = [[check_path(input_dir + f"{tlabel}_{reg}_{i}_clustering.ran.fits") for i in range(nrandoms)] for tlabel in tlabels] # random filenames
assert len(input_filenames) == ntracers, "Need randoms for all tracers"
nfiles = [len(input_filenames_group) for input_filenames_group in input_filenames]
if not cat_randoms or make_randoms:
    for i in range(1, ntracers):
        assert nfiles[i] == nfiles[0], "Need to have the same number of files for all tracers"
outdir = prevent_override("_".join(tlabels) + "_" + reg + f"_z{z_min}-{z_max}") # output file directory
tmpdir = os.path.join("tmpdirs", outdir) # directory to write intermediate files, mainly data processing steps

if cat_randoms: # move concatenated randoms file to tmpdir as well
    cat_randoms_files = [os.path.join(tmpdir, cat_randoms_file) for cat_randoms_file in cat_randoms_files]

##########################################################

# Create intermediate directory
os.makedirs(tmpdir, exist_ok=1)

# Create output directory
os.makedirs(outdir, exist_ok=1)

# Create an output file for errors
logfilename = "log.txt"
logfile = os.path.join(outdir, logfilename)

# Copy this script in for posterity
os.system(f"cp {__file__} {os.path.normpath(outdir)}")

def print_and_log(s: str) -> None:
    print(s)
    print_log(s)
print_log = lambda l: os.system(f"echo \"{l}\" >> {logfile}")

print_and_log(datetime.now())
print_and_log(f"Executing {__file__}")

def exec_print_and_log(commandline: str) -> None:
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
            print("Terminating the running script execution due to this error.")
            sys.exit(1)

print("Starting Computation")

def change_extension(name: str, ext: str) -> str:
    return os.path.join(tmpdir, os.path.basename(".".join(name.split(".")[:-1] + [ext]))) # change extension and switch to tmpdir

def append_to_filename(name: str, appendage: str) -> str:
    return os.path.join(tmpdir, os.path.basename(name + appendage)) # append part and switch to tmpdir

if create_jackknives and redshift_cut: # prepare reference file
    for t, data_ref_filename in enumerate(data_ref_filenames):
        if create_jackknives:
            print_and_log("Processing data file for jackknife reference")
            rdzw_ref_filename = change_extension(data_ref_filename, "rdzw")
            from python.redshift_cut import redshift_cut_files
            redshift_cut_files(data_ref_filename, rdzw_ref_filename, z_min, z_max, FKP_weights[t], masks[t], use_weights[t], print_and_log)
            data_ref_filenames[t] = rdzw_ref_filename

# processing steps for each random file
for t, (input_filenames_t, nfiles_t) in enumerate(zip(input_filenames, nfiles)):
    print_and_log(f"Starting preparing tracer {t+1} of {ntracers}")
    for i, input_filename in enumerate(input_filenames_t):
        print_and_log(f"Starting preparing file {i+1} of {nfiles_t}")
        print_and_log(datetime.now())
        if periodic and make_randoms: # just save randoms to this file
            pass
        else: # (potentially) run through all data processing steps
            if redshift_cut:
                rdzw_filename = change_extension(input_filename, "rdzw")
                from python.redshift_cut import redshift_cut_files
                redshift_cut_files(input_filename, rdzw_filename, z_min, z_max, FKP_weights[t], masks[t], use_weights[t], print_and_log)
                input_filename = rdzw_filename
            if convert_to_xyz:
                xyzw_filename = change_extension(input_filename, "xyzw")
                from python.convert_to_xyz import convert_to_xyz_files
                convert_to_xyz_files(input_filename, xyzw_filename, Omega_m, Omega_k, w_dark_energy, FKP_weights[t], masks[t], use_weights[t], print_and_log)
                input_filename = xyzw_filename
            if create_jackknives:
                xyzwj_filename = change_extension(input_filename, "xyzwj")
                from python.create_jackknives_pycorr import create_jackknives_pycorr_files
                create_jackknives_pycorr_files(data_ref_filenames[t], input_filename, xyzwj_filename, njack, print_and_log) # keep in mind some subtleties for multi-tracer jackknife assigment
                input_filename = xyzwj_filename
        input_filenames[t][i] = input_filename # save final input filename for next loop
        print_and_log(f"Finished preparing file {i+1} of {nfiles_t}")
# end processing steps for each random file

if cat_randoms: # concatenate randoms
    for t in range(ntracers):
        if nfiles[t] > 1: # real action is needed
            print_and_log(datetime.now())
            exec_print_and_log(f"cat {' '.join(input_filenames[t])} > {cat_randoms_files[t]}")
            input_filenames[t] = [cat_randoms_files[t]] # now it is the only file
        else: # skip actual concatenation, just reuse the only input file
            cat_randoms_files[t] = input_filenames[t][0]
    nfiles = 1
else:
    nfiles = nfiles[0]
    if ntracers > 1 and nfiles > 1 and cycle_randoms:
        for t in range(1, ntracers):
            input_filenames[t] = input_filenames[t][t*cycle_randoms:] + input_filenames[t][:t*cycle_randoms] # shift the filename list cyclically by number of tracer, this makes sure files with different numbers for different tracers are fed to the C++ code, otherwise overlapping positions are likely at least between LRG and ELG
    # now the number of files to process is the same for sure

# most sensible to normalize weights after concatenation but before counts computation and main code run
if normalize_weights:
    for t, input_filenames_t in enumerate(input_filenames):
        print_and_log(f"Normalizing weights for tracer {t+1} of {ntracers}")
        for i, input_filename in enumerate(input_filenames_t):
            print_and_log(f"Starting normalizing weights in file {i+1} of {nfiles}")
            print_and_log(datetime.now())
            n_filename = append_to_filename(input_filename, "n") # append letter n to the original filename
            from python.normalize_weights import normalize_weights_files
            normalize_weights_files(input_filename, n_filename, print_and_log)
            input_filenames[t][i] = n_filename # update input filename for later
            print_and_log(f"Finished normalizing weights in file {i+1} of {nfiles}")

# runthe test code
exec_print_and_log(f"./demo {nthread} {10**8}")

print_and_log(datetime.now())
print_and_log(f"Finished execution.")
