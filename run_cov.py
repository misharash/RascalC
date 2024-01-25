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

ntracers = 2 # number of tracers
if ntracers > 1:
    cycle_randoms = 1
periodic = 0 # whether to run with periodic boundary conditions (must also be set in Makefile)
make_randoms = 0 # whether to generate randoms and how many times the number of data points per file, only works in periodic case (cubic box)
jackknife = 0 # whether to compute jackknife integrals (must also be set in Makefile)
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

ndata = [None] * ntracers # number of data points for each tracer; set None to make sure it is overwritten before any usage and see an error otherwise
count_ndata = 1 # whether to count data galaxies if can't load useful info from pycorr

rmin = 0 # minimum output cov radius in Mpc/h
rmax = 200 # maximum output cov radius in Mpc/h
nbin = 50 # radial bins for output cov
mbin = 100 # angular (mu) bins for output cov; in original Legendre mode number of bins for correction function; in mixed Legendre mode the number of bins of the intermediate s,mu correlation function projected into multipoles
rmin_cf = 0 # minimum input 2PCF radius in Mpc/h
rmax_cf = 200 # maximum input 2PCF radius in Mpc/h
nbin_cf = 100 # radial bins for input 2PCF
mbin_cf = 10 # angular (mu) bins for input 2PCF
xicutoff = 250 # beyond this assume xi/2PCF=0

nthread = 256 # number of OMP threads to use
maxloops = 1024 # number of integration loops per filename
loopspersample = 64 # number of loops to collapse into one subsample
N2 = 5 # number of secondary cells/particles per primary cell
N3 = 10 # number of third cells/particles per secondary cell/particle
N4 = 20 # number of fourth cells/particles per third cell/particle

rescale = 1 # rescaling for co-ordinates
nside = 301 # grid size for accelerating pair count
boxsize = 2000 # only used if periodic=1

suffixes_tracer_all = ("", "2") # all supported tracer suffixes
suffixes_tracer = suffixes_tracer_all[:ntracers]
indices_corr_all = ("11", "12", "22") # all supported 2PCF indices
suffixes_corr_all = ("", "12", "2") # all supported 2PCF suffixes
tracer1_corr_all = (0, 0, 1)
tracer2_corr_all = (0, 1, 1)
ncorr = ntracers*(ntracers+1)//2 # number of correlation functions
indices_corr = indices_corr_all[:ncorr] # indices to use
suffixes_corr = suffixes_corr_all[:ncorr] # indices to use
tracer1_corr, tracer2_corr = tracer1_corr_all[:ncorr], tracer2_corr_all[:ncorr]

version_label = "v0.6"

id = int(sys.argv[1]) # SLURM_JOB_ID to decide what this one has to do
reg = "NGC" if id%2 else "SGC" # region for filenames
# need 2 jobs in this array

tlabels = ['LRG', 'ELG_LOPnotqso'] # tracer labels for filenames
corlabels = [tlabels[0], "_".join(tlabels), tlabels[1]]
assert len(tlabels) == ntracers, "Need label for each tracer"
nrandoms = 4

rectype = "IFTrecsym" # reconstruction type
sm = 15 # smoothing scale

assert maxloops % loopspersample == 0, "Group size need to divide the number of loops"
# no_subsamples_per_file = maxloops // loopspersample

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
do_counts = 0 # (re)compute total pair counts, jackknife weights/xi with RascalC script, on concatenated randoms, instead of reusing them from pycorr
cat_randoms = 1 # concatenate random files for RascalC input
if do_counts or cat_randoms:
    cat_randoms_files = [f"{tlabel}_{reg}_0-{nrandoms-1}_clustering.ran.xyzw" + ("j" if jackknife else "") for tlabel in tlabels]

z_min, z_max = 0.8, 1.1 # for redshift cut and filenames

input_dir = f"/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/{version_label}/blinded/recon_sm{sm}/"

# CF options
convert_cf = 1
if convert_cf:
    # first index is correlation function index
    counts_factor = 0 if normalize_weights else nrandoms if not cat_randoms else 1 # 0 is a special value for normalized counts; use number of randoms if they are not concatenated, otherwise 1
    split_above = np.inf
    pycorr_filenames = [[check_path(f"/global/cfs/cdirs/desi/users/dvalcin/EZMOCKS/Overlap/Y1/FOR_MISHA/{version_label}/recon_sm{sm}/allcounts_{corlabel}_{rectype}_{reg}_{z_min}_{z_max}_default_FKP_lin_njack{njack}_nran{nrandoms}.npy")] for corlabel in corlabels]
    assert len(pycorr_filenames) == ncorr, "Expected pycorr file(s) for each correlation"
smoothen_cf = 0
if smoothen_cf:
    max_l_smoothing = 4
    radial_window_len = 5
    radial_polyorder = 2

# cosmology
if convert_to_xyz:
    Omega_m = 0.31519
    Omega_k = 0
    w_dark_energy = -1

# File names and directories
if jackknife or count_ndata:
    data_ref_filenames = [check_path(input_dir + f"{tlabel}_{reg}_clustering.{rectype}.dat.fits") for tlabel in tlabels] # only for jackknife reference or ndata backup, has to have rdz contents
    assert len(data_ref_filenames) == ntracers, "Need reference data for all tracers"
input_filenames = [[check_path(input_dir + f"{tlabel}_{reg}_{i}_clustering.{rectype}.ran.fits") for i in range(nrandoms)] for tlabel in tlabels] # random filenames
assert len(input_filenames) == ntracers, "Need randoms for all tracers"
nfiles = [len(input_filenames_group) for input_filenames_group in input_filenames]
if not cat_randoms or make_randoms:
    for i in range(1, ntracers):
        assert nfiles[i] == nfiles[0], "Need to have the same number of files for all tracers"
outdir = prevent_override(f"recon_sm{sm}/" + "_".join(tlabels + [rectype, reg]) + f"_z{z_min}-{z_max}") # output file directory
tmpdir = os.path.join("tmpdirs", outdir) # directory to write intermediate files, mainly data processing steps
cornames = [os.path.join(outdir, f"xi/xi_n{nbin_cf}_m{mbin_cf}_{index}.dat") for index in indices_corr]
binned_pair_names = [os.path.join(outdir, "weights/" + ("binned_pair" if jackknife else "RR") + f"_counts_n{nbin}_m{mbin}" + (f"_j{njack}" if jackknife else "") + f"_{index}.dat") for index in indices_corr]
if jackknife:
    jackknife_weights_names = [os.path.join(outdir, f"weights/jackknife_weights_n{nbin}_m{mbin}_j{njack}_{index}.dat") for index in indices_corr]
    if convert_cf:
        xi_jack_names = [os.path.join(outdir, f"xi_jack/xi_jack_n{nbin}_m{mbin}_j{njack}_{index}.dat") for index in indices_corr]
        jackknife_pairs_names = [os.path.join(outdir, f"weights/jackknife_pair_counts_n{nbin}_m{mbin}_j{njack}_{index}.dat") for index in indices_corr]
if legendre_orig:
    phi_names = [f"BinCorrectionFactor_n{nbin}_" + ("periodic" if periodic else f'm{mbin}') + f"_{index}.txt" for index in indices_corr]

if do_counts or cat_randoms: # move concatenated randoms file to tmpdir as well
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

# binning files to be created automatically
binfile = os.path.join(outdir, "radial_binning_cov.csv")
binfile_cf = os.path.join(outdir, "radial_binning_corr.csv")
from RascalC.write_binning_file import write_binning_file_linear
write_binning_file_linear(binfile, rmin, rmax, nbin, print_and_log)
write_binning_file_linear(binfile_cf, rmin_cf, rmax_cf, nbin_cf, print_and_log)

if legendre_mix: # write mu bin Legendre factors for the code
    from RascalC.mu_bin_legendre_factors import write_mu_bin_legendre_factors
    mu_bin_legendre_file = write_mu_bin_legendre_factors(mbin, max_l, os.path.dirname(binned_pair_names[0]))

# full-survey CF conversion, will also load number of data points from pycorr
if convert_cf:
    r_step_cf = (rmax_cf - rmin_cf) / nbin_cf
    for c, corname in enumerate(cornames):
        os.makedirs(os.path.dirname(corname), exist_ok=1) # make sure all dirs exist
        from RascalC.pycorr_utils.input_xi import convert_xi_from_pycorr_files
        _, ndata2 = convert_xi_from_pycorr_files(pycorr_filenames[c], corname, n_mu = mbin_cf, r_step = r_step_cf, print_function = print_and_log)
        ndata[tracer2_corr[c]] = ndata2 # override ndata for second tracer, so that autocorrelations are prioritized
        if smoothen_cf:
            corname_old = corname
            corname = f"xi/xi_n{nbin_cf}_m{mbin_cf}_{indices_corr[c]}_smooth.dat"
            from RascalC.xi.smoothen import smoothen_xi_files
            smoothen_xi_files(corname_old, max_l_smoothing, radial_window_len, radial_polyorder, corname)
            cornames[c] = corname # save outside of the loop

ndata_is_bad = [ndata_i is None or not np.isfinite(ndata_i) or ndata_i <= 0 for ndata_i in ndata]
if count_ndata: count_ndata = any(ndata_is_bad) # no need to count data if all ndata are good
elif any(ndata_is_bad):
    print(f"One of normalizations ({ndata}) is not a positive number. Can not proceed.")
    sys.exit(1)

if periodic and make_randoms:
    # create random points
    print_and_log(f"Generating random points")
    np.random.seed(42) # for reproducibility
    randoms = [np.append(np.random.rand(nfiles_t, int(make_randoms * ndata_t), 3) * boxsize, np.ones((nfiles_t, int(make_randoms * ndata_t), 1)), axis=-1) for nfiles_t, ndata_t in zip(nfiles, ndata)]
    # 3 columns of random coordinates within [0, boxsize] and one of weights, all equal to unity. List of array; list index is tracer number, first array index is file number and the second is number of point. Keep number of points roughly equal to number of data for each tracer
    print_and_log(f"Generated random points")

def change_extension(name: str, ext: str) -> str:
    return os.path.join(tmpdir, os.path.basename(".".join(name.split(".")[:-1] + [ext]))) # change extension and switch to tmpdir

def append_to_filename(name: str, appendage: str) -> str:
    return os.path.join(tmpdir, os.path.basename(name + appendage)) # append part and switch to tmpdir

if (create_jackknives or count_ndata) and redshift_cut: # prepare reference file
    for t, data_ref_filename in enumerate(data_ref_filenames):
        if create_jackknives or ndata_is_bad[t]:
            print_and_log("Processing data file for" + create_jackknives * " jackknife reference" + (create_jackknives and count_ndata) * " and" + count_ndata * " galaxy counts")
            rdzw_ref_filename = change_extension(data_ref_filename, "rdzw")
            from RascalC.pre_process.redshift_cut import redshift_cut_files
            redshift_cut_files(data_ref_filename, rdzw_ref_filename, z_min, z_max, FKP_weights[t], masks[t], use_weights[t], print_and_log)
            data_ref_filenames[t] = rdzw_ref_filename
        if ndata_is_bad[t]:
            with open(data_ref_filenames[t]) as f:
                for lineno, _ in enumerate(f):
                    pass
                ndata[t] = lineno + 1

command = f"./cov -boxsize {boxsize} -nside {nside} -rescale {rescale} -nthread {nthread} -maxloops {maxloops} -loopspersample {loopspersample} -N2 {N2} -N3 {N3} -N4 {N4} -xicut {xicutoff} -binfile {binfile} -binfile_cf {binfile_cf} -mbin_cf {mbin_cf}" # here are universally acceptable parameters
command += "".join([f" -norm{suffixes_tracer[t]} {ndata[t]}" for t in range(ntracers)]) # provide all ndata for normalization
command += "".join([f" -cor{suffixes_corr[c]} {cornames[c]}" for c in range(ncorr)]) # provide all correlation functions
if legendre: # only provide max multipole l for now
    command += f" -max_l {max_l}"
if legendre_mix: # provide factors filename
    command += f" -mu_bin_legendre_file {mu_bin_legendre_file}"
if not legendre_orig: # provide binned pair counts files and number of mu bin
    command += "".join([f" -RRbin{suffixes_corr[c]} {binned_pair_names[c]}" for c in range(ncorr)]) + f" -mbin {mbin}"
if periodic: # append periodic flag
    command += " -perbox"
if jackknife: # provide jackknife weight files for all correlations
    command += "".join([f" -jackknife{suffixes_corr[c]} {jackknife_weights_names[c]}" for c in range(ncorr)])
command = "env OMP_PROC_BIND=spread OMP_PLACES=threads " + command # set OMP environment variables, they shall not be set for this Python script (IMPORTANT)
print_and_log(f"Common command for C++ code: {command}")

# processing steps for each random file
for t, (input_filenames_t, nfiles_t) in enumerate(zip(input_filenames, nfiles)):
    print_and_log(f"Starting preparing tracer {t+1} of {ntracers}")
    for i, input_filename in enumerate(input_filenames_t):
        print_and_log(f"Starting preparing file {i+1} of {nfiles_t}")
        print_and_log(datetime.now())
        if periodic and make_randoms: # just save randoms to this file
            input_filename = change_extension(input_filename, "xyzw")
            np.savetxt(input_filename, randoms[t][i])
        else: # (potentially) run through all data processing steps
            if redshift_cut:
                rdzw_filename = change_extension(input_filename, "rdzw")
                from RascalC.pre_process.redshift_cut import redshift_cut_files
                redshift_cut_files(input_filename, rdzw_filename, z_min, z_max, FKP_weights[t], masks[t], use_weights[t], print_and_log)
                input_filename = rdzw_filename
            if convert_to_xyz:
                xyzw_filename = change_extension(input_filename, "xyzw")
                from RascalC.pre_process.convert_to_xyz import convert_to_xyz_files
                convert_to_xyz_files(input_filename, xyzw_filename, Omega_m, Omega_k, w_dark_energy, FKP_weights[t], masks[t], use_weights[t], print_and_log)
                input_filename = xyzw_filename
            if create_jackknives:
                xyzwj_filename = change_extension(input_filename, "xyzwj")
                from RascalC.pre_process.create_jackknives_pycorr import create_jackknives_pycorr_files
                create_jackknives_pycorr_files(data_ref_filenames[t], input_filename, xyzwj_filename, njack, print_and_log) # keep in mind some subtleties for multi-tracer jackknife assigment
                input_filename = xyzwj_filename
        input_filenames[t][i] = input_filename # save final input filename for next loop
        print_and_log(f"Finished preparing file {i+1} of {nfiles_t}")
# end processing steps for each random file

if cat_randoms: # concatenate randoms
    for t in range(ntracers):
        if nfiles[t] > 1: # real action is needed
            print_and_log(datetime.now())
            exec_print_and_log(f"cat {' '.join(input_filenames[t])} > {cat_randoms_files[t]}") # should not be multi-threaded
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
            from RascalC.pre_process.normalize_weights import normalize_weights_files
            normalize_weights_files(input_filename, n_filename, print_and_log)
            input_filenames[t][i] = n_filename # update input filename for later
            print_and_log(f"Finished normalizing weights in file {i+1} of {nfiles}")

if convert_cf: # this is really for pair counts and jackknives
    print_and_log(datetime.now())
    if do_counts: # redo counts
        if jackknife: # do jackknife xi and all counts
            if nfiles > 1: # concatenate randoms now if needed
                for t in range(ntracers):
                    exec_print_and_log(f"cat {' '.join(input_filenames[t])} > {cat_randoms_files[t]}") # should not be multi-threaded
            else:
                cat_randoms_files[t] = input_filenames[t][0]
            # compute jackknife weights
            if ntracers == 1:
                exec_print_and_log(f"python scripts/legacy/jackknife_weights.py {cat_randoms_files[0]} {binfile} 1. {mbin} {nthread} {periodic} {os.path.dirname(jackknife_weights_names[0])}/") # 1. is max mu
            elif ntracers == 2:
                exec_print_and_log(f"python scripts/legacy/jackknife_weights_cross.py {' '.join(cat_randoms_files)} {binfile} 1. {mbin} {nthread} {periodic} {os.path.dirname(jackknife_weights_names[0])}/") # 1. is max mu
            else:
                print("Number of tracers not supported for this operation (yet)")
                sys.exit(1)
            # continue processing of data files - from redshift-cut rdzw to xyzw and xyzwj
            for t in range(ntracers):
                data_filename = data_ref_filenames[t]
                xyzw_filename = change_extension(data_filename, "xyzw")
                from RascalC.pre_process.convert_to_xyz import convert_to_xyz_files
                convert_to_xyz_files(data_filename, xyzw_filename, Omega_m, Omega_k, w_dark_energy, FKP_weights[t], masks[t], use_weights[t], print_and_log)
                data_filename = xyzw_filename
                xyzwj_filename = change_extension(data_filename, "xyzwj")
                # keep in mind some subtleties for multi-tracer jackknife assigment
                from RascalC.pre_process.create_jackknives_pycorr import create_jackknives_pycorr_files
                create_jackknives_pycorr_files(data_ref_filenames[t], data_filename, xyzwj_filename, njack, print_and_log) # the first file must be rdzw, the second xyzw!
                data_filename = xyzwj_filename
                if normalize_weights:
                    n_filename = append_to_filename(data_filename, "n") # append letter n to the original filename
                    from RascalC.pre_process.normalize_weights import normalize_weights_files
                    normalize_weights_files(data_filename, n_filename, print_and_log)
                    data_filename = n_filename
                data_ref_filenames[t] = data_filename # update the name in list
            # run RascalC own xi jack estimator
            if ntracers == 1:
                exec_print_and_log(f"python scripts/legacy/xi_estimator_jack.py {data_ref_filenames[0]} {cat_randoms_files[0]} {cat_randoms_files[0]} {binfile} 1. {mbin} {nthread} {periodic} {os.path.dirname(xi_jack_names[0])}/ {jackknife_pairs_names[0]}") # 1. is max mu
            elif ntracers == 2:
                exec_print_and_log(f"python scripts/legacy/xi_estimator_jack_cross.py {' '.join(data_ref_filenames)} {' '.join(cat_randoms_files)} {' '.join(cat_randoms_files)} {binfile} 1. {mbin} {nthread} {periodic} {os.path.dirname(xi_jack_names[0])}/ {' '.join(jackknife_pairs_names)}") # 1. is max mu
            else:
                print("Number of tracers not supported for this operation (yet)")
                sys.exit(1)
            if not cat_randoms: # reload full counts from pycorr, override jackknives - to prevent normalization issues
                r_step = (rmax - rmin) / nbin
                from RascalC.pycorr_utils.counts import convert_counts_from_pycorr_files
                for c in range(ncorr):
                    convert_counts_from_pycorr_files(pycorr_filenames[c][0], binned_pair_names[c], n_mu = mbin, r_step =  r_step, r_max = rmax, counts_factor = counts_factor, split_above = split_above)
        else: # only need full, binned pair counts
            if cat_randoms: # compute counts with our own script
                if ntracers == 1:
                    exec_print_and_log(f"python scripts/legacy/RR_counts.py {cat_randoms_files[0]} {binfile} 1. {mbin} {nthread} {periodic} {os.path.dirname(binned_pair_names[0])}/ 0") # 1. is max mu, 0 means not normed
                elif ntracers == 2:
                    exec_print_and_log(f"python scripts/legacy/RR_counts_multi.py {' '.join(cat_randoms_files)} {binfile} 1. {mbin} {nthread} {periodic} {os.path.dirname(binned_pair_names[0])}/ 0") # 1. is max mu, 0 means not normed
                else:
                    print("Number of tracers not supported for this operation (yet)")
                    sys.exit(1)
            else:
                print("Non-jackknife computation with not concatenated randoms not implemented yet")
                sys.exit(1)
    else: # convert from pycorr
        for c in range(ncorr):
            os.makedirs(os.path.dirname(binned_pair_names[c]), exist_ok=1) # make sure all dirs exist
            r_step = (rmax-rmin)//nbin
            if jackknife: # convert jackknife xi and all counts
                for filename in (xi_jack_names[c], jackknife_weights_names[c], jackknife_pairs_names[c]):
                    os.makedirs(os.path.dirname(filename), exist_ok=1) # make sure all dirs exist
                from RascalC.pycorr_utils.jack import convert_jack_xi_weights_counts_from_pycorr_files
                convert_jack_xi_weights_counts_from_pycorr_files(pycorr_filenames[c][0], xi_jack_names[c], jackknife_weights_names[c], jackknife_pairs_names[c], binned_pair_names[c], n_mu = mbin, r_step = r_step, r_max = rmax, counts_factor = counts_factor, split_above = split_above)
            elif not (periodic and legendre_orig): # convert full, binned pair counts, unless original Legendre and periodic when counts are not needed
                from RascalC.pycorr_utils.counts import convert_counts_from_pycorr_files
                convert_counts_from_pycorr_files(pycorr_filenames[c][0], binned_pair_names[c], n_mu = mbin, r_step = r_step, r_max = rmax, counts_factor = counts_factor, split_above = split_above)

# running main code for each random file/part
for i in range(nfiles):
    print_and_log(f"Starting main computation {i+1} of {nfiles}")
    print_and_log(datetime.now())
    # define output subdirectory
    this_outdir = os.path.join(outdir, str(i)) if nfiles > 1 else outdir # create output subdirectory only if processing multiple files
    this_outdir = os.path.normpath(this_outdir) + "/" # make sure there is exactly one slash in the end
    if legendre_orig: # need correction function
        os.makedirs(this_outdir, exist_ok=1)
        if ntracers == 1:
            from RascalC.correction_function import compute_correction_function
            compute_correction_function(input_filenames[0][i], binfile, this_outdir, periodic, binned_pair_names[0], print_and_log)
        elif ntracers == 2:
            from RascalC.correction_function import compute_correction_function_multi
            compute_correction_function_multi(input_filenames[0][i], input_filenames[1][i], binfile, this_outdir, periodic, *binned_pair_names, print_function = print_and_log)
        else:
            print("Number of tracers not supported for this operation (yet)")
            sys.exit(1)
    # run code
    exec_print_and_log(command + "".join([f" -in{suffixes_tracer[t]} {input_filenames[t][i]}" for t in range(ntracers)]) + f" -output {this_outdir}" + ("".join([f" -phi_file{suffixes_corr[c]} {os.path.join(this_outdir, phi_names[c])}" for c in range(ncorr)]) if legendre_orig else ""))
    print_and_log(f"Finished main computation {i+1} of {nfiles}")
# end running main code for each random file/part

print_and_log(datetime.now())

# Concatenate samples
if nfiles > 1:
    print_and_log("Concatenating samples")
    from RascalC.raw_covariance_matrices import cat_raw_covariance_matrices
    cat_raw_covariance_matrices(nbin, f'l{max_l}' if legendre else f'm{mbin}', [os.path.join(outdir, str(i)) for i in range(nfiles)], [None] * nfiles, outdir, print_function = print_and_log)
    print_and_log(datetime.now())

# Post-process
print_and_log("Post-processing")
# Parameters
skip_r_bins = 5
if not jackknife:
    shot_noise_rescaling = 1
    if ntracers >= 2: shot_noise_rescaling2 = 1
if legendre:
    skip_l = 0

# n_subsamples = no_subsamples_per_file * nfiles
if ntracers == 1:
    if legendre:
        if jackknife:
            from RascalC.post_process.legendre_mix_jackknife import post_process_legendre_mix_jackknife
            results = post_process_legendre_mix_jackknife(xi_jack_names[0], os.path.dirname(jackknife_weights_names[0]), outdir, mbin, max_l, outdir, skip_r_bins, skip_l, print_function = print_and_log)
        else:
            from RascalC.post_process.legendre import post_process_legendre
            results = post_process_legendre(outdir, nbin, max_l, outdir, shot_noise_rescaling, skip_r_bins, skip_l, print_function = print_and_log)
    elif jackknife:
        from RascalC.post_process.jackknife import post_process_jackknife
        results = post_process_jackknife(xi_jack_names[0], os.path.dirname(jackknife_weights_names[0]), outdir, mbin, outdir, skip_r_bins, print_function = print_and_log)
    else: # default
        from RascalC.post_process.default import post_process_default
        results = post_process_default(outdir, nbin, mbin, outdir, shot_noise_rescaling, skip_r_bins, print_function = print_and_log)
elif ntracers == 2:
    if legendre:
        from RascalC.post_process.legendre_multi import post_process_legendre_multi
        results = post_process_legendre_multi(outdir, nbin, max_l, outdir, shot_noise_rescaling, shot_noise_rescaling2, skip_r_bins, skip_l, print_function = print_and_log)
    elif jackknife:
        from RascalC.post_process.jackknife_multi import post_process_jackknife_multi
        results = post_process_jackknife_multi(*xi_jack_names, os.path.dirname(jackknife_weights_names[0]), outdir, mbin, outdir, skip_r_bins, print_function = print_and_log)
    else: # default
        from RascalC.post_process.default_multi import post_process_default_multi
        results = post_process_default_multi(outdir, nbin, mbin, outdir, shot_noise_rescaling, shot_noise_rescaling2, skip_r_bins, print_function = print_and_log)
else:
    print("Number of tracers not supported for this operation (yet)")
    sys.exit(1)

print_and_log(datetime.now())

# Convergence check
from RascalC.convergence_check_extra import convergence_check_extra
convergence_check_extra(results, print_function = print_and_log)

print_and_log(datetime.now())
print_and_log(f"Finished execution.")
