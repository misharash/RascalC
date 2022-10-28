### Python script for running RascalC in DESI setup (Michael Rashkovetskyi, 2022).
# It is important to check the log file in the output directory to make sure all went well

import os, sys
from datetime import datetime
import numpy as np

def check_path(filename, fallback_dir=""):
    if os.path.isfile(filename): return filename
    filename = os.path.join(fallback_dir, os.path.basename(filename))
    assert os.path.isfile(filename), f"{filename} missing"
    return filename

##################### INPUT PARAMETERS ###################

ntracers = 1 # number of tracers
periodic = 0 # whether to run with periodic boundary conditions (must also be set in Makefile)
make_randoms = 0 # how many randoms to generate in periodic case, 0 = don't make any
jackknife = 1 # whether to compute jackknife integrals (must also be set in Makefile)
if jackknife:
    njack = 60 # number of jackknife regions
legendre = 0
if legendre:
    max_l = 4

assert ntracers in (1, 2), "Only single- and two-tracer modes are currently supported"
assert not (make_randoms and jackknife), "Jackknives with generated randoms not implemented"
assert not (make_randoms and not periodic), "Non-periodic randoms not supported"
assert not (jackknife and legendre), "Jackknife and Legendre modes are incompatible"

ndata = [None] * ntracers # number of data points for each tracer; set None to make sure it is overwritten before any usage and see an error otherwise

rmin = 0 # minimum output cov radius in Mpc/h
rmax = 200 # maximum output cov radius in Mpc/h
nbin = 50 # radial bins for output cov
mbin = 1 # angular (mu) bins for output cov
rmin_cf = 0 # minimum input 2PCF radius in Mpc/h
rmax_cf = 200 # maximum input 2PCF radius in Mpc/h
nbin_cf = 100 # radial bins for input 2PCF
mbin_cf = 10 # angular (mu) bins for input 2PCF
xicutoff = 250 # beyond this assume xi/2PCF=0

nthread = 30 # number of OMP threads to use
maxloops = 30 # number of integration loops per filename
N2 = 20 # number of secondary cells/particles per primary cell
N3 = 40 # number of third cells/particles per secondary cell/particle
N4 = 80 # number of fourth cells/particles per third cell/particle

rescale = 1 # rescaling for co-ordinates
nside = 101 # grid size for accelerating pair count
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

reg = "N" # region for filenames
tlabels = ["LRG"] # tracer labels for filenames
assert len(tlabels) == ntracers, "Need label for each tracer"
nrandoms = 10

# data processing steps
redshift_cut = 1
convert_to_xyz = 1
if redshift_cut or convert_to_xyz:
    FKP_weight = 1
    mask = 0 # default, basically no mask. All bits set to 1 in the mask have to be set in the FITS data STATUS
    use_weights = 1
create_jackknives = jackknife and 1
do_counts = 0 # (re)compute total pair counts, jackknife weights/xi with RascalC script, on concatenated randoms, instead of reusing them from pycorr
cat_randoms = 0 # concatenate random files for RascalC input
if do_counts or cat_randoms:
    cat_randoms_files = [f"{tlabel}_{reg}_0-{nrandoms-1}_clustering.ran.xyzw" + ("j" if jackknife else "") for tlabel in tlabels]

z_min, z_max = 0.4, 1.1 # for redshift cut and filenames

# CF options
convert_cf = 1
if convert_cf:
    # first index is correlation function index
    counts_factor = nrandoms
    split_above = 20
    pycorr_filenames = [[check_path(f"/global/cfs/cdirs/desi/survey/catalogs/edav1/xi/da02/smu/allcounts_{corlabel}_{reg}_{z_min}_{z_max}_default_FKP_lin_njack{njack if jackknife else 60}_nran{nrandoms}_split{split_above}.npy")] for corlabel in ["LRG"]]
    assert len(pycorr_filenames) == ncorr, "Expected pycorr file(s) for each correlation"
smoothen_cf = 0
if smoothen_cf:
    max_l = 4
    radial_window_len = 5
    radial_polyorder = 2

# cosmology
if convert_to_xyz:
    Omega_m = 0.31519
    Omega_k = 0
    w_dark_energy = -1

# File names and directories
if jackknife:
    data_ref_filenames = [check_path(f"/global/cfs/cdirs/desi/survey/catalogs/edav1/da02/LSScats/clustering/{tlabel}_{reg}_clustering.dat.fits") for tlabel in tlabels] # for jackknife reference only, has to have rdz contents
    assert len(data_ref_filenames) == ntracers, "Need reference data for all tracers"
input_filenames = [[check_path(f"/global/cfs/cdirs/desi/survey/catalogs/edav1/da02/LSScats/clustering/{tlabel}_{reg}_{i}_clustering.ran.fits") for i in range(nrandoms)] for tlabel in tlabels] # random filenames
assert len(input_filenames) == ntracers, "Need randoms for all tracers"
nfiles = [len(input_filenames_group) for input_filenames_group in input_filenames]
if not cat_randoms or make_randoms:
    for i in range(1, ntracers):
        assert nfiles[i] == nfiles[0], "Need to have the same number of files for all tracers"
outdir = reg # output file directory
tmpdir = outdir # directory to write intermediate files, mainly data processing steps
cornames = [os.path.join(tmpdir, f"xi/xi_n{nbin_cf}_m{mbin_cf}_{index}.dat") for index in indices_corr]
binned_pair_names = [os.path.join(tmpdir, "weights/" + ("binned_pair" if jackknife else "RR") + f"_counts_n{nbin}_m{mbin}" + (f"_j{njack}" if jackknife else "") + f"_{index}.dat") for index in indices_corr]
if jackknife:
    jackknife_weights_names = [os.path.join(tmpdir, f"weights/jackknife_weights_n{nbin}_m{mbin}_j{njack}_{index}.dat") for index in indices_corr]
    if convert_cf:
        xi_jack_names = [os.path.join(tmpdir, f"xi_jack/xi_jack_n{nbin}_m{mbin}_j{njack}_{index}.dat") for index in indices_corr]
        jackknife_pairs_names = [os.path.join(tmpdir, f"weights/jackknife_pair_counts_n{nbin}_m{mbin}_j{njack}_{index}.dat") for index in indices_corr]
if legendre:
    phi_names = [f"BinCorrectionFactor_n{nbin}_" + ("periodic" if periodic else f'm{mbin}') + f"_{index}.txt" for index in indices_corr]

if do_counts or cat_randoms: # move concatenated randoms file to tmpdir as well
    cat_randoms_files = [os.path.join(tmpdir, cat_randoms_file) for cat_randoms_file in cat_randoms_files]

# binning files to be created automatically
binfile = "radial_binning_cov.csv"
binfile_cf = "radial_binning_corr.csv"
os.system(f"python python/write_binning_file_linear.py {nbin} {rmin} {rmax} {binfile}")
os.system(f"python python/write_binning_file_linear.py {nbin_cf} {rmin_cf} {rmax_cf} {binfile_cf}")

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

def print_and_log(s):
    print(s)
    print_log(s)
print_log = lambda l: os.system(f"echo \"{l}\" >> {logfile}")

print_and_log(datetime.now())
print_and_log(f"Executing {__file__}")

def exec_print_and_log(commandline):
    print_and_log(f"Running command: {commandline}")
    os.system(f"{commandline} 2>&1 | tee -a {logfile}")

print("Starting Computation")

# full-survey CF conversion, will also load number of data points from pycorr
if convert_cf:
    r_step_cf = (rmax_cf-rmin_cf)//nbin_cf
    for c, corname in enumerate(cornames):
        os.makedirs(os.path.dirname(corname), exist_ok=1) # make sure all dirs exist
        exec_print_and_log(f"python python/convert_xi_from_pycorr.py {' '.join(pycorr_filenames[c])} {corname} {r_step_cf} {mbin_cf}")
        ndata[tracer2_corr[c]] = np.loadtxt(corname + ".ndata")[1] # override ndata for second tracer, so that autocorrelations are prioritized
        if smoothen_cf:
            corname_old = corname
            corname = f"xi/xi_n{nbin_cf}_m{mbin_cf}_11_smooth.dat"
            exec_print_and_log(f"python python/smoothen_xi.py {corname_old} {max_l} {radial_window_len} {radial_polyorder} {corname}")
            cornames[c] = corname # save outside of the loop

if periodic and make_randoms:
    # create random points
    print_and_log(f"Generating random points")
    np.random.seed(42) # for reproducibility
    randoms = [np.append(np.random.rand(nfiles_t, int(ndata_t), 3) * boxsize, np.ones((nfiles_t, int(ndata_t), 1)), axis=-1) for nfiles_t, ndata_t in zip(nfiles, ndata)]
    # 3 columns of random coordinates within [0, boxsize] and one of weights, all equal to unity. List of array; list index is tracer number, first array index is file number and the second is number of point. Keep number of points roughly equal to number of data for each tracer
    print_and_log(f"Generated random points")

def change_extension(name, ext):
    return os.path.join(tmpdir, os.path.basename(".".join(name.split(".")[:-1] + [ext]))) # change extension and switch to tmpdir

if create_jackknives and redshift_cut: # prepare reference file
    for t, data_ref_filename in enumerate(data_ref_filenames):
        print_and_log(f"Processing data file for jackknife reference")
        rdzw_ref_filename = change_extension(data_ref_filename, "rdzw")
        exec_print_and_log(f"python python/redshift_cut.py {data_ref_filename} {rdzw_ref_filename} {z_min} {z_max} {FKP_weight} {mask} {use_weights}")
        data_ref_filenames[t] = rdzw_ref_filename

command = f"./cov -boxsize {boxsize} -nside {nside} -rescale {rescale} -nthread {nthread} -maxloops {maxloops} -N2 {N2} -N3 {N3} -N4 {N4} -xicut {xicutoff} -binfile {binfile} -binfile_cf {binfile_cf} -mbin_cf {mbin_cf}" # here are universally acceptable parameters
command += "".join([f" -norm{suffixes_tracer[t]} {ndata[t]}" for t in range(ntracers)]) # provide all ndata for normalization
command += "".join([f" -cor{suffixes_corr[c]} {cornames[c]}" for c in range(ncorr)]) # provide all correlation functions
if legendre: # only provide max multipole l for now
    command += f" -max_l {max_l}"
else: # provide binned pair counts files and number of mu bin
    command += "".join([f" -RRbin{suffixes_corr[c]} {binned_pair_names[c]}" for c in range(ncorr)]) + f" -mbin {mbin}"
if periodic: # append periodic flag
    command += " -perbox"
if jackknife: # provide jackknife weight files for all correlations
    command += "".join([f" -jackknife{suffixes_corr[c]} {jackknife_weights_names[c]}" for c in range(ncorr)])
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
                exec_print_and_log(f"python python/redshift_cut.py {input_filename} {rdzw_filename} {z_min} {z_max} {FKP_weight} {mask} {use_weights}")
                input_filename = rdzw_filename
            if convert_to_xyz:
                xyzw_filename = change_extension(input_filename, "xyzw")
                exec_print_and_log(f"python python/convert_to_xyz.py {input_filename} {xyzw_filename} {Omega_m} {Omega_k} {w_dark_energy} {FKP_weight} {mask} {use_weights}")
                input_filename = xyzw_filename
            if create_jackknives:
                xyzwj_filename = change_extension(input_filename, "xyzwj")
                exec_print_and_log(f"python python/create_jackknives_pycorr.py {data_ref_filenames[t]} {input_filename} {xyzwj_filename} {njack}") # keep in mind some subtleties for multi-tracer jackknife assigment
                input_filename = xyzwj_filename
        input_filenames[t][i] = input_filename # save final input filename for next loop
        print_and_log(f"Finished preparing file {i+1} of {nfiles_t}")
# end processing steps for each random file

if cat_randoms: # concatenate randoms
    for t in range(ntracers):
        exec_print_and_log(f"cat {' '.join(input_filenames[t])} > {cat_randoms_files[t]}")
        input_filenames[t] = [cat_randoms_files[t]] # now it is the only file
    nfiles = 1
else:
    nfiles = nfiles[0]
    for t in range(1, ntracers):
        input_filenames[t] = input_filenames[t][t:] + input_filenames[t][:t] # shift the filename list cyclically by number of tracer, this makes sure files with different numbers for different tracers are fed to the C++ code, otherwise overlapping positions are likely at least between LRG and ELG
# now the number of files to process is the same for sure

if convert_cf: # this is really for pair counts and jackknives
    if do_counts: # redo counts
        if jackknife: # do jackknife xi and all counts
            if not cat_randoms: # concatenate randoms now
                exec_print_and_log(f"cat {' '.join(input_filenames[i])} > {cat_randoms_files[i]}")
            # compute jackknife weights
            if ntracers == 1:
                exec_print_and_log(f"python python/jackknife_weights.py {cat_randoms_files[0]} {binfile} 1. {mbin} {nthread} {periodic} {os.path.dirname(jackknife_weights_names[0])}/") # 1. is max mu
            elif ntracers == 2:
                exec_print_and_log(f"python python/jackknife_weights_cross.py {' '.join(cat_randoms_files)} {binfile} 1. {mbin} {nthread} {periodic} {os.path.dirname(jackknife_weights_names[0])}/") # 1. is max mu
            else:
                print("Number of tracers not supported for this operation (yet)")
                sys.exit(1)
            # continue processing of data files - from redshift-cut rdzw to xyzw and xyzwj
            for t in range(ntracers):
                data_filename = data_ref_filenames[t]
                xyzw_filename = change_extension(data_filename, "xyzw")
                exec_print_and_log(f"python python/convert_to_xyz.py {data_filename} {xyzw_filename} {Omega_m} {Omega_k} {w_dark_energy} {FKP_weight} {mask} {use_weights}")
                data_filename = xyzw_filename
                xyzwj_filename = change_extension(data_filename, "xyzwj")
                exec_print_and_log(f"python python/create_jackknives_pycorr.py {data_filename} {data_filename} {xyzwj_filename} {njack}") # keep in mind some subtleties for multi-tracer jackknife assigment
                data_ref_filenames[t] = xyzwj_filename
            # run RascalC own xi jack estimator
            if ntracers == 1:
                exec_print_and_log(f"python python/xi_estimator_jack.py {data_ref_filenames[0]} {cat_randoms_files[0]} {cat_randoms_files[0]} {binfile} 1. {mbin} {nthread} {periodic} {os.path.dirname(xi_jack_names[0])}/ {jackknife_pairs_names[0]}") # 1. is max mu
            elif ntracers == 2:
                exec_print_and_log(f"python python/xi_estimator_jack_cross.py {' '.join(data_ref_filenames)} {' '.join(cat_randoms_files)} {' '.join(cat_randoms_files)} {binfile} 1. {mbin} {nthread} {periodic} {os.path.dirname(xi_jack_names[0])}/ {' '.join(jackknife_pairs_names)}") # 1. is max mu
            else:
                print("Number of tracers not supported for this operation (yet)")
                sys.exit(1)
            if not cat_randoms: # reload full counts from pycorr, override jackknives - to prevent normalization issues
                r_step = (rmax-rmin)//nbin
                for c in range(ncorr):
                    exec_print_and_log(f"python python/convert_counts_from_pycorr.py {pycorr_filenames[c][0]} {binned_pair_names[c]} {r_step} {mbin} {counts_factor} {split_above} {rmax}")
        else: # only need full, binned pair counts
            if cat_randoms: # compute counts with our own script
                if ntracers == 1:
                    exec_print_and_log(f"python python/RR_counts.py {cat_randoms_files[0]} {binfile} 1. {mbin} {nthread} {periodic} {os.path.dirname(binned_pair_names[0])}/ 0") # 1. is max mu, 0 means not normed
                elif ntracers == 2:
                    exec_print_and_log(f"python python/RR_counts_multi.py {' '.join(cat_randoms_files)} {binfile} 1. {mbin} {nthread} {periodic} {os.path.dirname(binned_pair_names[0])}/ 0") # 1. is max mu, 0 means not normed
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
                exec_print_and_log(f"python python/convert_xi_jack_from_pycorr.py {pycorr_filenames[c][0]} {xi_jack_names[c]} {jackknife_weights_names[c]} {jackknife_pairs_names[c]} {binned_pair_names[c]} {r_step} {mbin} {counts_factor} {split_above} {rmax}")
            else: # convert full, binned pair counts
                exec_print_and_log(f"python python/convert_counts_from_pycorr.py {pycorr_filenames[c][0]} {binned_pair_names[c]} {r_step} {mbin} {counts_factor} {split_above} {rmax}")

# running main code for each random file/part
for i in range(nfiles):
    print_and_log(f"Starting main computation {i+1} of {nfiles}")
    # define output subdirectory
    this_outdir = os.path.join(outdir, str(i)) if nfiles > 1 else outdir # create output subdirectory only if processing multiple files
    this_outdir = os.path.normpath(this_outdir) + "/" # make sure there is exactly one slash in the end
    if legendre: # need correction function
        if ntracers == 1:
            exec_print_and_log(f"python python/compute_correction_function.py {input_filenames[0][i]} {binfile} {this_outdir} {periodic}" + (not periodic) * f" {binned_pair_names[0]}")
        elif ntracers == 2:
            exec_print_and_log(f"python python/compute_correction_function_multi.py {' '.join([names[i] for names in input_filenames])} {binfile} {this_outdir} {periodic}" + (not periodic) * f" {' '.join(binned_pair_names)}")
        else:
            print("Number of tracers not supported for this operation (yet)")
            sys.exit(1)
    # run code
    exec_print_and_log(command + "".join([f" -in{suffixes_tracer[t]} {input_filenames[t][i]}" for t in range(ntracers)]) + f" -output {this_outdir}" + ("".join([f" -phi_file{suffixes_corr[c]} {os.path.join(this_outdir, phi_names[c])}" for c in range(ncorr)]) if legendre else ""))
    print_and_log(f"Finished main computation {i+1} of {nfiles}")
# end running main code for each random file/part

print_and_log(datetime.now())

# Concatenate samples
if nfiles > 1:
    print_and_log(f"Concatenating samples")
    exec_print_and_log(f"python python/cat_subsets_of_integrals.py {nbin} {'l' + str(max_l) if legendre else 'm' + str(mbin)} " + " ".join([f"{os.path.join(outdir, str(i))} {maxloops}" for i in range(nfiles)]) + f" {outdir}")

# Maybe post-processing will be here later

print_and_log(f"Finished execution.")
