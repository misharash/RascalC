### Python script for running RascalC in DESI setup (Michael Rashkovetskyi, 2022).
# It is important to check the log file in the output directory to make sure all went well

import os
from datetime import datetime
import numpy as np

##################### INPUT PARAMETERS ###################

periodic = 1 # whether to run with periodic boundary conditions (must also be set in Makefile)
make_randoms = 0 # how many randoms to generate in periodic case, 0 = don't make any
jackknife = 1 # whether to compute jackknife integrals (must also be set in Makefile)
njack = 60 # number of jackknife regions

assert not (make_randoms and jackknife), "Jackknives with generated randoms not implemented"

ndata = 3e6 # number of data points

rmin = 0 # minimum output cov radius in Mpc/h
rmax = 200 # maximum output cov radius in Mpc/h
nbin = 50 # radial bins for output cov
mbin = 1 # angular (mu) bins for output cov
rmin_cf = 0 # minimum input 2PCF radius in Mpc/h
rmax_cf = 200 # maximum input 2PCF radius in Mpc/h
nbin_cf = 200 # radial bins for input 2PCF
mbin_cf = 10 # angular (mu) bins for input 2PCF
xicutoff = 250 # beyond this assume xi/2PCF=0

nthread = 30 # number of OMP threads to use
maxloops = 60 # number of integration loops per filename
N2 = 20 # number of secondary cells/particles per primary cell
N3 = 40 # number of third cells/particles per secondary cell/particle
N4 = 80 # number of fourth cells/particles per third cell/particle

rescale = 1 # rescaling for co-ordinates
nside = 101 # grid size for accelerating pair count
boxsize = 2000 # only used if periodic=1

# data processing steps
redshift_cut = 1
FKP_weight = 1
convert_to_xyz = 1
create_jackknives = jackknife and 1
# CF options
convert_cf = 1
if convert_cf:
    pycorr_filename = "allcounts_LRG_NScomb_0.4_1.1_default_FKP_lin_njack60_nran10.npy"
    counts_factor = 10
smoothen_cf = 1
if smoothen_cf:
    max_l = 8
    radial_window_len = 5
    radial_polyorder = 2

# cosmology
if convert_to_xyz:
    Omega_m = 0.31519
    Omega_k = 0
    w_dark_energy = -1

z_min, z_max = 0.4, 1.1 # for redshift cut

# File names and directories
data_ref_filename = "LRG_N_clustering.dat.fits" # for jackknife reference only, has to have rdz contents
input_filenames = ["LRG_N_0_clustering.ran.fits"] # random filenames
nfiles = len(input_filenames)
corname = f"xi/xi_n{nbin}_m{mbin}_11.dat"
binned_pair_name = f"weights/binned_pair_counts_n{nbin}_m{mbin}_j{njack}_11.dat"
if jackknife:
    jackknife_weights_name = f"weights/jackknife_weights_n{nbin}_m{mbin}_j{njack}_11.dat"
    if convert_cf:
        xi_jack_name = f"xi_jack/xi_jack_n{nbin}_m{mbin}_j{njack}_11.dat"
        jackknife_pairs_name = f"weights/jackknife_pair_counts_n{nbin}_m{mbin}_j{njack}_11.dat"
outdir = "out" # output file directory

# CF conversion
if convert_cf:
    # full-survey CF
    os.makedirs(os.path.dirname(corname), exist_ok=1) # make sure all dirs exist
    r_step = (rmax_cf-rmin_cf)//nbin_cf
    os.system(f"python python/convert_xi_from_pycorr.py {pycorr_filename} {corname} {r_step} {mbin_cf} | tee -a {logfile}")
    if smoothen_cf:
        corname_old = corname
        corname = f"xi/xi_n{nbin}_m{mbin}_11_smooth.dat"
        os.system(f"python python/smoothen_xi.py {corname_old} {max_l} {radial_window_len} {radial_polyorder} {corname} | tee -a {logfile}")
    os.makedirs(os.path.dirname(binned_pair_name), exist_ok=1) # make sure all dirs exist
    if jackknife: # convert jackknife xi and all counts
        for filename in (xi_jack_name, jackknife_weights_name, jackknife_pairs_name):
            os.makedirs(os.path.dirname(filename), exist_ok=1) # make sure all dirs exist
        os.system(f"python python/convert_xi_jack_from_pycorr.py {pycorr_filename} {xi_jack_name} {jackknife_weights_name} {jackknife_pairs_name} {binned_pair_name} {r_step} {mbin_cf} {counts_factor} | tee -a {logfile}")
    else: # only convert full, binned pair counts
        os.system(f"python python/convert_xi_jack_from_pycorr.py {pycorr_filename} {binned_pair_name} {r_step} {mbin_cf} {counts_factor} | tee -a {logfile}")

# binning files to be created automatically
binfile = "radial_binning_cov.csv"
binfile_cf = "radial_binning_corr.csv"
os.system(f"python python/write_binning_file_linear.py {nbin} {rmin} {rmax} {binfile}")
os.system(f"python python/write_binning_file_linear.py {nbin_cf} {rmin_cf} {rmax_cf} {binfile_cf}")

##########################################################

# Define command to run the C++ code
code = "./cov"

command = f"{code} -perbox {periodic} -boxsize {boxsize} -ngrid {ngrid} -rescale {rescale} -nthread {nthread} -maxloops {maxloops} -N2 {N2} -N3 {N3} -N4 {N4} -xicut {xicutoff} -norm {ndata} -RRbin {binned_pair_name} -binfile {binfile} -binfile_cf {binfile_cf}"
if jackknife:
    command += f" -jackknife {jackknife_weights_name}"

# Create output directory
os.makedirs(outdir, exist_ok=1)

# Copy this script in for posterity
os.system(f"cp {__file__} {os.path.normpath(outdir)}")

# Create an output file for errors
logfilename = "log.txt"
logfile = os.path.join(outdir, logfilename)

def print_and_log(s):
    print(s)
    print_log(s)
print_log = lambda l: os.system(f"echo \"{l}\" >> {logfile}")

print_and_log(datetime.now())
print_and_log(f"Executing {__file__}")
print_and_log(command)

print("Starting Computation")

if periodic and make_randoms:
    # create random points
    print_and_log(f"Generating random points")
    ndata = int(ndata) # no harm since already used for command generation
    np.random.seed(42) # for reproducibility
    randoms = np.append(np.random.rand(nfiles, ndata, 3) * boxsize, np.ones((nfiles, Nrandoms, 1)), axis=-1)
    # 3 columns of random coordinates within [0, boxsize] and one of weights, all equal to unity
    print_and_log(f"Generated random points")

def change_extension(name, ext):
    return os.path.basename(".".join(name.split(".")[:-1] + [ext])) # change extension and switch to current dir

if create_jackknives and redshift_cut: # prepare reference file
    print_and_log(f"Processing data file for jackknife reference")
    rdzw_ref_filename = change_extension(data_ref_filename, "rdzw")
    os.system(f"python python/redshift_cut.py {input_filename} {rdzw_filename} {z_min} {z_max} {FKP_weight} | tee -a {logfile}")
    data_ref_filename = rdzw_ref_filename

# for each random file/part
for i, input_filename in enumerate(input_filenames):
    print_and_log(f"Starting computation {i+1} of {nfiles}")
    print_and_log(datetime.now())
    if periodic and make_randoms: # just save randoms to this file
        input_filename = change_extension(input_filename, "xyzw")
        np.savetxt(input_filename, randoms[i])
    else: # (potentially) run through all data processing steps
        if redshift_cut:
            rdzw_filename = change_extension(input_filename, "rdzw")
            os.system(f"python python/redshift_cut.py {input_filename} {rdzw_filename} {z_min} {z_max} {FKP_weight} | tee -a {logfile}")
            input_filename = rdzw_filename
        if convert_to_xyz:
            xyzw_filename = change_extension(input_filename, "xyzw")
            os.system(f"python python/convert_to_xyz.py {input_filename} {xyzw_filename} {Omega_m} {Omega_k} {w_dark_energy} {FKP_weight} | tee -a {logfile}")
            input_filename = xyzw_filename
        if create_jackknives:
            xyzwj_filename = change_extension(input_filename, "xyzwj")
            os.system(f"python python/create_jackknives_pycorr.py {data_ref_filename} {input_filename} {xyzwj_filename} {njack} | tee -a {logfile}")
            input_filename = xyzwj_filename
    # run code
    this_outdir = os.path.join(outdir, str(i)) if nfiles > 1 else outdir # create output subdirectory only if processing multiple files
    this_outdir = os.path.normpath(this_outdir) + "/" # make sure there is exactly one slash in the end
    os.system(f"{command} -in {input_filename} -output {this_outdir} 2>&1 | tee -a {logfile}")
    print_and_log(f"Finished computation {i+1} of {nfiles}")
# end for each random file/part

print_and_log(datetime.now())

# Concatenate samples
if nfiles > 1:
    print_and_log(f"Concatenating samples")
    os.system(f"python python/cat_subsets_of_integrals.py {nbin} {mbin} " + " ".join([f"{os.path.join(outdir, str(i))} {maxloops}" for i in range(nfiles)]) + f" {outdir} | tee -a {logfile}")

# Maybe post-processing will be here later

print_and_log(f"Finished execution.")
