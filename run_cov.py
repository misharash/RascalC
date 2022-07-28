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

rescale = 1 # rescaling for co-ordinates
nside = 101 # grid size for accelerating pair count
boxsize = 2000 # only used if periodic=1

# data processing steps
redshift_cut = 1
FKP_weight = 1
convert_to_xyz = 1
create_jackknives = jackknife and 1

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
workdir = os.getcwd()
indir = workdir # input directory (see above for required contents)
outdir = os.path.join(workdir, "out") # output file directory
scriptname = "run_cov.py"

# binning files to be created automatically
binfile = "radial_binning_cov.csv"
binfile_cf = "radial_binning_corr.csv"
os.system(f"python python/write_binning_file_linear.py {nbin} {rmin} {rmax} {binfile}")
os.system(f"python python/write_binning_file_linear.py {nbin_cf} {rmin_cf} {rmax_cf} {binfile_cf}")

##########################################################

# Define command to run the C++ code
code = "./cov"

command = f"{code} -perbox {periodic} -boxsize {boxsize} -ngrid {ngrid} -rescale {rescale} -nthread {nthread} -maxloops {maxloops} -xicut {xicutoff} -norm {ndata} -RRbin {binned_pair_name} -binfile {binfile} -binfile_cf {binfile_cf}"
if jackknife:
    command += f" -jackknife {jackknife_weights_name}"

# Create output directory
os.makedirs(outdir, exist_ok=1)

# Copy this script in for posterity
os.system(f"cp {scriptname} {os.path.normpath(outdir)}")

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
    return ".".join(name.split(".")[:-1] + [ext])

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
    os.system(f"{command} -in {input_filename} -output {os.path.join(outdir, str(i))}/ 2>&1 | tee -a {logfile}")
    print_and_log(f"Finished computation {i+1} of {nfiles}")
# end for each random file/part

print_and_log(datetime.now())

# Maybe post-processing will be here later

print_and_log(f"Finished with computation.")
