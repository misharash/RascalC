### Python script for running RascalC in DESI setup (Michael Rashkovetskyi, 2022).
# It is important to check the log file in the output directory to make sure all went well

import os
from datetime import datetime
import numpy as np

def check_path(filename):
    if os.path.isfile(filename): return filename
    filename = os.path.basename(filename)
    assert os.path.isfile(filename), f"{filename} missing"
    return filename

##################### INPUT PARAMETERS ###################

periodic = 0 # whether to run with periodic boundary conditions (must also be set in Makefile)
make_randoms = 0 # how many randoms to generate in periodic case, 0 = don't make any
jackknife = 1 # whether to compute jackknife integrals (must also be set in Makefile)
if jackknife:
    njack = 60 # number of jackknife regions
legendre = 0
if legendre:
    max_l = 4

assert not (make_randoms and jackknife), "Jackknives with generated randoms not implemented"
assert not (jackknife and legendre), "Jackknife and Legendre modes are incompatible"

ndata = 3e6 # number of data points

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

# data processing steps
redshift_cut = 1
FKP_weight = 0
convert_to_xyz = 1
create_jackknives = jackknife and 1
do_jack_counts = 1 # (re)compute jackknife weights/xi (and pair counts too) with RascalC script
if do_jack_counts:
    cat_randoms_file = "LRG_z0.800_cutsky_seed1_S100-1000_random.xyzw"
# CF options
convert_cf = 1
if convert_cf:
    pycorr_filenames = [check_path("/global/cfs/projectdirs/desi/users/dvalcin/EZMOCKS/LRG/Xi/xi_ez_LRG_cutsky_seed1_z0.4_1.1.npy")]
    pycorr_filename = pycorr_filenames[0]
    counts_factor = 10
    split_above = np.inf
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

z_min, z_max = 0.4, 1.1 # for redshift cut

# File names and directories
data_ref_filename = check_path("/global/cfs/projectdirs/desi/users/dvalcin/EZMOCKS/LRG/Mocks/LRG_z0.800_cutsky_seed1_data.fits") # for jackknife reference only, has to have rdz contents
input_filenames = [check_path(f"/global/cfs/projectdirs/desi/users/dvalcin/EZMOCKS/LRG/Mocks/LRG_z0.800_cutsky_S{i+1}00_random.fits") for i in range(10)] # random filenames
nfiles = len(input_filenames)
corname = f"xi/xi_n{nbin_cf}_m{mbin_cf}_11.dat"
binned_pair_name = f"weights/binned_pair_counts_n{nbin}_m{mbin}" + (f"_j{njack}" if jackknife else "") + "_11.dat"
if jackknife:
    jackknife_weights_name = f"weights/jackknife_weights_n{nbin}_m{mbin}_j{njack}_11.dat"
    if convert_cf:
        xi_jack_name = f"xi_jack/xi_jack_n{nbin}_m{mbin}_j{njack}_11.dat"
        jackknife_pairs_name = f"weights/jackknife_pair_counts_n{nbin}_m{mbin}_j{njack}_11.dat"
if legendre:
    phi_name = f"BinCorrectionFactor_n{nbin}_periodic_11.txt"
outdir = "out" # output file directory

# binning files to be created automatically
binfile = "radial_binning_cov.csv"
binfile_cf = "radial_binning_corr.csv"
os.system(f"python python/write_binning_file_linear.py {nbin} {rmin} {rmax} {binfile}")
os.system(f"python python/write_binning_file_linear.py {nbin_cf} {rmin_cf} {rmax_cf} {binfile_cf}")

##########################################################

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
    os.system(f"{commandline} | tee -a {logfile}")

print("Starting Computation")

if periodic and make_randoms:
    # create random points
    print_and_log(f"Generating random points")
    nrandoms = int(ndata)
    np.random.seed(42) # for reproducibility
    randoms = np.append(np.random.rand(nfiles, nrandoms, 3) * boxsize, np.ones((nfiles, nrandoms, 1)), axis=-1)
    # 3 columns of random coordinates within [0, boxsize] and one of weights, all equal to unity
    print_and_log(f"Generated random points")

def change_extension(name, ext):
    return os.path.basename(".".join(name.split(".")[:-1] + [ext])) # change extension and switch to current dir

if create_jackknives and redshift_cut: # prepare reference file
    print_and_log(f"Processing data file for jackknife reference")
    rdzw_ref_filename = change_extension(data_ref_filename, "rdzw")
    exec_print_and_log(f"python python/redshift_cut.py {data_ref_filename} {rdzw_ref_filename} {z_min} {z_max} {FKP_weight}")
    data_ref_filename = rdzw_ref_filename

command = f"./cov -boxsize {boxsize} -nside {nside} -rescale {rescale} -nthread {nthread} -maxloops {maxloops} -N2 {N2} -N3 {N3} -N4 {N4} -xicut {xicutoff} -norm {ndata} -cor {corname} -binfile {binfile} -binfile_cf {binfile_cf} -mbin_cf {mbin_cf}"
if legendre:
    command += f" -max_l {max_l} -phi_file {phi_name}"
else:
    command += f" -RRbin {binned_pair_name} -mbin {mbin}"
if periodic:
    command += " -perbox"
if jackknife:
    command += f" -jackknife {jackknife_weights_name}"
print_and_log(f"Common command for C++ code: {command}")

# processing steps for each random file
for i, input_filename in enumerate(input_filenames):
    print_and_log(f"Starting preparation {i+1} of {nfiles}")
    print_and_log(datetime.now())
    if periodic and make_randoms: # just save randoms to this file
        input_filename = change_extension(input_filename, "xyzw")
        np.savetxt(input_filename, randoms[i])
    else: # (potentially) run through all data processing steps
        if redshift_cut:
            rdzw_filename = change_extension(input_filename, "rdzw")
            exec_print_and_log(f"python python/redshift_cut.py {input_filename} {rdzw_filename} {z_min} {z_max} {FKP_weight}")
            input_filename = rdzw_filename
        if convert_to_xyz:
            xyzw_filename = change_extension(input_filename, "xyzw")
            exec_print_and_log(f"python python/convert_to_xyz.py {input_filename} {xyzw_filename} {Omega_m} {Omega_k} {w_dark_energy} {FKP_weight}")
            input_filename = xyzw_filename
        if create_jackknives:
            xyzwj_filename = change_extension(input_filename, "xyzwj")
            exec_print_and_log(f"python python/create_jackknives_pycorr.py {data_ref_filename} {input_filename} {xyzwj_filename} {njack}")
            input_filename = xyzwj_filename
    input_filenames[i] = input_filename # save final input filename for next loop
    print_and_log(f"Finished preparation {i+1} of {nfiles}")
# end processing steps for each random file

# CF conversion
if convert_cf:
    # full-survey CF
    os.makedirs(os.path.dirname(corname), exist_ok=1) # make sure all dirs exist
    r_step_cf = (rmax_cf-rmin_cf)//nbin_cf
    exec_print_and_log(f"python python/convert_xi_from_pycorr.py {' '.join(pycorr_filenames)} {corname} {r_step_cf} {mbin_cf}")
    ndata = np.loadtxt(corname + ".ndata")[0] # override ndata
    if smoothen_cf:
        corname_old = corname
        corname = f"xi/xi_n{nbin_cf}_m{mbin_cf}_11_smooth.dat"
        exec_print_and_log(f"python python/smoothen_xi.py {corname_old} {max_l} {radial_window_len} {radial_polyorder} {corname}")
    os.makedirs(os.path.dirname(binned_pair_name), exist_ok=1) # make sure all dirs exist
    r_step = (rmax-rmin)//nbin
    if jackknife: # convert jackknife xi and all counts
        for filename in (xi_jack_name, jackknife_weights_name, jackknife_pairs_name):
            os.makedirs(os.path.dirname(filename), exist_ok=1) # make sure all dirs exist
        if do_jack_counts:
            # concatenate randoms
            exec_print_and_log(f"cat {' '.join(input_filenames)} > {cat_randoms_file}")
            # continue processing of data file - from redshift-cut rdzw to xyzw and xyzwj
            data_filename = data_ref_filename
            xyzw_filename = change_extension(data_filename, "xyzw")
            exec_print_and_log(f"python python/convert_to_xyz.py {data_filename} {xyzw_filename} {Omega_m} {Omega_k} {w_dark_energy} {FKP_weight}")
            data_filename = xyzw_filename
            xyzwj_filename = change_extension(data_filename, "xyzwj")
            exec_print_and_log(f"python python/create_jackknives_pycorr.py {data_ref_filename} {data_filename} {xyzwj_filename} {njack}")
            data_filename = xyzwj_filename
            # run RascalC own xi jack estimator
            exec_print_and_log(f"python xi_estimator_jack.py {data_filename} {cat_randoms_file} {cat_randoms_file} {binfile} 1 {mbin} {nthread} {periodic} weights/")
            # reload full counts from pycorr, override jackknives
            exec_print_and_log(f"python python/convert_counts_from_pycorr.py {pycorr_filename} {binned_pair_name} {r_step} {mbin} {counts_factor} {split_above}")
        else:
            exec_print_and_log(f"python python/convert_xi_jack_from_pycorr.py {pycorr_filename} {xi_jack_name} {jackknife_weights_name} {jackknife_pairs_name} {binned_pair_name} {r_step} {mbin} {counts_factor} {split_above}")
    else: # only convert full, binned pair counts
        exec_print_and_log(f"python python/convert_counts_from_pycorr.py {pycorr_filename} {binned_pair_name} {r_step} {mbin} {counts_factor} {split_above}")

# running main code for each random file/part
for i, input_filename in enumerate(input_filenames):
    print_and_log(f"Starting main computation {i+1} of {nfiles}")
    # define output subdirectory
    this_outdir = os.path.join(outdir, str(i)) if nfiles > 1 else outdir # create output subdirectory only if processing multiple files
    this_outdir = os.path.normpath(this_outdir) + "/" # make sure there is exactly one slash in the end
    if legendre: # need correction function
        exec_print_and_log(f"python python/compute_correction_function.py {input_filename} {binfile} {this_outdir} {periodic}" + (not periodic) * f" {binned_pair_name}")
    # run code
    exec_print_and_log(f"{command} -in {input_filename} -output {this_outdir} 2>&1" + (f" -phi_file {os.path.join(this_outdir, phi_name)}" if legendre else ""))
    print_and_log(f"Finished main computation {i+1} of {nfiles}")
# end running main code for each random file/part

print_and_log(datetime.now())

# Concatenate samples
if nfiles > 1:
    print_and_log(f"Concatenating samples")
    exec_print_and_log(f"python python/cat_subsets_of_integrals.py {nbin} {mbin} " + " ".join([f"{os.path.join(outdir, str(i))} {maxloops}" for i in range(nfiles)]) + f" {outdir}")

# Maybe post-processing will be here later

print_and_log(f"Finished execution.")
