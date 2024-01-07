### Python script for running RascalC in DESI setup (Michael Rashkovetskyi, 2022).
# It is important to check the log file in the output directory to make sure all went well

import os
from datetime import datetime

##################### INPUT PARAMETERS ###################

terminate_on_error = 1 # whether to terminate if any of executed scripts returns nonzero

ntracers = 1 # number of tracers
jackknife = 1 # whether to compute jackknife integrals (must also be set in Makefile)
njack = 60 if jackknife else 0 # number of jackknife regions; if jackknife flag is not set this is used for the pycorr filenames and should be 0
legendre_orig = 0 # original Legendre mode - when each pair's contribution is accumulated to multipoles of 2PCF directly
legendre_mix = 1 # mixed Legendre mode - when the s,mu-binned 2PCF is estimated and then projected into Legendre multipoles using integrals of Legendre polynomials 
legendre = legendre_orig or legendre_mix # any Legendre
if legendre:
    max_l = 4

assert ntracers in (1, 2), "Only single- and two-tracer modes are currently supported"
assert not (jackknife and legendre_orig), "Jackknife and original Legendre modes are incompatible"

nthread = 256 # number of OMP threads to use

version_label = "v0.6"

reg = "SGC" # region for filenames

tlabels = ["LRG"] # tracer labels for filenames
nrandoms = 1 # one file should be enough

# data processing steps
create_jackknives = 1

z_min, z_max = 0.8, 1.1 # for redshift cut and filenames

# File names and directories
if jackknife:
    data_ref_filenames = [f"{tlabel}_{reg}_clustering.dat.fits" for tlabel in tlabels] # only for jackknife reference or ndata backup, has to have rdz contents
    assert len(data_ref_filenames) == ntracers, "Need reference data for all tracers"
input_filenames = [[f"{tlabel}_{reg}_{i}_clustering.ran.fits" for i in range(nrandoms)] for tlabel in tlabels] # random filenames
assert len(input_filenames) == ntracers, "Need randoms for all tracers"
nfiles = [len(input_filenames_group) for input_filenames_group in input_filenames]
outdir = "_".join(tlabels) + "_" + reg + f"_z{z_min}-{z_max}" # output file directory
tmpdir = os.path.join("tmpdirs", outdir) # directory to write intermediate files, mainly data processing steps

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

print("Starting Computation")

def change_extension(name: str, ext: str) -> str:
    return os.path.join(tmpdir, os.path.basename(".".join(name.split(".")[:-1] + [ext]))) # change extension and switch to tmpdir

for t, data_ref_filename in enumerate(data_ref_filenames):
    if create_jackknives:
        data_ref_filenames[t] = change_extension(data_ref_filename, "rdzw")

# processing steps for each random file
for t, (input_filenames_t, nfiles_t) in enumerate(zip(input_filenames, nfiles)):
    print_and_log(f"Starting preparing tracer {t+1} of {ntracers}")
    for i, input_filename in enumerate(input_filenames_t):
        print_and_log(f"Starting preparing file {i+1} of {nfiles_t}")
        print_and_log(datetime.now())
        input_filename = change_extension(input_filename, "xyzw")
        if create_jackknives:
            xyzwj_filename = change_extension(input_filename, "xyzwj")
            from python.create_jackknives_pycorr import create_jackknives_pycorr_files
            create_jackknives_pycorr_files(data_ref_filenames[t], input_filename, xyzwj_filename, njack, print_and_log) # keep in mind some subtleties for multi-tracer jackknife assigment
            input_filename = xyzwj_filename
        input_filenames[t][i] = input_filename # save final input filename for next loop
        print_and_log(f"Finished preparing file {i+1} of {nfiles_t}")
# end processing steps for each random file

# run the test code
os.system(f"./demo {nthread} {10**8}")

print_and_log(datetime.now())
print_and_log(f"Finished execution.")
