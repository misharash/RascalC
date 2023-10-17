"""Convenience script to perform redshift cut on an input (Ra,Dec,z,w) FITS or txt file and save the result in a .txt file for use with further scripts. (Michael Rashkovetskyi 2022).
Output file format has (Ra,Dec,z,w) coordinates

    Parameters:
        INFILE = input ASCII or FITS file
        OUTFILE = output .txt or .csv file specifier
        Z_MIN = Minimum redshift (inclusive, i.e. will require Z >= Z_MIN)
        Z_MAX = Maximum redshift (non-inclusive, i.e. will require Z < Z_MAX)
        ---OPTIONAL---
        USE_FKP_WEIGHTS = whether to use FKP weights column (default False/0; only applies to (DESI) FITS files)
        MASK = sets bins that all must be set in STATUS for the particle to be selected (default 0, only applies to (DESI) FITS files)
        USE_WEIGHTS = whether to use WEIGHTS column, if not, set unit weights (default True/1)

"""

import sys
import numpy as np

# Check number of parameters
if len(sys.argv) not in (5, 6, 7, 8):
    print("Usage: python redshift_cut.py {INFILE} {OUTFILE} {Z_MIN} {Z_MAX} [{USE_FKP_WEIGHTS or P0,NZ_name} [{MASK} [{USE_WEIGHTS}]]]")
    sys.exit(1)
          
# Load file names
input_file = str(sys.argv[1])
output_file = str(sys.argv[2])
print("\nUsing input file %s in Ra,Dec,z coordinates\n"%input_file)

# Load min and max redshifts
z_min = float(sys.argv[3])
z_max = float(sys.argv[4])

# Determine whether to use FKP weights, only applies to (DESI) FITS files
use_FKP_weights = (sys.argv[5].lower() not in ("0", "false")) if len(sys.argv) >= 6 else False # bool(string) is True for non-empty string, so need to be more specific to allow explicit False from a command-line argument
# determine if it actually has P0,NZ_name format. Such strings should give True above.
if use_FKP_weights:
    arg_FKP_split = sys.argv[5].split(",")
    manual_FKP = (len(arg_FKP_split) == 2) # whether to compute FKP weights manually
    if manual_FKP:
        P0 = float(arg_FKP_split[0])
        NZ_name = arg_FKP_split[1]
# Load mask to select STATUS that has all 1-bits set in mask. Also only applies to (DESI) FITS files
mask = int(sys.argv[6]) if len(sys.argv) >= 7 else 0 # default is 0 - no filtering
use_weights = (sys.argv[7].lower() not in ("0", "false")) if len(sys.argv) >= 8 else True # use weights by default
filt = True # default pre-filter is true

if input_file.endswith(".fits"):
    # read fits file, correct for DESI format
    from astropy.io import fits
    print("Reading in data")
    with fits.open(input_file) as f:
        data = f[1].data
        all_ra = data["RA"]
        all_dec = data["DEC"]
        all_z = data["Z"]
        colnames = data.columns.names
        all_w = data["WEIGHT"] if "WEIGHT" in colnames and use_weights else np.ones_like(all_z)
        if use_FKP_weights:
            all_w *= 1/(1+P0*data[NZ_name]) if manual_FKP else data["WEIGHT_FKP"]
        if "WEIGHT" not in colnames and not use_FKP_weights: print("WARNING: no weights found, assigned unit weight to each particle.")
        if mask: filt = (data["STATUS"] & mask == mask) # all 1-bits from mask have to be set in STATUS; skip if mask=0
else:
    # read text file
    all_ra, all_dec, all_z, all_w = np.loadtxt(input_file, usecols=range(4)).T

# perform redshift cut
filt = np.logical_and(filt, np.logical_and(z_min <= all_z, all_z < z_max)) # full filtering condition
all_ra = all_ra[filt]
all_dec = all_dec[filt]
all_z = all_z[filt]
all_w = all_w[filt]

print("Writing to file %s:"%output_file)
# Now write to file:
with open(output_file,"w+") as outfile:
    for p in range(len(all_z)):
        outfile.write("%.8f %.8f %.8f %.8f\n" %(all_ra[p],all_dec[p],all_z[p],all_w[p]))
print("Output positions (of length %d) written succesfully!"%len(all_z))
