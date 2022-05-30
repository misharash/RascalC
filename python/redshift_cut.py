"""Convenience script to perform redshift cut on an input (Ra,Dec,w) FITS or txt file and save the result in a .txt file for use with further scripts. (Michael Rashkovetskyi 2022).
Output file format has (x,y,z,w) coordinates in Mpc/h units 

    Parameters:
        INFILE = input ASCII or FITS file
        OUTFILE = output .txt or .csv file specifier
        Z_MIN = Minimum redshift (inclusive, i.e. will require Z >= Z_MIN)
        Z_MAX = Maximum redshift (non-inclusive, i.e. will require Z < Z_MAX)
        ---OPTIONAL---
        USE_FKP_WEIGHTS = whether to use FKP weights column (default False/0; only applies to (DESI) FITS files)

"""

import sys
import numpy as np

# Check number of parameters
if len(sys.argv) not in (5, 6):
    print("Please specify input arguments in the form convert_to_xyz.py {INFILE} {OUTFILE} {Z_MIN} {Z_MAX} [{USE_FKP_WEIGHTS}]")
    sys.exit()

# Determine whether to use FKP weights, only applies to (DESI) FITS files
if len(sys.argv==6):
    use_FKP_weights = bool(sys.argv[5])
else:
    use_FKP_weights = False
          
# Load file names
input_file = str(sys.argv[1])
output_file = str(sys.argv[2])
print("\nUsing input file %s in Ra,Dec,z coordinates\n"%input_file)

# Load min and max redshifts
z_min = float(sys.argv[3])
z_max = float(sys.argv[4])

if input_file.endswith(".fits"):
    # read fits file, correct for DESI format
    from astropy.io import fits
    print("Reading in data")
    f = fits.open(input_file)
    data = f[1].data
    all_ra = data["RA"]
    all_dec = data["DEC"]
    all_z = data["Z"]
    all_w = data["WEIGHT"]
    if use_FKP_weights:
        all_w *= data["WEIGHT_FKP"]
else:
    # read text file
    # Load in data:
    print("Counting lines in file")
    total_lines=0
    for n, line in enumerate(open(input_file, 'r')):
        total_lines+=1

    all_ra,all_dec,all_z,all_w=[np.zeros(total_lines) for _ in range(4)]

    print("Reading in data");
    for n, line in enumerate(open(input_file, 'r')):
        if n%1000000==0:
            print("Reading line %d of %d" %(n,total_lines))
        split_line=np.array(line.strip().split(), dtype=float) 
        all_ra[n]=split_line[0];
        all_dec[n]=split_line[1];
        all_z[n]=split_line[2];
        all_w[n]=split_line[3];

# perform redshift cut
filt = np.logical_and(z_min <= all_z, all_z < z_max) # filtering condition
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
