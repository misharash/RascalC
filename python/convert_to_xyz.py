"""Convenience script to convert an input (Ra,Dec,w) FITS or txt file to comoving (x,y,z) coordinates saved as a .txt file for use with the main C++ code. (Oliver Philcox 2018 with modifications by Michael Rashkovetskyi 2022, using Daniel Eisenstein's 2015 WCDM Coordinate Converter).
Output file format has (x,y,z,w) coordinates in Mpc/h units 

    Parameters:
        INFILE = input ASCII or FITS file
        OUTFILE = output .txt or .csv file specifier
        ---OPTIONAL---
        OMEGA_M = Matter density (default 0.31)
        OMEGA_K = Curvature density (default 0.0)
        W_DARK_ENERGY = Dark Energy equation of state parameter (default -1.)
        ---FURTHER OPTIONAL---
        USE_FKP_WEIGHTS = whether to use FKP weights column (default False/0; only applies to (DESI) FITS files)
        MASK = sets bins that all must be set in STATUS for the particle to be selected (default 0, only applies to (DESI) FITS files)
        USE_WEIGHTS = whether to use WEIGHTS column, if not, set unit weights (default True/1)

"""

import sys
import numpy as np

if len(sys.argv) not in (3, 4, 5, 6, 7, 8, 9):
    print("Usage: python convert_to_xyz.py {INFILE} {OUTFILE} [{OMEGA_M} {OMEGA_K} {W_DARK_ENERGY} [{USE_FKP_WEIGHTS or P0,NZ_name} [{MASK} [{USE_WEIGHTS}]]]]")
    sys.exit(1)
          
# Load file names
input_file = str(sys.argv[1])
output_file = str(sys.argv[2])
print("\nUsing input file %s in Ra,Dec,z coordinates\n"%input_file)

# Read in optional cosmology parameters
omega_m = float(sys.argv[3]) if len(sys.argv) >= 4 else 0.31
omega_k = float(sys.argv[4]) if len(sys.argv) >= 5 else 0
w_dark_energy = float(sys.argv[5]) if len(sys.argv) >= 6 else -1
# defaults are from the BOSS DR12 2016 clustering paper assuming LCDM

print("\nUsing cosmological parameters as Omega_m = %.2f, Omega_k = %.2f, w = %.2f" %(omega_m,omega_k,w_dark_energy))

# Determine whether to use FKP weights, only applies to (DESI) FITS files
use_FKP_weights = (sys.argv[6].lower() not in ("0", "false")) if len(sys.argv) >= 7 else False # bool(string) is True for non-empty string, so need to be more specific to allow explicit False from a command-line argument
# determine if it actually has P0,NZ_name format. Such strings should give True above.
if use_FKP_weights:
    arg_FKP_split = sys.argv[6].split(",")
    manual_FKP = (len(arg_FKP_split) == 2) # whether to compute FKP weights manually
    if manual_FKP:
        P0 = float(arg_FKP_split[0])
        NZ_name = arg_FKP_split[1]
mask = int(sys.argv[7]) if len(sys.argv) >= 8 else 0 # default is 0 - no filtering
use_weights = (sys.argv[8].lower() not in ("0", "false")) if len(sys.argv) >= 9 else True # use weights by default
filt = True # default pre-filter is true

# Load the wcdm module from Daniel Eisenstein
import os
dirname=os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, str(dirname)+'/wcdm/')
import wcdm

print("Reading in data")
if input_file.endswith(".fits"):
    # read fits file, correct for DESI format
    from astropy.io import fits
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
    
from astropy.constants import c as c_light
import astropy.units as u

print("Converting z to comoving distances:")
all_comoving_radius=wcdm.coorddist(all_z,omega_m,w_dark_energy,omega_k)

# Convert to Mpc/h
H_0_h=100*u.km/u.s/u.Mpc # to ensure we get output in Mpc/h units
H_0_SI = H_0_h.to(1./u.s)
comoving_radius_Mpc = ((all_comoving_radius/H_0_SI*c_light).to(u.Mpc)).value

# Convert to polar coordinates in radians
all_phi_rad = all_ra*np.pi/180.
all_theta_rad = np.pi/2.-all_dec*np.pi/180.

# Now convert to x,y,z coordinates
all_z = comoving_radius_Mpc*np.cos(all_theta_rad)
all_x = comoving_radius_Mpc*np.sin(all_theta_rad)*np.cos(all_phi_rad)
all_y = comoving_radius_Mpc*np.sin(all_theta_rad)*np.sin(all_phi_rad)

print("Writing to file %s:"%output_file)
# Now write to file:
with open(output_file,"w+") as outfile:
    for p in range(len(all_z)):
        outfile.write("%.8f %.8f %.8f %.8f\n" %(all_x[p],all_y[p],all_z[p],all_w[p]))
print("Output positions (of length %d) written succesfully!"%len(all_z))
