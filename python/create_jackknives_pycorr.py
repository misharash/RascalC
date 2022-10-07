""" This function will assign a jackknife region to each input particle, in a way compatible with pycorr used in DESI data processing. Data is saved as a 5 column text file."""

import numpy as np
from pycorr import KMeansSubsampler
import sys

## PARAMETERS
if len(sys.argv) != 5:
    print("Usage: python create_jackknives_pycorr.py {REF_RDZ_FILE} {INPUT_XYZW_FILE} {OUTPUT_XYZWJ_FILE} {NJACK}.")
    sys.exit()
reffile_name = str(sys.argv[1])
infile_name = str(sys.argv[2])
outfile_name = str(sys.argv[3])
njack=int(sys.argv[4])

import time
init=time.time()

print("Reading reference positions from %s" % reffile_name)
ref_positions = np.loadtxt(reffile_name, usecols=range(3)).T # only read positions
print("Initializing a K-means subsampler")
subsampler = KMeansSubsampler('angular', positions=ref_positions, nsamples=njack, nside=512, random_state=42, position_type='rdd')

print("Reading positions and weights from %s" % infile_name)
pos_weights = np.loadtxt(infile_name, usecols=range(4)).T
positions = pos_weights[:3]
print("Assigning jackknives")
jackknives = subsampler.label(positions, position_type='xyz')
print("Saving to %s" % outfile_name)
pos_weights_jack = np.vstack((pos_weights, jackknives))
np.savetxt(outfile_name, pos_weights_jack.T)
end=time.time()-init
print('Task completed in %.2f seconds' %end) 
