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

ref_positions = np.loadtxt(reffile_name, usecols=range(3)).T # only read positions
subsampler = KMeansSubsampler('angular', positions=ref_positions, nsamples=njack, nside=512, random_state=42, position_type='rdd')

# First count number of lines
print('Counting number of lines')
with open(infile_name) as f:
    for i, l in enumerate(f):
        pass
total_lines = i + 1
print('Found %s lines in input file' %total_lines)

import time
percent_count=0
init=time.time()
with open(infile_name) as infile:
    with open(outfile_name, "w") as outfile:
        for l,line in enumerate(infile):
            if 100*l/total_lines>=percent_count:
                print(" %d%% done: Reading line %s of %s" %(percent_count,l,total_lines))
                percent_count+=1
            split_line=line.split()
            pix = int(subsampler.label(np.array([[float(s)] for s in split_line[:3]]), position_type='xyz')) # 3 columns for coordinates
            outfile.write(" ".join(split_line[:4])+" "+str(pix)+"\n") # 4 columns for coordinates AND weights
end=time.time()-init
print('Task completed in %.2f seconds' %end) 
