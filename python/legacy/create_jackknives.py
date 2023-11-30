""" This function will assign a jackknife region to each input particle, by assigning a HEALPix pixel number to each datapoint, with a given value of NSIDE. Data is saved as a 5 column text file."""

import numpy as np
import healpy as hp
import sys
import time
from tqdm import tqdm

## PARAMETERS
if len(sys.argv)<4:
    print("Please specify input parameters in the form {INPUT_FILE} {OUTPUT_FILE} {HEALPIX_NSIDE}.")
    sys.exit(1)
infile_name = str(sys.argv[1])
outfile_name = str(sys.argv[2])
NSIDE = int(sys.argv[3])
NEST = False # Whether to use Healpix NEST or RING ordering

# First count number of lines
print('Counting number of lines')
with open(infile_name) as f:
    for i, l in enumerate(f):
        pass
total_lines = i + 1
print('Found %s lines in this file' % total_lines)

init = time.time()
with open(infile_name) as infile:
    with open(outfile_name, "w") as outfile:
        for l, line in tqdm(enumerate(infile), total=total_lines, desc="Reading particles"):
            split_line = line.split()
            pix = int(hp.vec2pix(NSIDE, float(split_line[0]), float(split_line[1]), float(split_line[2]), nest=NEST))
            outfile.write(line[:-1] + " " + str(pix) + "\n")
end = time.time() - init
print('Task completed in %.2f seconds' % end) 
