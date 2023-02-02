"""Convenience script to normalize weights in an input (x,y,z,w) or (x,y,z,w,j) txt file and save the result in another .txt file for use with further scripts. (Michael Rashkovetskyi 2023).

    Parameters:
        INFILE = input ASCII file
        OUTFILE = output ASCII file

"""

import sys
import numpy as np

# Check number of parameters
if len(sys.argv) != 3:
    print("Usage: python normalize_weights.py {INFILE} {OUTFILE}")
    sys.exit(1)

# Load file names
input_file = str(sys.argv[1])
output_file = str(sys.argv[2])

print("Reading from file %s" % input_file)
contents = np.loadtxt(input_file)
shape = np.shape(contents)
print("Read %d particles" % shape[0])
assert len(shape) == 2, "Input file not read as 2D array"
assert shape[1] >= 4, "Not enough columns to get weights"

print("Normalizing weights")
contents[:, 3] /= np.sum(contents[:, 3]) # weights are the 4th column in r,d,z,w, x,y,z,w and x,y,z,w,j formats

print("Saving to file %s" % output_file)
np.savetxt(output_file, contents) # format?
