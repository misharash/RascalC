""" This function will assign a jackknife region to each input particle, by finding a point on the sky in the other file closest to particle. Data is saved as a 5 column text file."""

import numpy as np
import sys

## PARAMETERS
if len(sys.argv)<4:
    print("Please specify input parameters in the form {CENTERS_FILE} {INPUT_PARTICLES_FILE} {OUTPUT_PARTICLES_FILE}.")
    sys.exit(1)
centersfile_name = str(sys.argv[1])
infile_name = str(sys.argv[2])
outfile_name = str(sys.argv[3])

# Read reference points (circle centers) positions
centers_ra, centers_dec = np.loadtxt(centersfile_name, usecols=(0,1)).T
# convert to spherical angles in radians
centers_phi = centers_ra * np.pi / 180.
centers_theta = np.pi / 2. - centers_dec * np.pi / 180.
# convert to unit vector cartesian components
centers_z = np.cos(centers_theta)
centers_x, centers_y = np.sin(centers_theta) * np.array((np.cos(centers_phi), np.sin(centers_phi)))
centers_coords = np.array((centers_x, centers_y, centers_z))

# Count number of lines
print('Counting number of lines')
with open(infile_name) as f:
    for i, l in enumerate(f):
        pass
total_lines = i + 1
print('Found %s lines in this file' %total_lines)

import time
percent_count=0
init=time.time()
with open(infile_name) as infile:
    with open(outfile_name,"w") as outfile:
        for l,line in enumerate(infile):
            if 100*l/total_lines>=percent_count:
                print(" %d%% done: Reading line %s of %s" %(percent_count,l,total_lines))
                percent_count+=1
            split_line=line.split()
            coords = np.array([float(s) for s in split_line[:3]]) # Cartesian coordinates
            r = np.sqrt(np.sum(coords**2))
            x, y, z = coords / r # unit vector coordinates
            d2 = np.sum((coords[:, None] - centers_coords)**2, axis=0) # squared distance between unit vectors of this particle and reference points
            pix = np.argmin(d2) # region = number of closest ref. point
            outfile.write(line[:-1]+" "+str(pix)+"\n")
end=time.time()-init
print('Task completed in %.2f seconds' %end) 
