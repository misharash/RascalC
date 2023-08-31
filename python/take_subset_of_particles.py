"""This script takes an input txt file listing the (x,y,z,w,j) coordinates of random particles and selects N random particles from this, which it writes to file."""

import numpy as np
import sys
import time
from tqdm import tqdm
init_time=time.time()

if len(sys.argv)<4:
    print("Please specify input parameters in the form {INPUT_FILE} {OUTPUT_FILE} {N_PARTICLES}.")
    sys.exit(1)
infile_name = str(sys.argv[1])
outfile_name = str(sys.argv[2])
N = int(sys.argv[3])
    
# First count number of lines
print('Counting particles in file')
with open(infile_name) as f:
    for i, l in enumerate(f):
        pass
total_lines = i + 1
print('Found %s lines in this file' % total_lines)

# Now create an ordered list of random particle numbers from the list of particles
list_of_indices = np.arange(0,total_lines)
random_indices = np.random.choice(list_of_indices,N,replace=False)
random_indices.sort()

count = 0 # number of particles chosen so far

# Now read in correct particles and save to a file
with open(infile_name) as infile:
    with open(outfile_name, "w") as outfile:
        for l, line in tqdm(enumerate(infile), total=total_lines, desc="Reading particles"):
            if l == random_indices[count]:
                count += 1
                outfile.write(line)
            if count >= N: break # have chosen the required number of particles, can skip the rest of the loop
            
end = time.time() - init_time
print('Task took %d seconds in total and uses %.2f%% of the available particles' % (end, float(N)/float(total_lines)*100.)) 
