"""This script splits fourpoint_info_out files: {DIR}/{i}.txt into {DIR}/{i}/{j}.txt, where {i} is the number of ij bin and {j} is the number of kl bin contained in first column of original file"""

import numpy as np
import sys
import os
import time
init_time=time.time()

if len(sys.argv)<2:
    print("Please specify input parameters in the form {DIR} {NBINS}.")
    sys.exit()
dir_name = str(sys.argv[1])
nbins = int(sys.argv[2])

os.chdir(dir_name)
# loop over i - ij bin number
for i in range(nbins):
    os.mkdir("%d" % i) # make directory
    outfiles = []
    for j in range(nbins):
        outfiles.append(open("%d/%d.txt" % (i, j), "w"))
    print("Processing file %d" % i)
    with open("%d.txt" % i) as infile:
        for l, inline in enumerate(infile):
            j, outline = inline.split(" ", 1) # j (kl bin number) and remainder of the line
            j = int(j) # make j an integer
            outfiles[j].write(outline) # write the remainder of the line to j'th file
        print("Processed file %d with %d lines" % (i, l+1))
    for outfile in outfiles:
        outfile.close()

end=time.time()-init_time
print('Task took %d seconds in total' % end)
