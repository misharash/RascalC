"""This script splits c4_details_... file: {INDIR}/c4_details_...bin into {OUTDIR}/{i}/{j}.txt, where {i} is the number of ij bin and {j} is the number of kl bin"""

import numpy as np
import sys
import os
import time
init_time=time.time()

if len(sys.argv)<7:
    print("Please specify input parameters in the form {INDIR} {NBINS} {MBINS} {NBINS_DETAILED} {SUFFIX} {OUTDIR}.")
    sys.exit()
indir_name = str(sys.argv[1])
nbins = int(sys.argv[2])
mbins = int(sys.argv[3])
nbins_tot = nbins*mbins
nbins_detailed = int(sys.argv[4])
suffix = str(sys.argv[5])
outdir_name = str(sys.argv[6])

main_filename = os.path.join(indir_name, "CovMatricesAll/c4_detailed_n%d_m%d_%d_%d%d,%d%d_%s.bin" % (nbins, mbins, nbins_detailed, 1,1,1,1, suffix)) # hardcoded for all particle classes = 1 so far
data_len = (nbins_tot*nbins_detailed)**2
print("Loading %d floats from %s" % (data_len, main_filename))
data = np.fromfile(main_filename, count=data_len) # load binary data
print("Data loaded, reshaping to ({0:d}, {0:d}, {1:d}, {1:d})".format(nbins_tot, nbins_detailed))
data = data.reshape(nbins_tot, nbins_tot, nbins_detailed, nbins_detailed) # reshape into 4D array
print("Data reshaped, symmetrizing in first pair of indices")
data = (data + data.transpose(1, 0, 2, 3))/2 # symmetrize in first two indices
print("Data symmetrized in first pair of indices, symmetrizing in last pair of indices")
data = (data + data.transpose(0, 1, 3, 2))/2 # symmetrize in last two indices
print("Data symmetrized in last pair of indices")

os.mkdir(outdir_name)
print("Writing into separate files by covariance bins, to directory %s" % outdir_name)
# loop over i - ij bin number
for i in range(nbins_tot):
    os.mkdir(os.path.join(outdir_name, str(i)))
    # loop over j - kl bin number to save data, but only j <= i
    for j in range(i+1):
        np.save(os.path.join(outdir_name, str(i), str(j)+".npy"), data[i, j])
    # loop over j - kl bin number to save data, but only j > i - now create symlinks by symmetry
    for j in range(i+1, nbins_tot):
        os.symlink(os.path.join("..", str(j), str(i)+".npy"), os.path.join(outdir_name, str(i), str(j)+".npy"))
print("Written into separate files by covariance bins")

end=time.time()-init_time
print('Task took %d seconds in total' % end)