## Convenience function to create RR pair counts for two sets of random particles, based on Corrfunc.
## This just computes the global RR pair counts, not the jackknife counts.
## If the periodic flag is set, we assume a periodic simulation and measure mu from the Z-axis.

import sys
import os
import numpy as np
import math

# PARAMETERS
if len(sys.argv)!=10:
    print("Usage: python RR_counts_multi.py {RANDOM_PARTICLE_FILE_1} {RANDOM_PARTICLE_FILE_2} {BIN_FILE} {MU_MAX} {N_MU_BINS} {NTHREADS} {PERIODIC} {OUTPUT_DIR} {NORMED}")
    sys.exit(1)
fname = str(sys.argv[1])
fname2 = str(sys.argv[2])
binfile = str(sys.argv[3])
mu_max = float(sys.argv[4])
nmu_bins = int(sys.argv[5])
nthreads = int(sys.argv[6])
periodic = int(sys.argv[7])
outdir=str(sys.argv[8])
normed=int(sys.argv[9])

## First read in weights and positions:

print("Reading in data for file 1")
X, Y, Z, W = np.loadtxt(fname, usecols=range(4)).T
    
print("Reading in data for file 2")
X2, Y2, Z2, W2 = np.loadtxt(fname2, usecols=range(4)).T
    
N = len(X) # number of particles
N2 = len(X2)
weight_sum = np.sum(W)#  normalization by summed weights
weight_sum2 = np.sum(W2)

print("\nNumber of random particles: %.1e (field 1), %.1e (field 2)"%(N,N2))

## Determine number of radial bins in binning file:
print("Counting lines in binfile");
with open(binfile) as f:
    for i, l in enumerate(f):
        pass
nrbins = i + 1
print('%s radial bins are used in this file.' %nrbins)

if not periodic:
    # Compute RR counts for the non-periodic case (measuring mu from the radial direction)
    print("\nUsing non-periodic input data");
    def coord_transform(x,y,z):
        # Convert the X,Y,Z coordinates into Ra,Dec,comoving_distance (for use in corrfunc)
        # Shamelessly stolen from astropy
        xsq = x ** 2.
        ysq = y ** 2.
        zsq = z ** 2.

        com_dist = (xsq + ysq + zsq) ** 0.5
        s = (xsq + ysq) ** 0.5 

        if np.isscalar(x) and np.isscalar(y) and np.isscalar(z):
            Ra = math.atan2(y, x)*180./np.pi
            Dec = math.atan2(z, s)*180./np.pi
        else:
            Ra = np.arctan2(y, x)*180./np.pi+180.
            Dec = np.arctan2(z, s)*180./np.pi

        return com_dist, Ra, Dec

    # Convert coordinates to spherical coordinates
    com_dist,Ra,Dec = coord_transform(X,Y,Z);
    com_dist2,Ra2,Dec2 = coord_transform(X2,Y2,Z2);

    # Now compute RR counts
    from Corrfunc.mocks.DDsmu_mocks import DDsmu_mocks
    
    print("\nComputing pair counts for field 1 x field 1")
    RR11=DDsmu_mocks(1,2,nthreads,mu_max,nmu_bins,binfile,Ra,Dec,com_dist,weights1=W,weight_type='pair_product',
                   verbose=False,is_comoving_dist=True)
    print("\nComputing pair counts for field 1 x field 2")
    RR12=DDsmu_mocks(0,2,nthreads,mu_max,nmu_bins,binfile,Ra,Dec,com_dist,weights1=W,weight_type='pair_product',
                   verbose=False,is_comoving_dist=True,RA2=Ra2,DEC2=Dec2,CZ2=com_dist2,weights2=W2)
    print("\nComputing pair counts for field 2 x field 2")
    RR22=DDsmu_mocks(1,2,nthreads,mu_max,nmu_bins,binfile,Ra2,Dec2,com_dist2,weights1=W2,weight_type='pair_product',
                   verbose=False,is_comoving_dist=True)
    
    # Weight by average particle weighting
    RR_counts11=RR11[:]['npairs']*RR11[:]['weightavg']
    if normed:
        RR_counts11/=np.sum(W)**2.
    RR_counts12=RR12[:]['npairs']*RR12[:]['weightavg']
    if normed:
        RR_counts12/=(np.sum(W)*np.sum(W2))
    RR_counts22=RR22[:]['npairs']*RR22[:]['weightavg']
    if normed:
        RR_counts22/=np.sum(W2)**2.
        
else:
    # Compute RR counts for the periodic case (measuring mu from the Z-axis)
    print("\nUsing periodic input data");
    from Corrfunc.theory.DDsmu import DDsmu
    
    # Iterate over jackknife regions
    print("\nComputing pair counts for field 1 x field 1")
    RR11=DDsmu(1,nthreads,binfile,mu_max,nmu_bins,X,Y,Z,weights1=W,weight_type='pair_product',
             periodic=False,verbose=False)
    print("\nComputing pair counts for field 1 x field 2")
    RR12=DDsmu(0,nthreads,binfile,mu_max,nmu_bins,X,Y,Z,weights1=W,weight_type='pair_product',
             periodic=False,verbose=False,X2=X2,Y2=Y2,Z2=Z2,weights2=W2)
    print("\nComputing pair counts for field 2 x field 2")
    RR22=DDsmu(1,nthreads,binfile,mu_max,nmu_bins,X2,Y2,Z2,weights1=W2,weight_type='pair_product',
             periodic=False,verbose=False)
    # Weight by average particle weighting
    RR_counts11=RR11[:]['npairs']*RR11[:]['weightavg']
    if normed:
        RR_counts11/=np.sum(W)**2.
    RR_counts12=RR12[:]['npairs']*RR12[:]['weightavg']
    if normed:
        RR_counts12/=(np.sum(W)*np.sum(W2))
    RR_counts22=RR22[:]['npairs']*RR22[:]['weightavg']
    if normed:
        RR_counts22/=np.sum(W2)**2.
    

# Make sure output dir exists 
if len(outdir)>0:
    os.makedirs(outdir, exist_ok=1)

outfile11 = os.path.join(outdir, "RR_counts_n%d_m%d_11.txt"%(nrbins,nmu_bins))
print("\nSaving field 1 x field 1 binned pair counts as %s" %outfile11);
with open(outfile11,"w+") as RRfile:
    for i in range(len(RR_counts11)):
        RRfile.write("%.8e\n" %RR_counts11[i])

outfile12 = os.path.join(outdir, "RR_counts_n%d_m%d_12.txt"%(nrbins,nmu_bins))
print("Saving field 1 x field 2 binned pair counts as %s" %outfile12);
with open(outfile12,"w+") as RRfile:
    for i in range(len(RR_counts12)):
        RRfile.write("%.8e\n" %RR_counts12[i])
    
outfile22 = os.path.join(outdir, "RR_counts_n%d_m%d_22.txt"%(nrbins,nmu_bins))
print("Saving field 2 x field 2 binned pair counts as %s" %outfile22);
with open(outfile22,"w+") as RRfile:
    for i in range(len(RR_counts22)):
        RRfile.write("%.8e\n" %RR_counts22[i])

