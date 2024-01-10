## Script to generate RR pair counts from a given set of random particles. This is based on the Corrfunc code of Sinha & Garrison.
## Weights and weighted pair counts are saved in the ../weight_files/ subdirectory. 
## If the periodic flag is set, we assume a periodic simulation and measure mu from the Z-axis.

import sys
import numpy as np
import math

# PARAMETERS
if len(sys.argv)!=8:
    print("Usage: python jackknife_weights.py {RANDOM_PARTICLE_FILE} {BIN_FILE} {MU_MAX} {N_MU_BINS} {NTHREADS} {PERIODIC} {OUTPUT_DIR}")
    sys.exit(1)
fname = str(sys.argv[1])
binfile = str(sys.argv[2])
mu_max = float(sys.argv[3])
nmu_bins = int(sys.argv[4])
nthreads = int(sys.argv[5])
periodic = int(sys.argv[6])
outdir=str(sys.argv[7])


## First read in weights and positions:

print("Reading in data")
X, Y, Z, W, J = np.loadtxt(fname, usecols=range(5)).T
J = J.astype(int) # jackknife region is integer

#J = np.array(J,dtype=int) # convert jackknives to integers
N = len(X) # number of particles
weight_sum = np.sum(W)#  normalization by summed weights
J_regions = np.unique(J) # jackknife regions in use
N_jack = len(J_regions) # number of non-empty jackknife regions

print("Number of random particles %.1e"%N)

## Determine number of radial bins in binning file:
print("Counting lines in binfile");
with open(binfile) as f:
    for i, l in enumerate(f):
        pass
nrbins = i + 1
print('%s radial bins are used in this file.' %nrbins)

if not periodic:
    # Compute RR counts for the non-periodic case (measuring mu from the radial direction)
    print("Using non-periodic input data");
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

    # Now compute RR counts
    from Corrfunc.mocks.DDsmu_mocks import DDsmu_mocks
    RR_aA=np.zeros([N_jack,nrbins*nmu_bins]);

    # Iterate over jackknife regions
    for i,j in enumerate(J_regions):
        filt=np.where(J==j)
        print("Computing pair counts for non-empty jackknife %s of %s." %(i+1,N_jack))
        # Compute pair counts between jackknife region and entire survey volume
        cross_RR=DDsmu_mocks(0,2,nthreads,mu_max,nmu_bins,binfile,Ra,Dec,com_dist,weights1=W,weight_type='pair_product',
                    RA2=Ra[filt],DEC2=Dec[filt],CZ2=com_dist[filt],weights2=W[filt],verbose=False,is_comoving_dist=True)
        # Weight by average particle weighting
        RR_aA[i]=cross_RR[:]['npairs']*cross_RR[:]['weightavg']
        
else:
    # Compute RR counts for the periodic case (measuring mu from the Z-axis)
    print("Using periodic input data");
    from Corrfunc.theory.DDsmu import DDsmu
    
    RR_aA=np.zeros([N_jack,nrbins*nmu_bins]);

    # Iterate over jackknife regions
    for i,j in enumerate(J_regions):
        filt=np.where(J==j)
        print("Computing pair counts for non-empty jackknife %s of %s." %(i+1,N_jack))
        # Compute pair counts between jackknife region and entire survey volume
        cross_RR=DDsmu(0,nthreads,binfile,mu_max,nmu_bins,X,Y,Z,weights1=W,weight_type='pair_product',
                    X2=X[filt],Y2=Y[filt],Z2=Z[filt],weights2=W[filt],periodic=False,verbose=False)
        # Weight by average particle weighting
        RR_aA[i]=cross_RR[:]['npairs']*cross_RR[:]['weightavg']
    

# Now compute weights from pair counts
w_aA=np.zeros_like(RR_aA)
RR_a = np.sum(RR_aA,axis=0)
for i,j in enumerate(J_regions):
    w_aA[i]=RR_aA[i]/RR_a # jackknife weighting for bin and region

# Save output files:
import os
if not os.path.exists(outdir):
    os.makedirs(outdir)

weight_file='jackknife_weights_n%d_m%d_j%d_11.dat'%(nrbins,nmu_bins,N_jack)
print("Saving jackknife weight as %s"%weight_file)
with open(os.path.join(outdir, weight_file), "w+") as weight_file:
    for j_id,jackknife_weight in enumerate(w_aA):
        weight_file.write("%d\t" %J_regions[j_id])
        for i in range(len(jackknife_weight)):
            weight_file.write("%.8e" %jackknife_weight[i])
            if i == len(jackknife_weight)-1:
                weight_file.write("\n");
            else:
                weight_file.write("\t");
RR_a_file = 'binned_pair_counts_n%d_m%d_j%d_11.dat'%(nrbins,nmu_bins,N_jack)
print("Saving binned pair counts as %s" %RR_a_file);
with open(os.path.join(outdir, RR_a_file), "w+") as RR_file:
    for i in range(len(RR_a)):
        RR_file.write("%.8e\n" %RR_a[i])
        
RR_aA_file = 'jackknife_pair_counts_n%d_m%d_j%d_11.dat'%(nrbins,nmu_bins,N_jack)
print("Saving normalized jackknife pair counts as %s"%RR_aA_file)
with open(os.path.join(outdir, RR_aA_file), "w+") as jackRR_file:
    for j_id,pair_count in enumerate(RR_aA):
        this_jk = J_regions[j_id]
        norm = weight_sum**2.
        jackRR_file.write("%d\t" %this_jk)
        for i in range(len(pair_count)):
            jackRR_file.write("%.8e" %(pair_count[i]/norm))
            if i == len(pair_count)-1:
                jackRR_file.write("\n");
            else:
                jackRR_file.write("\t");
        
print("Jackknife weights and pair counts written successfully to the %s directory"%outdir)
        
