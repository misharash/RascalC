## Script to compute jackknife estimates of the correlation function xi^J(r,mu) via the Landy-Szalay estimator for a single set of random particles.
## If the periodic flag is set, we assume a periodic simulation and measure mu from the Z-axis.
## This must be binned in the same binning as the desired covariance matrix

import sys
import numpy as np
import math

# PARAMETERS
if len(sys.argv) not in (10, 11, 12):
    print("Usage: python xi_estimator_jack.py {GALAXY_FILE} {RANDOM_FILE_DS} {RANDOM_FILE_SS} {RADIAL_BIN_FILE} {MU_MAX} {N_MU_BINS} {NTHREADS} {PERIODIC} {OUTPUT_DIR} [{RR_jackknife_counts} [{RECOMPUTE_SS}]]")
    sys.exit()

Dname = str(sys.argv[1])
RnameDS = str(sys.argv[2])
RnameSS = str(sys.argv[3])
binfile = str(sys.argv[4])
mu_max = float(sys.argv[5])
nmu_bins = int(sys.argv[6])
nthreads = int(sys.argv[7])
periodic = int(sys.argv[8])
outdir = str(sys.argv[9])

if len(sys.argv) >= 11:
    print("Using pre-defined RR counts")
    RRname=str(sys.argv[10])
else:
    RRname=""

recompute_SS = False
if len(sys.argv) >= 12:
    recompute_SS = (sys.argv[11].lower() not in ("0", "false"))
    if recompute_SS:
        print("Will recompute the SS counts using the input random file")

## First read in weights and positions:
dtype = np.double 

# Read first set of randoms
print("Counting lines in DS random file")
total_lines = 0
for n, line in enumerate(open(RnameDS, 'r')):
    total_lines += 1

rX_DS, rY_DS, rZ_DS, rW_DS, rJ_DS = np.zeros((5, total_lines))

print("Reading in DS random data");
for n, line in enumerate(open(RnameDS, 'r')):
    if n % 1000000 == 0:
        print("Reading line %d of %d" % (n, total_lines))
    split_line = np.array(line.split(" "), dtype=float) 
    rX_DS[n], rY_DS[n], rZ_DS[n], rW_DS[n], rJ_DS[n] = split_line[:5]

N_randDR = len(rX_DS) # number of particles

if len(RRname) == 0 or recompute_SS: # read SS file if RR are not given or asked to recompute the SS
    if RnameSS != RnameDS:
        # only read in SS file if distinct from DS and needed by the code:
        print("Counting lines in SS random file")
        total_lines = 0
        for n, line in enumerate(open(RnameSS, 'r')):
            total_lines += 1

        rX_SS, rY_SS, rZ_SS, rW_SS, rJ_SS = np.zeros((5, total_lines))

        print("Reading in SS random data");
        for n, line in enumerate(open(RnameSS, 'r')):
            if n % 1000000 == 0:
                print("Reading line %d of %d" % (n, total_lines))
            split_line=np.array(line.split(" "), dtype=float) 
            rX_SS[n], rY_SS[n], rZ_SS[n], rW_SS[n], rJ_SS[n] = split_line[:5]
            
        N_randRR = len(rX_SS) # number of particles
    else:
        # just copy if its the same
        rX_SS = rX_DS
        rY_SS = rY_DS
        rZ_SS = rZ_DS
        rW_SS = rW_DS
        rJ_SS = rJ_DS
        N_randRR = N_randDR
else:
    # empty placeholders
    rX_SS, rY_SS, rZ_SS, rW_SS, rJ_SS = [], [], [], [], []
    print("Counting lines in SS random file")
    N_randRR = 0
    for n, line in enumerate(open(RnameSS, 'r')):
        N_randRR += 1

print("Counting lines in galaxy file")
total_lines = 0
for n, line in enumerate(open(Dname, 'r')):
    total_lines += 1

dX, dY, dZ, dW, dJ = np.zeros((5, total_lines))

print("Reading in galaxy data");
for n, line in enumerate(open(Dname, 'r')):
    if n % 1000000 == 0:
        print("Reading line %d of %d" % (n, total_lines))
    split_line = np.array(line.split(" "), dtype=float) 
    dX[n], dY[n], dZ[n], dW[n], dJ[n] = split_line[:5]

N_gal = len(dX) # number of particles

print("Number of random particles %.1e (DR) %.1e (RR)"%(N_randDR, N_randRR))
print("Number of galaxy particles %.1e"%N_gal)

# Determine number of jackknifes
J_regions = np.unique(np.concatenate([rJ_SS, rJ_DS, dJ])) # no harm in checking them all in any case
N_jack = len(J_regions)

print("Using %d non-empty jackknife regions" % N_jack)

## Determine number of radial bins in binning file:
print("Counting lines in binfile");
with open(binfile) as f:
    for i, l in enumerate(f):
        pass
nrbins = i + 1
all_bins = np.loadtxt(binfile)
mean_bins = 0.5*(all_bins[:, 0] + all_bins[:, 1])
print('%s radial bins are used in this file in the range [%d, %d]' % (nrbins, all_bins[0, 0], all_bins[-1, 1]))

if not periodic:
    # Compute RR, DR and DD counts for the non-periodic case (measuring mu from the radial direction)
    print("Using non-periodic input data");
    def coord_transform(x, y, z):
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
    r_com_dist_DS, r_Ra_DS, r_Dec_DS = coord_transform(rX_DS, rY_DS, rZ_DS)
    d_com_dist, d_Ra, d_Dec = coord_transform(dX, dY, dZ)

    from Corrfunc.mocks.DDsmu_mocks import DDsmu_mocks
    
    import time
    init=time.time()
    
    # Now compute RR counts
    RR_counts, SS_counts = np.zeros([2, N_jack, nrbins*nmu_bins])    
    if len(RRname)!=0:
        RRfile = np.loadtxt(RRname) # read pre-computed RR counts
        if len(RRfile[0, :]) != (1+nrbins*nmu_bins):
            raise Exception("Incorrect number of bins in RR file. Either provide the relevant file or recompute RR pair counts for each unrestricted jackknife.")
        if len(RR_counts[:,0]) != N_jack:
            raise Exception("Incorrect number of jackknives in RR file. Either provide the relevant file or recompute RR pair counts for each unrestricted jackknife.")
        for jk in range(N_jack):
            RR_counts[jk, :] = RRfile[jk, 1:] # first index is jackknife number usually
            # NB: these are already normalized
        if not recompute_SS:
            SS_counts = RR_counts
    
    if recompute_SS or len(RRname) == 0:
        r_com_dist_SS, r_Ra_SS, r_Dec_SS = coord_transform(rX_SS, rY_SS, rZ_SS)
        # Compute RR pair counts
        for i, j in enumerate(J_regions):
            # Compute pair counts between jackknife region and entire survey regions
            filt = np.where(rJ_SS==j)
            print("Computing RR pair counts for non-empty jackknife %d of %d"%(i+1,N_jack))
            if len(filt[0])>0:
                cross_SS = DDsmu_mocks(0,2,nthreads,mu_max,nmu_bins,binfile,r_Ra_SS,r_Dec_SS,r_com_dist_SS,weights1=rW_SS,
                                      weight_type='pair_product',RA2=r_Ra_SS[filt],DEC2=r_Dec_SS[filt],CZ2=r_com_dist_SS[filt],
                                      weights2=rW_SS[filt],verbose=False,is_comoving_dist=True)
                SS_counts[i, :] += cross_SS[:]['npairs']*cross_SS[:]['weightavg']
                SS_counts[i, :] /= np.sum(rW_SS)**2. # normalize by product of sum of weights
        if len(RRname) == 0:
            RR_counts = SS_counts
    print("Finished RR pair counts after %d seconds" % (time.time()-init))
    
    # Now compute DR counts
    DS_counts = np.zeros_like(RR_counts)
    for i, j in enumerate(J_regions):
        print("Computing DR pair counts for non-empty jackknife %d of %d" % (i+1, N_jack))
        
        # Compute pair counts between data jackknife and random survey
        filt = np.where(dJ == j)
        if len(filt[0]) > 0:
            cross_DS = DDsmu_mocks(0,2,nthreads,mu_max,nmu_bins,binfile,d_Ra[filt],d_Dec[filt],d_com_dist[filt],
                                   weights1=dW[filt],weight_type='pair_product', RA2=r_Ra_DS, DEC2=r_Dec_DS, 
                                   CZ2 = r_com_dist_DS, weights2 = rW_DS, verbose=False,is_comoving_dist=True)
            DS_counts[i,:] += 0.5*cross_DS[:]['npairs']*cross_DS[:]['weightavg']
        
        # Compute pair coutnts between random jackknife and data survey
        filt2 = np.where(rJ_DS == j)
        if len(filt2[0]) > 0:
            cross_DS = DDsmu_mocks(0,2,nthreads,mu_max,nmu_bins,binfile,d_Ra,d_Dec,d_com_dist,
                                   weights1=dW,weight_type='pair_product', RA2=r_Ra_DS[filt2], DEC2=r_Dec_DS[filt2], 
                                   CZ2 = r_com_dist_DS[filt2], weights2 = rW_DS[filt2], verbose=False,is_comoving_dist=True)
            DS_counts[i,:] += 0.5*cross_DS[:]['npairs']*cross_DS[:]['weightavg']
        DS_counts[i,:] /= np.sum(rW_DS)*np.sum(dW) # normalize by product of sum of weights
    print("Finished DR pair counts after %d seconds"%(time.time()-init))
    
    # Now compute DD counts
    DD_counts = np.zeros_like(RR_counts)
    for i, j in enumerate(J_regions):
        # Compute pair counts between jackknife region and entire survey regions
        filt = np.where(dJ == j)
        print("Computing DD pair counts for non-empty jackknife %d of %d"%(i+1,N_jack))
        if len(filt[0])>0:
            cross_DD = DDsmu_mocks(0,2,nthreads,mu_max,nmu_bins,binfile,d_Ra,d_Dec,d_com_dist,weights1=dW,
                                    weight_type='pair_product',RA2=d_Ra[filt],DEC2=d_Dec[filt],CZ2=d_com_dist[filt],
                                    weights2=dW[filt],verbose=False,is_comoving_dist=True)
            DD_counts[i,:]+=cross_DD[:]['npairs']*cross_DD[:]['weightavg']
            DD_counts[i,:] /= np.sum(dW)**2. # normalize by product of sum of weights
    print("Finished after %d seconds"%(time.time()-init))
    
else:
    # Compute xi for the periodic case (measuring mu from the Z-axis)
    print("Using periodic input data");
    from Corrfunc.theory.DDsmu import DDsmu
    
    import time
    init = time.time()
    
    # Now compute RR counts
    RR_counts, SS_counts = np.zeros([2, N_jack, nrbins * nmu_bins])    
    if len(RRname) != 0:
        RRfile = np.loadtxt(RRname) # read pre-computed RR counts
        if len(RRfile[0, :]) != (1+nrbins*nmu_bins):
            raise Exception("Incorrect number of bins in RR file. Either provide the relevant file or recompute RR pair counts for each unrestricted jackknife.")
        if len(RR_counts[:, 0]) != N_jack:
            raise Exception("Incorrect number of jackknives in RR file. Either provide the relevant file or recompute RR pair counts for each unrestricted jackknife.")
        for jk in range(N_jack):
            RR_counts[jk,:] = RRfile[jk,1:] # first index is jackknife number usually
            # NB: these are already normalized
        if not recompute_SS:
            SS_counts = RR_counts
    
    if recompute_SS or len(RRname) == 0:
        # Compute SS pair counts
        for i, j in enumerate(J_regions):
            # Compute pair counts between jackknife region and entire survey regions
            filt = np.where(rJ_SS==j)
            print("Computing RR pair counts for non-empty jackknife %d of %d"%(i+1,N_jack))
            if len(filt[0])>0:
                cross_SS = DDsmu(0,nthreads,binfile,mu_max,nmu_bins,rX_SS,rY_SS,rZ_SS,weights1=rW_SS,
                                      weight_type='pair_product',X2=rX_SS[filt],Y2=rY_SS[filt],Z2=rZ_SS[filt],
                                      weights2=rW_SS[filt],verbose=False,periodic=True)
                SS_counts[i,:] += cross_SS[:]['npairs']*cross_SS[:]['weightavg']
                SS_counts[i,:] /= np.sum(rW_SS)**2. # normalize by product of sum of weights
        if len(RRname) == 0:
            RR_counts = SS_counts
    
    print("Finished RR/SS pair counts after %d seconds"%(time.time()-init))
    
    # Now compute DR counts
    DS_counts = np.zeros_like(RR_counts)
    for i,j in enumerate(J_regions):
        print("Computing DR pair counts for non-empty jackknife %d of %d"%(i+1,N_jack))
        
        # Compute pair counts between data jackknife and random survey
        filt = np.where(dJ==j)
        if len(filt[0])>0:
            cross_DS1 = DDsmu(0,nthreads,binfile,mu_max,nmu_bins,dX[filt],dY[filt],dZ[filt],
                                   weights1=dW[filt],weight_type='pair_product', X2=rX_DS, Y2=rY_DS, 
                                   Z2 = rZ_DS, weights2 = rW_DS, verbose=False,periodic=True)
            DS_counts[i,:] += 0.5*cross_DS1[:]['npairs']*cross_DS1[:]['weightavg']
        
        # Compute pair coutnts between random jackknife and data survey
        filt2 = np.where(rJ_DS==j)
        if len(filt2[0])>0:
            cross_DS2 = DDsmu(0,nthreads,binfile,mu_max,nmu_bins,dX,dY,dZ,
                                   weights1=dW,weight_type='pair_product', X2=rX_DS[filt2], Y2=rY_DS[filt2], 
                                   Z2 = rZ_DS[filt2], weights2 = rW_DS[filt2], verbose=False,periodic=True)
            DS_counts[i,:] += 0.5*cross_DS2[:]['npairs']*cross_DS2[:]['weightavg']
        DS_counts[i,:] /= np.sum(rW_DS)*np.sum(dW) # normalize by product of sum of weights
    print("Finished DR pair counts after %d seconds"%(time.time()-init))
    
    # Now compute DD counts
    DD_counts = np.zeros_like(RR_counts)
    for i,j in enumerate(J_regions):
        # Compute pair counts between jackknife region and entire survey regions
        filt = np.where(dJ==j)
        print("Computing DD pair counts for non-empty jackknife %d of %d"%(i+1,N_jack))
        if len(filt[0])>0:
            cross_DD = DDsmu(0,nthreads,binfile,mu_max,nmu_bins,dX,dY,dZ,weights1=dW,
                                    weight_type='pair_product',X2=dX[filt],Y2=dY[filt],Z2=dZ[filt],
                                    weights2=dW[filt],verbose=False,periodic=True)
            DD_counts[i,:]+=cross_DD[:]['npairs']*cross_DD[:]['weightavg']
            DD_counts[i,:] /= np.sum(dW)**2. # normalize by product of sum of weights
    print("Finished after %d seconds"%(time.time()-init))
    

# Now compute correlation function
xi_function = np.zeros_like(RR_counts)
for j in range(N_jack):
    xi_function[j] = (DD_counts[j] - 2. * DS_counts[j] + SS_counts[j])/RR_counts[j]

# Save output files:
import os
if not os.path.exists(outdir):
    os.makedirs(outdir)

# Define mu centers
mean_mus = np.linspace(0.5/nmu_bins,1-0.5/nmu_bins,nmu_bins)

outname='xi_jack_n%d_m%d_j%d_11.dat'%(nrbins,nmu_bins,N_jack)
print("Saving correlation function")
with open(os.path.join(outdir, outname), "w+") as outfile:
    for r in mean_bins:
        outfile.write("%.8e "%r)
    outfile.write("\n")
    for mu in mean_mus:
        outfile.write("%.8e "%mu)
    outfile.write("\n");
    for i in range(N_jack):
        for j in range(nrbins*nmu_bins):
            outfile.write("%.8e "%xi_function[i,j])
        outfile.write("\n")
        
print("Correlation function written successfully to %s"%(outdir+outname))

print("NB: Number of galaxies is %d"%N_gal)

#print("ADD IN NORM")

np.savez("test_jack_xi.npz",xi_jack=xi_function,DD=DD_counts,RR=RR_counts,DR=DS_counts);
