# Set the filenames
galaxies_filename = "mock_galaxy_DR12_CMASS_N_QPM_0001.rdzw"
randoms_filename = "mock_random_DR12_CMASS_N_50x1.rdzw"


# Read the original files: text with RA, DEC, Z (redshift) and weight columns.
import numpy as np
from astropy.table import Table
galaxies = Table(np.loadtxt(galaxies_filename, usecols = range(4)), names = ["RA", "DEC", "Z", "WEIGHT"]) # ignore the last column, not sure what it is
randoms = Table(np.loadtxt(randoms_filename, usecols = range(4)), names = ["RA", "DEC", "Z", "WEIGHT"])


# Compute the comoving distance within the fiducial (grid) cosmology. Here we use a utility function from the RascalC library to do this.
from RascalC.pre_process.convert_to_xyz import comoving_distance_Mpch
Omega_m = 0.29; Omega_k = 0; w_DE = -1 # density parameters of matter and curvature, and the equation-of-state parameter of dark energy
galaxies["comov_dist"] = comoving_distance_Mpch(galaxies["Z"], Omega_m, Omega_k, w_DE)
randoms["comov_dist"] = comoving_distance_Mpch(randoms["Z"], Omega_m, Omega_k, w_DE)


# Let us define a utility function for position formatting that will be useful on several more occasions.
def get_rdd_positions(catalog: Table) -> tuple[np.ndarray[float]]: # utility function to format positions from a catalog
    return (catalog["RA"], catalog["DEC"], catalog["comov_dist"])


# Assign jackknife regions to both galaxies and randoms.
from RascalC.pre_process.create_jackknives_pycorr import get_subsampler_xirunpc
n_jack = 60 # number of regions
subsampler = get_subsampler_xirunpc(get_rdd_positions(galaxies), n_jack, position_type = "rdd") # "rdd" means RA, DEC in degrees and then distance (corresponding to pycorr)
galaxies["JACK"] = subsampler.label(get_rdd_positions(galaxies), position_type = "rdd")
randoms["JACK"] = subsampler.label(get_rdd_positions(randoms), position_type = "rdd")


# Select a smaller subset of randoms to make pair counting and `RascalC` importance sampling more feasible
x_randoms = 10 # how many times the number of galaxies should the number of randoms be; the total number of randoms is ≈50x the number of galaxies
np.random.seed(42) # for reproducibility
randoms_subset = randoms[np.random.choice(len(randoms), x_randoms * len(galaxies), replace = False, p = randoms["WEIGHT"] / np.sum(randoms["WEIGHT"]))]


# ## Pair counts and correlation functions with [`pycorr`](https://github.com/cosmodesi/pycorr)

# First, we choose whether to use full randoms or a smaller subset for pair counting.
# The latter is faster but a bit less precise.
# randoms_for_counts = randoms
randoms_for_counts = randoms_subset


# We continue by splitting the randoms into parts of roughly the same size as data.
# This gives high precision at fixed computing cost [(Keihänen et al 2019)](https://arxiv.org/abs/1905.01133).
n_splits = int(np.rint(len(randoms_for_counts) / len(galaxies))) # the number of parts to split the randoms to
print(f"Splitting randoms into {n_splits} parts")

# split randoms into the desired number of parts randomly
random_indices = np.arange(len(randoms_for_counts))
np.random.seed(42) # for reproducibility
np.random.shuffle(random_indices) # random shuffle in place
random_parts = [randoms_for_counts[random_indices[i_random::n_splits]] for i_random in range(n_splits)] # quick way to produce parts of almost the same size

# normalize the weights in each part — fluctuations in their sums may be a bit of a problem
for i_random in range(n_splits): random_parts[i_random]["WEIGHT"] /= np.sum(random_parts[i_random]["WEIGHT"])


# Import libraries and select settings
from pycorr import TwoPointCorrelationFunction, setup_logging
from tqdm import trange # nice progress bar
import os


n_threads = 10 # number of threads for pycorr computation
s_max = 200 # maximal separation in Mpc/h
n_mu = 200 # number of angular (µ) bins
counts_filename = f"allcounts_mock_galaxy_DR12_CMASS_N_QPM_0001_lin_njack{n_jack}_nran{n_splits}_split{0}.npy" # filename to save counts


s_edges = np.arange(s_max + 1) # 1 Mpc/h wide separation bins from 0 to s_max Mpc/h
mu_edges = np.linspace(-1, 1, n_mu + 1) # make uniform µ bins between -1 and 1, or twice less bins between 0 and 1 after wrapping (will be done within RascalC wrapper)

def run_pair_counts(): # the code to run pycorr, needs to be invoked in a separate process from what will run RascalC then!
    setup_logging()
    result = 0
    D1D2 = None # to compute the data-data counts on the first go but not recompute then
    for i_random in trange(n_splits, desc="Computing counts with random part"):
        these_randoms = random_parts[i_random]
        tmp = TwoPointCorrelationFunction(mode = 'smu', edges = (s_edges, mu_edges),
                                        data_positions1 = get_rdd_positions(galaxies), data_weights1 = galaxies["WEIGHT"], data_samples1 = galaxies["JACK"],
                                        randoms_positions1 = get_rdd_positions(these_randoms), randoms_weights1 = these_randoms["WEIGHT"], randoms_samples1 = these_randoms["JACK"],
                                        position_type = "rdd", engine = "corrfunc", D1D2 = D1D2, gpu = False, nthreads = n_threads)
        # "rdd" means RA, DEC in degrees and then distance
        D1D2 = tmp.D1D2 # once computed, becomes not None and will not be recomputed
        result += tmp
    result.D1D2.attrs['nsplits'] = n_splits

    result.save(counts_filename)


# Next is a tricky part where we create a child process to run pair counts to avoid OpenMP multi-threading interference.
# 
# The computation does take a while (about 10 minutes for me at NERSC login node; expect to see 1/10 progress in a few minutes) even with multi-threading (which I haven't managed to make work with `Corrfunc` in macOS yet).
# If you already have the counts saved into file, no need to run this cell again – they will be loaded later.
# 
# It is **best not to interrupt the next cell** because it is rather hard to control the forked child process.

# In[11]:


child_pid = os.fork() # creates a new process at the same execution stage with access to all the data. Returns 0 in child process and non-zero child process ID in the original, parent process
if not child_pid:
    # new child process
    print(f"Child process ID {os.getpid()}") # printing from the parent process may interfere with the progress bar
    import traceback
    try:
        run_pair_counts()
    except Exception:
        traceback.print_exc()
        os._exit(-1) # terminate the process with error
    # in a script, would not need try-except and import traceback above. This is because script process is already terminated on error/exception with a non-zero exit code, in notebook it remains to hang around
    os._exit(0) # terminate ok if no error occured
else:
    # original parent process
    child_pid2, child_status = os.waitpid(child_pid, 0) # wait for the child process to terminate
    if child_status: raise RuntimeError(f"Child process exited with error (code {os.waitstatus_to_exitcode(child_status)}). See its traceback above.")


# Finally, we should load the saved counts in the original process. You can continue from this step if you computed the counts before.
allcounts = TwoPointCorrelationFunction.load(counts_filename)


# ## Covariance settings and computation

# Mode parameters
mode = "legendre_projected" # Legendre multipoles projected from s,µ bins
max_l = 4 # max multipole to compute the covariance for
periodic_boxsize = None # not a periodic box


# Runtime and convergence parameters
n_threads = 10
N2 = 10 # number of secondary points sampled per primary random point
N3 = 20 # number of tertiary points sampled per each secondary
N4 = 40 # number of quaternary points sampled per each tertiary
n_loops = 80 # must be divisible by n_threads
loops_per_sample = 5 # must divide n_loops


# In the next cell we set the radial/separation binning of the covariance matrix by rebinning the counts. Here we leave the original number of angular bins – the covariance will be project into only a few (even) multipoles using all of them.
ds_cov = 4
s_min_cov = 20
s_max_cov = 200
allcounts_rebinned_cov = allcounts[s_min_cov:s_max_cov:ds_cov]


# Also rebin the pycorr counts into reasonable (typically a bit finer) bins for the input two-point correlation function.
ds_xi = 2
n_mu_xi = 20 # between µ = 0 and 1, i.e. after wrapping
assert allcounts.wrap().shape[1] % n_mu_xi == 0, "Counts not rebinnable to the desired number of angular bins"
allcounts_rebinned_xi = allcounts[::ds_xi, ::allcounts.wrap().shape[1] // n_mu_xi]


# Output and temporary directories
outdir = "out"
tmpdir = "tmp"


# ### The main covariance computation with `RascalC` interface
from RascalC import run_cov
results = run_cov(mode = mode, max_l = max_l, boxsize = periodic_boxsize,
                  nthread = n_threads, N2 = N2, N3 = N3, N4 = N4, n_loops = n_loops, loops_per_sample = loops_per_sample,
                  pycorr_allcounts_11 = allcounts_rebinned_cov,
                  xi_table_11 = allcounts_rebinned_xi,
                  no_data_galaxies1 = len(galaxies),
                  position_type = "rdd",
                  randoms_positions1 = get_rdd_positions(randoms_subset), randoms_weights1 = randoms_subset["WEIGHT"], randoms_samples1 = randoms_subset["JACK"],
                  normalize_wcounts = True,
                  out_dir = outdir, tmp_dir = tmpdir)