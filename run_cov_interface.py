### Python script for running RascalC in DESI setup (Michael Rashkovetskyi, 2024).
import sys, os
import numpy as np
from astropy.table import Table, vstack
from pycorr import TwoPointCorrelationFunction, KMeansSubsampler
from pycorr.utils import sky_to_cartesian
from LSS.tabulated_cosmo import TabulatedDESI
from RascalC.pycorr_utils.utils import fix_bad_bins_pycorr
from RascalC.interface import run_cov
from RascalC.convergence_check_extra import convergence_check_extra

def prevent_override(filename: str, max_num: int = 10) -> str: # append _{number} to filename to prevent override
    for i in range(max_num+1):
        trial_name = filename + ("_" + str(i)) * bool(i) # will be filename for i=0
        if not os.path.exists(trial_name): return trial_name
    print(f"Could not prevent override of {filename}, aborting.")
    sys.exit(1)

def read_catalog(filename: str, z_min: float = -np.inf, z_max: float = np.inf, FKP_weight: bool = True):
    catalog = Table.read(filename)
    if FKP_weight: catalog["WEIGHT"] *= catalog["WEIGHT_FKP"] # apply FKP weight multiplicatively
    catalog.keep_columns(["RA", "DEC", "Z", "WEIGHT"]) # discard everything else
    filtering = np.logical_and(catalog["Z"] >= z_min, catalog["Z"] <= z_max) # logical index of redshifts within the range
    return catalog[filtering] # filtered catalog

# Mode settings

mode = "legendre_projected"
max_l = 4 # maximum (even) multipole index

njack = 60 # turns the jackknife off

periodic_boxsize = None # aperiodic

# Covariance matrix binning
nbin = 50 # number of radial bins for output cov
mbin = None # number of angular (mu) bins to use for projections, None means to keep the original number from pycorr files
skip_nbin_pre = 0 # number of first radial bins to exclude before running the C++ code
skip_nbin_post = 5 # number of first radial bins to exclude at post-processing, in addition to the above
skip_l_post = 0 # number of higher (even) multipoles to exclude at post-processing

# Input correlation function binning
nbin_cf = 100 # number of radial bins for input 2PCF
mbin_cf = 10 # number of angular (mu) bins for input 2PCF

# Settings related to time and convergence

nthread = 256 # number of OMP threads to use
n_loops = 1024 # number of integration loops per filename
loops_per_sample = 64 # number of loops to collapse into one subsample
N2 = 5 # number of secondary cells/particles per primary cell
N3 = 10 # number of third cells/particles per secondary cell/particle
N4 = 20 # number of fourth cells/particles per third cell/particle

# Settings for filenames; many are decided by the first command-line argument

version_label = "v1"
rectype = "IFFT_recsym" # reconstruction type

id = int(sys.argv[1]) # SLURM_JOB_ID to decide what this one has to do

mock_id = 1 + id // 2 # mock number, starting from 1, IDs should start from 0
id = 4 + id % 2 # this is now tracer, redshift bin and region index; corresponds to LRG z0.8-1.1 SGC/NGC

reg = "NGC" if id%2 else "SGC" # region for filenames
# known cases where more loops are needed consistently
if id in (4,): n_loops *= 2
elif id in (0, 1, 3, 15): n_loops *= 3
elif id in (2, 14): n_loops *= 4
elif id in (17,): n_loops //= 2 # QSO NGC converge well and take rather long time

id //= 2 # extracted all needed info from parity, move on
tracers = ['LRG'] * 4 + ['ELG_LOP'] * 3 + ['BGS_BRIGHT-21.5', 'QSO']
zs = [[0.4, 0.6], [0.6, 0.8], [0.8, 1.1], [0.4, 1.1], [0.8, 1.1], [1.1, 1.6], [0.8, 1.6], [0.1, 0.4], [0.8, 2.1]]
sms = [15] * 8 + [30]
ns_randoms = [4] * 7 + [1, 4] # BGS missing but presumed 1 random; others 4
# need 2 * 9 = 18 jobs in this array

tlabels = [tracers[id]] # tracer labels for filenames
sm = sms[id] # smoothing scale in Mpc/h
nrandoms = ns_randoms[id]
z_min, z_max = zs[id] # for redshift cut and filenames

# Output and temporary directories

outdir_base = os.path.join(f"mock{mock_id}/recon_sm{sm}_{rectype}", "_".join(tlabels + [reg]) + f"_z{z_min}-{z_max}")
outdir = prevent_override(os.path.join("outdirs", outdir_base)) # output file directory
tmpdir = os.path.join("tmpdirs", outdir_base) # directory to write intermediate files, kept in a different subdirectory for easy deletion, almost no need to worry about not overwriting there

# Form correlation function labels
assert len(tlabels) in (1, 2), "Only 1 and 2 tracers are supported"
corlabels = [tlabels[0]]
if len(tlabels) == 2: corlabels += ["_".join(tlabels), tlabels[1]] # cross-correlation comes between the auto-correlatons

# Common part of the path to avoid repetitions
input_dir = f"/global/cfs/cdirs/desi/survey/catalogs/Y1/mocks/SecondGenMocks/EZmock/desipipe/{version_label}/ffa/2pt/mock{mock_id}/"

# Filenames for saved pycorr counts
split_above = 20
pycorr_filenames = [[f"/pscratch/sd/m/mrash/Y1-EZmock-{version_label}-ffa/mock{mock_id}/recon_sm{sm}_{rectype}/xi/smu/allcounts_{corlabel}_{reg}_{z_min}_{z_max}_default_FKP_lin_njack{njack}_nran{nrandoms}_split20.npy"] for corlabel in corlabels]

# Filenames for randoms and galaxy catalogs
random_filenames = [[input_dir + f"recon_sm{sm}_{rectype}/{tlabel}_{reg}_{i}_clustering.ran.fits" for i in range(nrandoms)] for tlabel in tlabels]
if njack: data_ref_filenames = [input_dir + f"recon_sm{sm}_{rectype}/{tlabel}_{reg}_clustering.dat.fits" for tlabel in tlabels] # only for jackknife reference, could be used for determining the number of galaxies but not in this case

# Load pycorr counts
pycorr_allcounts = [0] * len(pycorr_filenames)
input_xis = [0] * len(pycorr_filenames)
ndata = [None] * 2
for c, pycorr_filenames_group in enumerate(pycorr_filenames):
    cumulative_ndata = 0
    for pycorr_filename in pycorr_filenames_group:
        these_counts = fix_bad_bins_pycorr(TwoPointCorrelationFunction.load(pycorr_filename)) # load and attempt to fix faulty bins using symmetry
        cumulative_ndata += these_counts.D1D2.size1 # accumulate number of data
        # reshape for covariance
        if mbin: assert these_counts.shape[1] % (2 * mbin) == 0, "Angular rebinning is not possible"
        pycorr_allcounts[c] += these_counts[::these_counts.shape[0] // nbin][skip_nbin_pre:, ::these_counts.shape[1] // 2 // mbin if mbin else 1].wrap()
        # reshape for input correlation function
        if mbin_cf: assert these_counts.shape[1] % (2 * mbin_cf) == 0, "Angular rebinning is not possible"
        input_xis[c] += these_counts[::these_counts.shape[0] // nbin_cf, ::these_counts.shape[1] // 2 // mbin_cf if mbin_cf else 1].wrap()
    if c % 2 == 0: ndata[c // 2] = cumulative_ndata / len(pycorr_filenames_group) # set the average number of data based on auto-correlatons
# add None's for missing counts
ncorr_max = 3 # maximum number of correlations
pycorr_allcounts += [None] * (ncorr_max - len(pycorr_filenames))
input_xis += [None] * (ncorr_max - len(pycorr_filenames))

# Load randoms and galaxy catalogs
cosmology = TabulatedDESI() # for conversion from RA,DEC,Z to Cartesian
ntracers_max = 2 # maximum number of tracers
randoms_positions = [None] * ntracers_max
randoms_weights = [None] * ntracers_max
randoms_samples = [None] * ntracers_max
for t in range(len(tlabels)):
    # read randoms with redshift cut
    random_catalog = vstack([read_catalog(random_filename, z_min = z_min, z_max = z_max) for random_filename in random_filenames[t]])
    randoms_weights[t] = random_catalog["WEIGHT"]
    # create jackknives
    if njack:
        data_catalog = read_catalog(data_ref_filenames[t], z_min = z_min, z_max = z_max)
        subsampler = KMeansSubsampler('angular', positions = [data_catalog["RA"], data_catalog["DEC"], data_catalog["Z"]], position_type = 'rdd', nsamples = njack, nside = 512, random_state = 42)
        randoms_samples[t] = subsampler.label(positions = [random_catalog["RA"], random_catalog["DEC"], random_catalog["Z"]], position_type = 'rdd')
    # convert to comoving Cartesian coordinates
    comoving_dist = cosmology.comoving_radial_distance(random_catalog["Z"])
    randoms_positions[t] = np.column_stack(sky_to_cartesian([random_catalog["RA"], random_catalog["DEC"], comoving_dist], degree = True))

# Run the main code and post-processing
results = run_cov(mode = mode, max_l = max_l, boxsize = periodic_boxsize,
                  nthread = nthread, N2 = N2, N3 = N3, N4 = N4, n_loops = n_loops, loops_per_sample = loops_per_sample,
                  pycorr_allcounts_11 = pycorr_allcounts[0], pycorr_allcounts_12 = pycorr_allcounts[1], pycorr_allcounts_22 = pycorr_allcounts[2],
                  xi_table_11 = input_xis[0], xi_table_12 = input_xis[1], xi_table_22 = input_xis[2],
                  no_data_galaxies1 = ndata[0], no_data_galaxies2 = ndata[1],
                  randoms_positions1 = randoms_positions[0], randoms_weights1 = randoms_weights[0], randoms_samples1 = randoms_samples[0],
                  randoms_positions2 = randoms_positions[1], randoms_weights2 = randoms_weights[1], randoms_samples2 = randoms_samples[1],
                  normalize_wcounts = True,
                  out_dir = outdir, tmp_dir = tmpdir,
                  skip_s_bins = skip_nbin_post, skip_l = skip_l_post)

# Additional convergence check
convergence_check_extra(results, print_function = print)