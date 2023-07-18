# This script generates all covs

import os

max_l = 4
nbin = 50 # radial bins for output cov
rmax = 200 # maximum output cov radius in Mpc/h

version_label = "v0.4"
rectype = "IFTrecsym" # reconstruction type

regs = ('SGC', 'NGC') # regions for filenames
reg_comb = "GCcomb"

tracers = ['LRG'] * 4 + ['ELG_LOPnotqso'] * 3 + ['BGS_BRIGHT-21.5', 'QSO']
zs = [[0.4, 0.6], [0.6, 0.8], [0.8, 1.1], [0.4, 1.1], [0.8, 1.1], [1.1, 1.6], [0.8, 1.6], [0.1, 0.4], [0.8, 2.1]]
sms = [10] * 7 + [15] * 2

skip_bins = 5
skip_l = 0
shot_noise_rescaling = 1

maxloops = 2048 # number of integration loops per filename
loopspersample = 256 # number of loops to collapse into one subsample

nrandoms = 4
split_above = 20

xilabel = "".join([str(i) for i in range(0, max_l+1, 2)])

nfiles = nrandoms
no_subsamples_per_file = maxloops // loopspersample
n_subsamples = no_subsamples_per_file * nfiles

r_step = rmax // nbin
nbin_final = nbin - skip_bins
rmin_real = r_step * nbin_final

def my_make(goal, deps, *cmds, force=False):
    if force or need_make(goal, deps):
        print(f"Making {goal} from {deps}")
        for cmd in cmds:
            ret = exec_function(cmd)
            if ret:
                print(f"{cmd} exited with error (code {ret}). Aborting")
                return

def need_make(goal, srcs):
    src_mtime = float('-inf')
    for src in srcs:
        if not os.path.exists(src):
            print(f"Can not make {goal} from {srcs}: {src} missing")
            return False
        src_mtime = max(src_mtime, os.path.getmtime(src))
    if not os.path.exists(goal): return True
    dest_mtime = os.path.getmtime(goal)
    if src_mtime < dest_mtime:
        print(f"{goal} is newer than {srcs}, not making")
        return False
    return True

def exec_function(cmdline): # common function to invoke other processes
    print(f"Running command: {cmdline}")
    return os.system(cmdline) # simple now but could be changed quickly later

cov_names = []
# Make steps for making covs
for tracer, (z_min, z_max), sm in zip(tracers, zs, sms):
    tlabels = [tracer]
    reg_results, reg_pycorr_names = [], []
    for reg in regs:
        outdir = os.path.join(f"recon_sm{sm}", "_".join(tlabels + [rectype, reg]) + f"_z{z_min}-{z_max}") # output file directory
        first_output_name = os.path.join(outdir, "CovMatricesAll/0/c4_n%d_l%d_11,11_0.txt" % (nbin, max_l))
        full_output_name = os.path.join(outdir, "CovMatricesAll/c4_n%d_l%d_11,11_full.txt" % (nbin, max_l))
        results_name = os.path.join(outdir, 'Rescaled_Covariance_Matrices_Legendre_n%d_l%d.npz' % (nbin, max_l))
        reg_results.append(results_name)
        cov_name = "xi" + xilabel + "_" + "_".join(tlabels + [reg, z_min, z_max]) + f"_default_FKP_lin{r_step}_s{rmin_real}-{rmax}_cov_RascalC_Gaussian.txt"
        cov_names.append(cov_name)
        reg_pycorr_names.append(f"/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/{version_label}/blinded/recon_sm{sm}/xi/smu/allcounts_{tracer}_{rectype}_{reg}_{z_min}_{z_max}_default_FKP_lin_njack{0}_nran{nrandoms}_split{split_above}.npy")

        # Full output depends on first output name
        my_make(full_output_name, [first_output_name], f"python python/cat_subsets_of_integrals.py {nbin} l{max_l} " + " ".join([f"{os.path.join(outdir, str(i))} {no_subsamples_per_file}" for i in range(nfiles)]) + f" {outdir}")
        # Recipe: run subsample catenation

        # RascalC results depend on full output
        my_make(results_name, [full_output_name], f"python python/post_process_legendre.py {outdir} {nbin} {max_l} {n_subsamples} {outdir} {shot_noise_rescaling} {skip_bins} {skip_l}", f"python python/convergence_check_extra.py {results_name}")
        # Recipe: run post-processing
        # Also perform convergence check (optional but nice)

        # Individual cov file depends on RascalC results
        my_make(cov_name, [results_name], f"python python/convert_cov_legendre.py {results_name} {nbin_final} {cov_name}")
        # Recipe: run convert cov
    
    cov_name = "xi" + xilabel + "_" + "_".join(tlabels + [reg_comb, z_min, z_max]) + f"_default_FKP_lin{r_step}_s{rmin_real}-{rmax}_cov_RascalC_Gaussian.txt" # combined cov name

    # Comb cov depends on the region RascalC results
    my_make(cov_name, reg_results, "python python/combine_covs_legendre.py " + " ".join(reg_results) + " " + " ".join(reg_pycorr_names) + f" {nbin} {max_l} {skip_bins} {cov_name}")
    # Recipe: run combine covs