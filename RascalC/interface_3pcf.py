"""
Python wrapper of ``RascalC``, heavily interfaced with ``pycorr`` `library for 2-point correlation function estimation <https://github.com/cosmodesi/pycorr>`_.
Many of the arguments are intentionally similar to ``pycorr.TwoPointCorrelationFunction`` `high-level interface <https://py2pcf.readthedocs.io/en/latest/api/api.html#pycorr.correlation_function.TwoPointCorrelationFunction>`_.

Please bear with the long description; you can pay less attention to settings labeled optional in the beginning.
"""

import pycorr
import numpy as np
import os
from datetime import datetime
from typing import Iterable, Literal
from warnings import warn
from .pycorr_utils.utils import fix_bad_bins_pycorr, write_xi_file
from .write_binning_file import write_binning_file
from .pycorr_utils.input_xi import get_input_xi_from_pycorr
from correction_function_3pcf import compute_3pcf_correction_function
from .convergence_check_extra import convergence_check_extra
from .utils import rmdir_if_exists_and_empty, suffixes_tracer_all, indices_corr_all, suffixes_corr_all
from .post_process_3pcf import post_process_3pcf


def run_cov(mode: Literal["legendre_accumulated"],
            s_edges: np.ndarray[float], max_l: int,
            nthread: int, N2: int, N3: int, N4: int, N5: int, N6: int, n_loops: int, loops_per_sample: int,
            out_dir: str, tmp_dir: str,
            randoms_positions1: np.ndarray[float], randoms_weights1: np.ndarray[float],
            xi_table_11: pycorr.twopoint_estimator.BaseTwoPointEstimator | tuple[np.ndarray[float], np.ndarray[float], np.ndarray[float]] | tuple[np.ndarray[float], np.ndarray[float], np.ndarray[float], np.ndarray[float], np.ndarray[float]] | list[np.ndarray[float]],
            no_data_galaxies1: float,
            RRR_counts: np.ndarray[float] | None = None, # probably plug it from ENCORE/CADENZA when available and otherwise compute internally with triple_counts
            n_mu_bins: int = 120,
            position_type: Literal["rdd", "xyz", "pos"] = "pos",
            xi_cut_s: float = 250, xi_refinement_iterations: int = 10,
            normalize_wcounts: bool = True,
            boxsize: float | None = None,
            shot_noise_rescaling1: float = 1,
            sampling_grid_size: int = 301, coordinate_scaling: float = 1, seed: int | None = None,
            verbose: bool = False) -> dict[str, np.ndarray[float]]:
    r"""
    Run the 3-point correlation function covariance integration.
    Only supports single tracer in "accumulated" Legendre mode without jackknives.

    Parameters
    ----------
    mode : string
        Choice of binning setup, only one is supported now:

            - ``"legendre_accumulated"``: compute covariance of the 3-point correlation function Legendre multipoles in pairs separation (s) bins accumulated directly, without first doing :math:`\mu`-binned counts.
    
    s_edges : Numpy array of floats
        Separation/radial bin edges.
    
    max_l : integer
        Max Legendre multipole index (required in Legendre ``mode``\s).
        Can be odd.
    
    boxsize : None or float
        Periodic box side (one number — so far, only cubic boxes are supported).
        All the coordinates need to be between 0 and ``boxsize``.
        If None (default) or 0, assumed aperiodic.

    position_type : string, default="pos"
        Type of input positions, one of:

            - "rdd": RA, Dec (both in degrees), distance; shape (3, N)
            - "xyz": Cartesian positions, shape (3, N)
            - "pos": Cartesian positions, shape (N, 3).
    
    randoms_positions1 : array of floats, shaped according to `position_type`
        Coordinates of random points for the first tracer.
    
    randoms_weights1 : array of floats of length N_randoms
        Weights of random points for the first tracer.

    normalize_wcounts : boolean
        (Optional) whether to normalize the weights and weighted counts.
        If False, the provided RR counts must match what can be obtained from given randoms, otherwise the covariance matrix will be off by a constant factor.
        Example: if counts were computed with ``n_randoms`` roughly similar random chunks and only one is provided to RascalC here, the counts should be divided by ``n_random`` where ``s > split_above`` and by ``n_random ** 2`` where ``s < split_above``.
        If True (default), the weights will be normalized so that their sum is 1 and the counts will be normalized by their ``wnorm``, which gives a match with default ``pycorr`` normalization settings.
    
    no_data_galaxies1 : float
        Number of first tracer data (not random!) points for the covariance rescaling.
    
    RRR_counts : Numpy array of floats, or None
        (Optional) RRR (random triplet) counts in ENCORE format.
        If not provided and the data is not in a periodic box, triple counts will be estimated with importance sampling (expect longer runtime).
        In case of periodic box, the RRR counts are not needed because they are trivial.
    
    n_mu_bins : integer
        (Optional) number of angular (mu) bins for the RRR (random triplet) counts computation. Default 120.
        If the RRR counts are provided in ENCORE format, this parameter has no effect.
    
    xi_table_11 : :class:`pycorr.TwoPointEstimator`, or sequence (tuple or list) of 3 elements: ``(s_values, mu_values, xi_values)``, or sequence (tuple or list) of 4 elements: ``(s_values, mu_values, xi_values, s_edges)``
        Table of first tracer auto-correlation function in separation (s) and :math:`\mu` bins.
        The code will use it for interpolation in the covariance matrix integrals.
        Important: if the given correlation function is an average in :math:`(s, \mu)` bins, the separation bin edges need to be provided (and the :math:`\mu` bins are assumed to be linear) for rescaling procedure which ensures that the interpolation results averaged over :math:`(s, \mu)` bins returns the given correlation function. In case of ``pycorr.TwoPointEstimator``, the edges will be recovered automatically. Unwrapped estimators (:math:`\mu` from -1 to 1) are preferred, because symmetry allows to fix some issues.
        In the sequence format:

            - ``s_values`` must be a 1D array of reference separation (s) values for the table, of length N;
            - ``mu_values`` must be a 1D array of reference :math:`\mu` values (covering the range from 0 to 1) for the table, of length M;
            - ``xi_values`` must be an array of correlation function values at those :math:`(s, \mu)` values of shape (N, M);
            - ``s_edges``, if given, must be a 1D array of separation bin edges of length N+1. The bins must come close to zero separation (say start at ``s <= 0.01``).
        
        The sequence containing 3 elements should be used for theoretical models evaluated at a grid of s, mu values.
        The 4-element format should be used for bin-averaged estimates (unless they are in a :class:`pycorr.TwoPointEstimator`).
    
    xi_cut_s : float
        (Optional) separation value beyond which the correlation function is assumed to be zero for the covariance matrix integrals. Default: 250.
        Between the maximum separation from ``xi_table``\s and ``xi_cut_s``, the correlation function is extrapolated as :math:`\propto s^{-4}`.
    
    xi_refinement_iterations : integer
        (Optional) number of iterations in the correlation function refinement procedure for interpolation inside the code, ensuring that the bin-averaged interpolated values match the binned correlation function estimates. Default: 10.
        Important: the refinement procedure is disabled completely regardless of this setting if the ``xi_table``\s are sequences of 3 elements, since they are presumed to be a theoretical model evaluated at a grid of s, mu values and not bin averages.

    nthread : integer
        Number of (OpenMP) threads to use.
        Can not utilize more threads than ``n_loops``.

            - IMPORTANT: AVOID multi-threading in the Python process calling this function (e.g. at NERSC this would mean not setting ``OMP_*`` and other ``*_THREADS`` environment variables; the code should be able to set them by itself). Otherwise the code may run effectively single-threaded. If you need other multi-threaded calculations, run them separately or spawn sub-processes.
    
    N2 : integer
        Number of secondary points to sample per each primary random point.
        Setting too low (below 5) is not recommended.
    
    N3 : integer
        Number of tertiary points to sample per each secondary point.
        Setting too low (below 5) is not recommended.
    
    N4 : integer
        Number of quaternary points to sample per each tertiary point.
        Setting too low (below 5) is not recommended.
    
    N5 : integer
        Number of quinary points to sample per each quaternary point.
        Setting too low (below 5) is not recommended.
    
    N6 : integer
        Number of senary points to sample per each quinary point.
        Setting too low (below 5) is not recommended.

    n_loops : integer
        Number of integration loops.
        For optimal balancing and minimal idle time, should be a few times (at least twice) ``nthread`` and exactly divisible by it.
        The runtime roughly scales as the number of quads per the number of threads, :math:`\mathcal{O}`\(``N_randoms * N2 * N3 * N4 * n_loops / nthread``).
        For reference, on NERSC Perlmutter CPU half-node the code processed about 27 millions (``2.7e7``) quads per second per thread (using 64 threads on half a node) as of December 2024. (In Legendre projected mode, which is probably the slowest, with ``N2 = 5``, ``N3 = 10``, ``N4 = 20``.)
        In single-tracer mode, the number of quads is ``N_randoms * N2 * N3 * N4 * n_loops``.
        In two-tracer mode, the number of quads is ``(5 * N_randoms1 + 2 * N_randoms2) * N2 * N3 * N4 * n_loops``.

    loops_per_sample : integer
        Number of loops to merge into one output sample.
        Must divide ``max_loops``.
        Recommended to keep the number of samples = ``n_loops / loops_per_sample`` roughly between 10 and 30.

    out_dir : string
        Directory for important outputs.
        Moderate disk space required (up to a few hundred megabytes), but increases with covariance matrix size and number of samples (see above).

    tmp_dir : string
        Directory for temporary files. Can be deleted after running the code, and is cleaned up after the normal execution.
        More disk space required - needs to store all the input arrays in the current implementation.

    skip_s_bins : integer or tuple of two integers
        (Optional) removal of separations bins at the post-processing stage.
        First (or the only) number sets the number of radial/separation bins to skip from the beginning (lowest-separation bins tend to converge worse and probably will not be precise due to the limitations of the formalism).
        Second number (if provided) sets the number of radial/separation bins to skip from the end.
        By default, no bins are skipped at the post-processing stage.

    skip_l : integer
        (Only for the Legendre modes; optional) number of highest (even) multipoles to skip at the post-processing stage. (Higher multipole moments of the correlation function tend to converge worse.) Default 0 (no skipping).

    shot_noise_rescaling1 : float
        (Optional) shot-noise rescaling value for the first tracer if known beforehand. Default 1 (no rescaling).

    seed : integer or None
        (Optional) If given as an integer, sets the base RNG (random number generator) seed, allowing to reproduce the results with the same input data and settings (except the number of threads, which can be varied).
        Individual subsamples (and accordingly the intrinsic precision estimates and convergence test results) may differ (slightly) because they are accumulated/written in order of loop completion which may depend on external factors at runtime, but the final integrals should be the same.
        If None (default), the initialization will be random. The randomly generated seed value can be found afterwards in the log file and/or output after ``the base RNG seed is``, but using the same seed might not reproduce the runs without a preset seed using the code before commit fd2d2c41 (12 June 2025).
        Note that False in Python is equivalent to 0, which is a legitimate RNG seed, and therefore falls under the integer case, not like None. (True is equivalent to 1.)

    sampling_grid_size : integer
        (Optional) first guess for the sampling grid size.
        The code should be able to find a suitable number automatically.

    verbose : bool
        (Optional) report each 5% of each loop's progress by printing. Default False (off).
        This can be a lot of output, only use when the number of loops is small.

    coordinate_scaling : float
        (Optional) scaling factor for all the Cartesian coordinates. Default 1 (no rescaling).
        This option is supported by the C++ code, but its use cases are not very clear.
        Zero or negative value is reset to ``boxsize``, rescaling an unit cube to full periodicity.

    Returns
    -------
    post_processing_results : dict[str, np.ndarray[float]]
        Post-processing results as a dictionary with string keys and Numpy array values. All this information is also saved in a ``Rescaled_Covariance_Matrices*.npz`` file in the output directory.
        Selected common keys are: ``"full_theory_covariance"`` for the final covariance matrix and ``"shot_noise_rescaling"`` for the shot-noise rescaling value(s).
        There will also be a ``Raw_Covariance_Matices*.npz`` file in the output directory (as long as the C++ code has run without errors), which can be post-processed separately in a different way using e.g. :func:`RascalC.post_process_auto`.
        For convenience, in the output dictionary only, ``"filename"`` contains the exact name of this file, and ``"path"`` contains path to it (also obtainable by :func:`os.path.join`-ing ``out_dir`` with the filename)
    """

    if mode not in ("legendre_accumulated",): raise ValueError("Given mode not supported")

    if not isinstance(max_l, int): raise TypeError("Max ell must be an integer")
    if max_l < 0: raise ValueError("Max ell must not be negative")

    # Set some other flags
    periodic = bool(boxsize) # False for None (default) and 0
    
    if periodic and boxsize < 0: raise ValueError("Periodic box size must be positive")
    
    if n_loops % loops_per_sample != 0: raise ValueError("The sample collapsing factor must divide the number of loops")

    if not isinstance(RRR_counts, (np.ndarray, type(None))): raise TypeError("RRR_counts must be a Numpy array or None")

    if not isinstance(seed, (int, type(None))): raise TypeError("Seed must be an integer or None")

    if no_data_galaxies1 <= 0: raise ValueError("Number of data galaxies (no_data_galaxies1) must be positive")

    ntracers = 1
    ncorr = ntracers * (ntracers + 1) // 2
    suffixes_tracer = suffixes_tracer_all[:ntracers]
    indices_corr = indices_corr_all[:ncorr]
    suffixes_corr = suffixes_corr_all[:ncorr]

    n_r_bins = len(s_edges) - 1

    # this condition should be updated for 3PCF, but it is not top priority
    # if periodic and 2 * (max(s_edges) + xi_cut_s) > boxsize:
    #     warn("Some of the interparticle distances may not be correctly periodically wrapped because of the small box period, so some 6-point configurations may be missed in error. To avoid this, keep the sum of s_max (maximum separation in the covariance bins) and the xi cutoff scale smaller than half of the box size.")

    # set the technical filenames
    input_filenames = [os.path.join(tmp_dir, str(t) + ".txt") for t in range(ntracers)]
    cornames = [os.path.join(out_dir, f"xi/xi_{index}.dat") for index in indices_corr]
    # RRR_name = 
    # inv_phi_name = # inverse correction function # [os.path.join(out_dir, f"BinCorrectionFactor_n{n_r_bins}_" + ("periodic" if periodic else f'm{n_mu_bins}') + f"_{index}.txt") for index in indices_corr]
    
    # make sure the dirs exist
    # os.makedirs() will become confused if the path elements to create include pardir (eg. “..” on UNIX systems).
    # os.path.abspath should prevent this
    out_dir_safe = os.path.abspath(out_dir)
    os.makedirs(out_dir_safe, exist_ok = True)
    os.makedirs(os.path.join(out_dir_safe, "xi"), exist_ok = True)
    os.makedirs(os.path.join(out_dir_safe, "weights"), exist_ok = True)

    # before creating the temporary directory, find which of its parent directories do not exist (yet), to remove them once they are not needed, but only if they are empty
    tmp_max_iterations = 100
    tmp_dirs_to_clean_up = []
    tmp_path = os.path.abspath(tmp_dir) # start walking up the path from the temporary directory; this should also remove the slash at the end of tmp_dir if it was there
    for _ in range(tmp_max_iterations): # this could be a "while not" loop, but I wanted to prevent a possibility of an endless/unreasonably long loop
        if os.path.exists(tmp_path): break # terminate the loop when we found a pre-existing directory
        tmp_dirs_to_clean_up.append(tmp_path) # if we reach this, this directory does not exist, add it to the cleanup list
        tmp_path = os.path.dirname(tmp_path) # should always jump to the parent directory
    else: # this is when the loop executes all tmp_max_iterations iterations without encountering the break statement, i.e. potentially becomes infinite even though it shouldn't
        tmp_dirs_to_clean_up = [os.path.abspath(tmp_dir)] # reset the cleanup list to the temporary directory only (the old default behavior)
        warn(f"The check of the newly created temporary directory path components failed to finish in {tmp_max_iterations} steps. This is not critical, but should not happen. Please inform the RascalC maintainer of this occurrence.")
    os.makedirs(os.path.abspath(tmp_dir), exist_ok = True) # finally, create the temporary dir and all its missing parents
    
    # Create a log file in output directory
    logfilename = "log.txt"
    logfile = os.path.join(out_dir, logfilename)

    def print_log(s: object) -> None:
        os.system(f"echo \"{s}\" >> {logfile}")

    def print_and_log(s: object) -> None:
        print(s)
        print_log(s)
    
    print_and_log(f"Mode: {mode}")
    print_and_log(f"Periodic box: {periodic}")
    if periodic: print_and_log(f"Box side: {boxsize}")
    print_and_log(f"Normalizing weights and weighted counts: {normalize_wcounts}")
    print_and_log(datetime.now())

    ndata = [no_data_galaxies1]

    print_and_log(f"Number(s) of data galaxies: {ndata}")
    
    # write the xi file(s); need to set the 2PCF binning (even if only technical) and decide whether to rescale the 2PCF in the C++ code
    all_xi = (xi_table_11,)
    xi_s_edges = None
    xi_n_mu_bins = None
    refine_xi = False
    for c, xi in enumerate(all_xi[:ncorr]):
        if c > 0:
            if type(xi) != type(all_xi[0]): raise TypeError(f"xi_table_{indices_corr[c]} must have the same type as xi_table_11")
            if xi.shape != all_xi[0].shape: raise ValueError(f"xi_table_{indices_corr[c]} must have the same shape as xi_table_11")
        if isinstance(xi, pycorr.twopoint_estimator.BaseTwoPointEstimator):
            refine_xi = True
            if xi.edges[1][0] < 0:
                xi = fix_bad_bins_pycorr(xi)
                print_and_log(f"Wrapping xi_table_{indices_corr[c]} to mu>=0")
                xi = xi.wrap()
            if c == 0:
                xi_n_mu_bins = xi.shape[1]
                xi_s_edges = xi.edges[0]
            elif not np.allclose(xi_s_edges, xi.edges[0]): raise ValueError("Different binning for different correlation functions not supported")
            if not np.allclose(xi.edges[1], np.linspace(0, 1, xi_n_mu_bins + 1)): raise ValueError(f"xi_table_{indices_corr[c]} mu binning is not consistent with linear between 0 and 1 (after wrapping)")
            write_xi_file(cornames[c], xi.sepavg(axis = 0), xi.sepavg(axis = 1), get_input_xi_from_pycorr(xi))
        elif isinstance(xi, Iterable):
            if len(xi) == 4: # the last element is the edges
                refine_xi = True
                if c == 0: xi_s_edges = xi[-1]
                elif not np.allclose(xi_s_edges, xi[-1]): raise ValueError("Different binning for different correlation functions not supported")
                xi = xi[:-1]
            if len(xi) != 3: raise ValueError(f"xi_table {indices_corr[c]} must have 3 or 4 elements if a tuple/list")
            r_vals, mu_vals, xi_vals = xi
            if len(xi_vals) != len(r_vals): raise ValueError(f"xi_values {indices_corr[c]} must have the same number of rows as r_values")
            if len(xi_vals[0]) != len(mu_vals): raise ValueError(f"xi_values {indices_corr[c]} must have the same number of columns as mu_values")
            if c == 0:
                xi_n_mu_bins = len(mu_vals)
                if not refine_xi:
                    xi_s_edges = (r_vals[:-1] + r_vals[1:]) / 2 # middle values as midpoints of r_vals to be safe
                    xi_s_edges = [1e-4] + xi_s_edges + [2 * r_vals[-1] - xi_s_edges[-1]] # set the lowest edge near 0 and the highest beyond the last point of r_vals
            write_xi_file(cornames[c], r_vals, mu_vals, xi_vals)
        else: raise TypeError(f"Xi table {indices_corr[c]} must be either a pycorr.TwoPointEstimator or a tuple/list")
    xi_refinement_iterations *= refine_xi # True is 1; False is 0 => 0 iterations => no refinement
    
    # write the randoms file(s)
    randoms_positions = [randoms_positions1]
    randoms_weights = [randoms_weights1]
    for t, input_filename in enumerate(input_filenames):
        randoms_properties = pycorr.twopoint_counter._format_positions(randoms_positions[t], mode = "smu", position_type = position_type, dtype = np.float64) # list of x, y, z coordinate arrays; weights (and jackknife region numbers if any) will be appended
        randoms_positions[t] = np.array(randoms_properties) # save the formatted positions as an array for correction function computation
        nrandoms = len(randoms_properties[0])
        if randoms_weights[t].ndim != 1: raise ValueError(f"Weights of randoms {t+1} not contained in a 1D array")
        if len(randoms_weights[t]) != nrandoms: raise ValueError(f"Number of weights for randoms {t+1} mismatches the number of positions")
        if normalize_wcounts: randoms_weights[t] /= np.sum(randoms_weights[t])
        randoms_properties.append(randoms_weights[t])
        np.savetxt(input_filename, np.column_stack(randoms_properties))
        randoms_properties = None

    # write the binning files
    binfile = os.path.join(out_dir, "radial_binning_cov.csv")
    write_binning_file(binfile, s_edges)
    binfile_cf = os.path.join(out_dir, "radial_binning_corr.csv")
    write_binning_file(binfile_cf, xi_s_edges)

    counts_factor = None if normalize_wcounts else 1

    # deal with RRR counts
    # need to check normalize_wcounts logic
    if RRR_counts is None:
        if periodic: RRR_filename = None # RRR counts not needed
        else: # need to run triple_counts
            # Select the executable name
            exec_name = "bin/triple.s_mu" + "_periodic" * periodic + "_verbose" * verbose
            # the above must be true relative to the script location
            # below we should make it absolute, i.e. right regardless of the working directory
            exec_path = os.path.join(os.path.realpath(os.path.dirname(__file__)), exec_name)

            # form the command line
            command = "env OMP_PROC_BIND=spread OMP_PLACES=threads " # set OMP environment variables, should not be set before
            command += f"{exec_path} -output {out_dir}/ -nside {sampling_grid_size} -rescale {coordinate_scaling} -nthread {nthread} -maxloops {n_loops} -loopspersample {loops_per_sample} -N2 {N2} -N3 {N3} -xicut {xi_cut_s} -binfile {binfile} -binfile_cf {binfile_cf} -mbin_cf {xi_n_mu_bins} -cf_loops {xi_refinement_iterations}" # here are universally acceptable parameters
            command += "".join([f" -in{suffixes_tracer[t]} {input_filenames[t]}" for t in range(ntracers)]) # provide all the random filenames
            command += "".join([f" -norm{suffixes_tracer[t]} {ndata[t]}" for t in range(ntracers)]) # provide all ndata for normalization
            command += "".join([f" -cor{suffixes_corr[c]} {cornames[c]}" for c in range(ncorr)]) # provide all correlation functions
            command += f" -mbin {n_mu_bins}"
            if periodic: # append periodic flag and box size
                command += f" -perbox -boxsize {boxsize}"

            # deal with the seed
            if seed is not None: # need to pass to the C++ code and make sure it can be received properly. 0 (False) is not equivalent to None in this case
                seed &= 2**32 - 1 # this bitwise AND truncates the seed into a 32-bit unsigned (positive) integer (definitely a subset of unsigned long)
                command += f" -seed {seed}"
        
            # run the triple_counts code
            print_and_log(datetime.now())
            print_and_log(f"Launching the triple_counts C++ code with command: {command}")
            status = os.system(f"bash -c 'set -o pipefail; stdbuf -oL -eL {command} 2>&1 | tee -a {logfile}'")
            # tee prints what it gets to stdout AND saves to file
            # stdbuf -oL -eL should solve the output delays due to buffering without hurting the performance too much
            # without pipefail, the exit_code would be of tee, not reflecting main command failures
            # feed the command to bash because on Ubuntu it was executed in sh (dash) where pipefail is not supported

            # check the run status
            exit_code = os.waitstatus_to_exitcode(status) # assumes we are in Unix-based OS; on Windows status is the exit code
            if exit_code: raise RuntimeError(f"The triple_counts C++ code terminated with an error: exit code {exit_code}")
            print_and_log("The triple_counts C++ code finished succesfully")

            RRR_filename = f"{out_dir}/RRR_counts_n{n_r_bins}_m{n_mu_bins}_full.txt"
        
        inv_phi_filename = compute_3pcf_correction_function(input_filenames[0], binfile, out_dir, periodic, RRR_filename, print_function=print_and_log)
    else: # need to convert RRR counts from ENCORE/CADENZA format
        pass

    # Select the executable name
    exec_name = "bin/cov.3pcf_" + mode + "_periodic" * periodic + "_verbose" * verbose
    # the above must be true relative to the script location
    # below we should make it absolute, i.e. right regardless of the working directory
    exec_path = os.path.join(os.path.realpath(os.path.dirname(__file__)), exec_name)

    # form the command line
    command = "env OMP_PROC_BIND=spread OMP_PLACES=threads " # set OMP environment variables, should not be set before
    command += f"{exec_path} -output {out_dir}/ -nside {sampling_grid_size} -rescale {coordinate_scaling} -nthread {nthread} -maxloops {n_loops} -loopspersample {loops_per_sample} -N2 {N2} -N3 {N3} -N4 {N4} -N5 {N5} -N6 {N6} -xicut {xi_cut_s} -binfile {binfile} -binfile_cf {binfile_cf} -mbin_cf {xi_n_mu_bins} -cf_loops {xi_refinement_iterations}" # here are universally acceptable parameters
    command += "".join([f" -in{suffixes_tracer[t]} {input_filenames[t]}" for t in range(ntracers)]) # provide all the random filenames
    command += "".join([f" -norm{suffixes_tracer[t]} {ndata[t]}" for t in range(ntracers)]) # provide all ndata for normalization
    command += "".join([f" -cor{suffixes_corr[c]} {cornames[c]}" for c in range(ncorr)]) # provide all correlation functions
    command += f" -max_l {max_l}"
    command += f" -phi_file {inv_phi_filename}"
    if periodic: # append periodic flag and box size
        command += f" -perbox -boxsize {boxsize}"

    # deal with the seed
    if seed is not None: # need to pass to the C++ code and make sure it can be received properly. 0 (False) is not equivalent to None in this case
        seed &= 2**32 - 1 # this bitwise AND truncates the seed into a 32-bit unsigned (positive) integer (definitely a subset of unsigned long)
        command += f" -seed {seed}"
    
    # run the main code
    print_and_log(datetime.now())
    print_and_log(f"Launching the C++ code with command: {command}")
    status = os.system(f"bash -c 'set -o pipefail; stdbuf -oL -eL {command} 2>&1 | tee -a {logfile}'")
    # tee prints what it gets to stdout AND saves to file
    # stdbuf -oL -eL should solve the output delays due to buffering without hurting the performance too much
    # without pipefail, the exit_code would be of tee, not reflecting main command failures
    # feed the command to bash because on Ubuntu it was executed in sh (dash) where pipefail is not supported

    # clean up
    for input_filename in input_filenames: os.remove(input_filename) # delete the larger (temporary) input files
    for tmp_path in tmp_dirs_to_clean_up: rmdir_if_exists_and_empty(tmp_path) # safely remove the temporary directory and all its parents that did not exist before, but only if they are empty

    # check the run status
    exit_code = os.waitstatus_to_exitcode(status) # assumes we are in Unix-based OS; on Windows status is the exit code
    if exit_code: raise RuntimeError(f"The C++ code terminated with an error: exit code {exit_code}")
    print_and_log("The C++ code finished succesfully")

    # post-processing
    print_and_log(datetime.now())
    print_and_log("Starting post-processing")
    results = post_process_3pcf(out_dir, n_r_bins, max_l, n_loops // loops_per_sample, out_dir, shot_noise_rescaling1, print_function=print_and_log)
    # TODO:
    # eliminate n_samples argument
    # add skip_s_bins and skip_l functionality
    # add mock post-processing

    print_and_log("Finished post-processing")
    print_and_log(datetime.now())

    print_and_log("Performing an extra convergence check")
    convergence_check_extra(results, print_function = print_and_log)

    print_and_log("Finished.")
    print_and_log(datetime.now())
    return results