Computing Jackknife Weights
=============================

**NB**: This is *only* required when running the RascalC code in the JACKKNIFE mode.

Here, we compute the weights assigned to each jackknife region for each bin. This is done using the `Corrfunc <https://corrfunc.readthedocs.io>`_ code of Sinha & Garrison to compute the weights :math:`w_{aA}^{XY} = RR_{aA}^{XY} / \sum_B RR_{aB}^{XY}` for bin :math:`a`, jackknife :math:`A` and fields :math:`X` and :math:`Y`. 

Two codes are supplied; one using a single set of tracer particles and the other with two input sets, for computation of cross-covariance matrices. These are in the ``scripts/legacy/`` directory. This must be run before the main C++ code.

Usage
~~~~~~~
For a single field analysis::

    python scripts/legacy/jackknife_weights.py {RANDOM_PARTICLE_FILE} {BIN_FILE} {MU_MAX} {N_MU_BINS} {NTHREADS} {PERIODIC} {OUTPUT_DIR}

For an analysis using two distinct fields::

    python scripts/legacy/jackknife_weights_cross.py {RANDOM_PARTICLE_FILE_1} {RANDOM_PARTICLE_FILE_2} {BIN_FILE} {MU_MAX} {N_MU_BINS} {NTHREADS} {PERIODIC} {OUTPUT_DIR}
    
**NB**: The two field script computes all three combinations of weights between the two random fields, thus has a runtime :math:`\sim` 3 times that of ``jackknife_weights.py``. Running these together in one script ensures that we have the same number of jackknives for all fields. Also, the two fields must be distinct, else there are issues with double counting. 


**Input Parameters**

- {RANDOM_PARTICLE_FILE}, {RANDOM_PARTICLE_FILE_1}, {RANDOM_PARTICLE_FILE_2}: Input ASCII file containing random particle positions and jackknife numbers in {x,y,z,weight,jackknife_ID} format, such as that created with the :doc:`pre-processing` scripts. This should be in ``.csv``, ``.txt`` or ``.dat`` format with space-separated columns.
- {BIN_FILE}: ASCII file specifying the radial bins, as described in :ref:`file-inputs`. This can be user-defined or created by the :ref:`write-binning-file` scripts.
- {MU_MAX}: Maximum :math:`\mu = \cos\theta` used in the angular binning.
- {N_MU_BINS}: Number of angular bins used in the range :math:`[0,\mu]`.
- {NTHREADS}: Number of CPU threads to use for pair counting parallelization.
- {PERIODIC}: Whether the input dataset has periodic boundary conditions (0 = non-periodic, 1 = periodic). See note below.
- {OUTPUT_DIR}: Directory in which to house the jackknife weights and pair counts. This will be created if not in existence.


**Notes**:

- This is a very CPU intensive computation since we must compute pair counts between every pair of random particles up to the maximum particle separation. The process can be expedited using multiple CPU cores or a reduced number of random particles (e.g. via the :ref:`particle-subset` script).

**Note on Periodicity**

The code can be run for datasets created with either periodic or non-periodic boundary conditions. Periodic boundary conditions are often found in cosmological simlulations. If periodic, the pair-separation angle :math:`\theta` (used in :math:`\mu=\cos\theta`) is measured from the :math:`z` axis, else it is measured from the radial direction. If periodic data is used, the C++ code **must** be compiled with the -DPERIODIC flag.

Output files
~~~~~~~~~~~~~

This code creates ASCII files containing the jackknife weights for each bin, the RR pair counts for each bin in each jackknife and the summed RR pair counts in each bin. The output files have the format ``jackknife_weights_n{N}_m{M}_j{J}_{INDEX}.dat``, ``jackknife_pair_counts_n{N}_m{M}_j{J}_{INDEX}.dat`` and ``binned_pair_counts_n{N}_m{M}_j{J}_{INDEX}.dat`` respectively N and M specify the number of radial and angular bins respectively and J gives the number of non-empty jackknives. INDEX specifies which fields are being used i.e. INDEX = 12 implies the :math:`w_{aA}^{12}`, :math:`RR_{aA}^{12}` and :math:`RR_a^{12}` quantities.

The binned pair counts is a list of weighted pair counts for each bin, summed over all jackknife regions, in the form :math:`RR_a^{J,XY} = \sum_B RR_{aB}^{XY}`, with each bin on a separate row. The jackknife pair counts and jackknife weights files list the quantities :math:`RR_{aA}^{XY}` and :math:`w_{aA}^{XY}` for each bin and jackknife region respectively. We note that the :math:`RR_{aA}^{XY}` quantities (and only these) are normalized by the whole-survey product of summed weights :math:`\left(\sum_i w_i\right)^2` for later convenience.

The :math:`j`-th row contains the (tab-separated) quantities for each bin using the :math:`j`-th jackknife. The first value in each row is the jackknife number, and the bins are ordered using the collapsed binning :math:`\mathrm{bin}_\mathrm{collapsed} = \mathrm{bin}_\mathrm{radial}\times n_\mu + \mathrm{bin}_\mathrm{angular}` for a total of :math:`n_\mu` angular bins.  
