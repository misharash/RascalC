# This is the custom script to compute the 2PCF with jackknives

import os
import numpy as np
from pycorr import TwoPointCorrelationFunction
from tqdm import trange

z_min, z_max = 0.43, 0.7

work_dir = f"CMASS_N_data_{z_min}_{z_max}"

data_filename = os.path.join(work_dir, "CMASS_N_data.dat.xyzwj")
random_filename = os.path.join(work_dir, "CMASS_N_data.ran.xyzwj")

print("Reading data")
all_data = np.loadtxt(data_filename)
n_data, n_columns = all_data.shape
assert n_columns == 5, "Unexpected number of columns in data"
# first index is number of points, second is column: 0-2 for positions, 3 for weights, 4 for jackknife region

print("Reading randoms")
all_randoms_raw = np.loadtxt(random_filename)
n_randoms, n_columns = all_randoms_raw.shape
assert n_columns == 5, "Unexpected number of columns in randoms"

# decide number of splits so that random parts are about the data size
n_splits = n_randoms // n_data

# get the desired weight ratio
sum_w_randoms = sum(all_randoms_raw[:, 3])
sum_w_goal = sum_w_randoms / n_splits

print("Shuffling randoms")
np.random.shuffle(all_randoms_raw) # in place, by first axis
print("Splitting randoms")
all_randoms = [all_randoms_raw[i::n_splits] for i in range(n_splits)]

# reweigh each part so that the ratio of randoms to data is the same, just in case
for i_random in trange(n_splits, desc="Reweighting random part"):
    sum_w_random_part = sum(all_randoms[i_random][:, 3])
    w_ratio = sum_w_goal / sum_w_random_part
    all_randoms[i_random][:, 3] *= w_ratio

# tuple of edges, so that non-split randoms are below 20 and above they are split
all_edges = ((s_edges, np.linspace(-1, 1, 201)) for s_edges in (np.arange(21), np.arange(20, 201)))

results = []
# compute
for i_split_randoms, edges in enumerate(all_edges):
    result = 0
    D1D2 = None
    for i_random in trange(n_splits if i_split_randoms else 1, desc="Computing xi with random part"):
        these_randoms = all_randoms[i_random] if i_split_randoms else np.concatenate(all_randoms, axis=0)
        tmp = TwoPointCorrelationFunction(mode='smu', edges=edges,
                                          data_positions1=all_data[:, :3], data_weights1=all_data[:, 3], data_samples1=all_data[:, 4],
                                          randoms_positions1=these_randoms[:, :3], randoms_weights1=these_randoms[:, 3], randoms_samples1=these_randoms[:, 4],
                                          position_type='pos', engine='corrfunc', D1D2=D1D2, gpu=True, nthreads=4)
        # position_type='pos' corresponds to (N, 3) x,y,z positions shape like we have here
        D1D2 = tmp.D1D2
        result += tmp
    results.append(result)
print("Finished xi computations")
corr = results[0].concatenate_x(*results)
corr.D1D2.attrs['nsplits'] = n_splits

print("Saving the result")
corr.save(f"allcounts_BOSS_CMASS_N_{z_min}_{z_max}_lin_njack{60}_nran{n_splits}_split{20}.npy")
print("Finished")