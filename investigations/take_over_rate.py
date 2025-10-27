# %%
import pandas as pd
import numpy as np
from numba import njit
import argparse
import multiprocessing
from importlib import reload
from routines.parameterTable import parameterTable
import tools.dataModels.measurements as _measurements
from tools.dataModels.measurements import Measurements
import matplotlib.pyplot as plt

reload(_measurements)

# . read cmd line arguments, fallback to defined values
# try:
#    parser = argparse.ArgumentParser()
#    parser.add_argument("--basePath", type=str, required=True)
#    parser.add_argument("--outputPath", type=str, required=True)
#    parser.add_argument("--n_cores", type=int, default=10)
#    args = parser.parse_args()
# except Exception as _:
base_path = input("Enter the base path: ")
output_path = input("Enter the output path: ")
args = argparse.Namespace(
    basePath=base_path,
    outputPath=output_path,
    n_cores=10,
)

# %%
df = parameterTable(args.basePath)


# %%
from numba.typed import List
import os


@njit(cache=True)
def bin_angle(angle, n_bins):
    bin_size = 2 * np.pi / n_bins
    return int((angle % (2 * np.pi)) // bin_size)


@njit(cache=True)
def partition_circle(x, y, center, n_parts):
    # Create a Numba-typed list of lists of ints
    partitions = List()
    for _ in range(n_parts):
        partitions.append(List.empty_list(np.int64))
    particle_angles = np.mod(np.arctan2(y - center[1], x - center[0]), 2 * np.pi)
    for i in range(len(x)):
        bin_idx = bin_angle(particle_angles[i], n_parts)
        partitions[bin_idx].append(i)
    return partitions


def colony_center(x, y):
    return np.array([x.mean(), y.mean()])


def outer_fraction(x, y, id, center, fraction):
    distances = np.hypot(x - center[0], y - center[1])
    threshold = np.percentile(distances, fraction * 100)
    return np.mean(id[distances >= threshold])


def sub_task(dataPath, time):
    data = Measurements(dataPath=dataPath, time=time)
    mask = data.color2 == 1
    center = colony_center(data.x[mask], data.y[mask])
    p = partition_circle(data.x, data.y, center, 8)
    mean_tot = 0
    for p_ in p:
        mean_id = outer_fraction(data.x[p_], data.y[p_], data.color2[p_], center, 0.95)
        if mean_id < np.inf:
            mean_tot += mean_id
    return np.log2(len(data.x) / 40), mean_tot / len(p)


def main_task(path):
    _, times = Measurements.extract_times(path)
    if len(times) == 0:
        return None
    timeseries = []
    for time in times:
        t_gen, m = sub_task(path, time)
        timeseries.append([time, m, t_gen])
    return np.array(timeseries)


# %%
# iterate through all rows of the data frame
with multiprocessing.Pool(args.n_cores) as pool:
    results = pool.map(main_task, list(df.basePath))

# %%
# aggregate results and plot mean with error bars
valid = [arr for arr in results if arr is not None and len(arr) > 0]
if len(valid) == 0:
    raise RuntimeError("No valid time series in results to plot.")

# build a common time grid (unique sorted times across all runs)
common_times = np.unique(np.concatenate([arr[:, 2] for arr in valid]))

# interpolate each series onto the common grid
interpolated = []
for arr in valid:
    times = np.array(arr[:, 2])
    values = np.array(arr[:, 1])
    sort_idx = np.argsort(times)
    times = times[sort_idx]
    values = values[sort_idx]
    interp_vals = np.interp(common_times, times, values, left=np.nan, right=np.nan)
    interpolated.append(interp_vals)
stack = np.vstack(interpolated)  # shape: (n_runs, n_times)

# compute mean and standard error (ignore NaNs)
mean = np.nanmean(stack, axis=0)
counts = np.sum(~np.isnan(stack), axis=0)
sem = np.nanstd(stack, axis=0) / np.sqrt(np.maximum(counts, 1))
sem[counts == 0] = np.nan

# plot

plt.rcParams['font.size'] = 16
plt.figure(figsize=(10, 6))
plt.errorbar(common_times, mean, yerr=sem, fmt='-o', capsize=3, c="black", label='mean ± SEM')
plt.xlabel("time [generations]", fontsize=22)
plt.ylabel("$f_{B, outer}$", fontsize=34)
#plt.title("Mean time series of population B fraction (with SEM)")
plt.legend()
plt.tight_layout()
