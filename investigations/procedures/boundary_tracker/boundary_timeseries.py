# %%
import os
import glob
import pandas as pd
import numpy as np
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union
from concurrent.futures import ProcessPoolExecutor
import matplotlib.pyplot as plt
from tqdm import tqdm
import shutil
import matplotlib.pyplot as plt


def process_opts(csv_path):
    opts = pd.read_csv(csv_path, header=0).to_dict(orient="index")[0]
    return opts

path = "/home/jgonzaleznunez/Projects/spontaneousSpatialSorting/investigations/procedures/boundary_tracker/timeseries_out/"

sims = os.listdir(path)
data = dict()

fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlabel("area")
ax.set_ylabel("signed periphery distance")

for sim in sims:
    files = list(glob.glob(os.path.join(path, sim, "*.csv")))
    
    opts_index = 0 if "opts" in files[0] else 1 
    opts = process_opts(os.path.join(path, sim, files[opts_index]))

    if "passive_ring" != opts["initial_type"]:
        print(f"Skipping {sim} due to no passive ring condition.")
        continue

    timeseries = pd.read_csv(os.path.join(path, sim, files[1-opts_index]))

    arr = np.array(timeseries.mean_signed_distance.values)
    nonzero_indices = np.nonzero(arr)[0]
    if nonzero_indices.size > 0:
        last_nonzero_index = nonzero_indices[-1]
    else:
        last_nonzero_index = None    # Or handle as needed

    mean_signed_distance = timeseries["mean_signed_distance"].values
    std_signed_distance = timeseries["std_signed_distance"].values

    dist = mean_signed_distance[last_nonzero_index]
    dist_std = std_signed_distance[last_nonzero_index]

    length_1 = opts["division_length_1"]
    length_2 = opts["division_length_2"]
    thickness_1 = opts["thickness_1"]
    thickness_2 = opts["thickness_2"]

    area = length_2 * thickness_2

    if opts["max_aspect_ratio_1"] > 4:
        print(f"Skipping {sim} due to max aspect ratio condition.")
        continue

    ax.errorbar(area, dist, yerr=dist_std, fmt="o", label=sim)
    #ax.errorbar(thickness_2 / thickness_1, dist, yerr=dist_std, fmt="o", label=sim)

    #break
    
        #sim_csv_folder = os.path.join(csv_folder, sim)
        #sim_periphery_folder = os.path.join(periphery_dir, sim)
        #sim_outdir = os.path.join(outdir, sim)
        #timeseries_csv = os.path.join(timeseries_dir, sim, output_csv)
        #os.makedirs(os.path.join(timeseries_dir, sim), exist_ok=True)

       
