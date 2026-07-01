# %%
import pandas as pd
import numpy as np
import argparse
import multiprocessing
import functools
import os
from importlib import reload
from routines.parameterTable import parameterTable
import tools.dataModels.measurements as _measurements

reload(_measurements)
from tools.dataModels.measurements import Measurements
from scipy.optimize import curve_fit

# . read cmd line arguments, fallback to defined values
try:
    parser = argparse.ArgumentParser()
    parser.add_argument("--basePath", type=str, required=True)
    parser.add_argument("--outputPath", type=str, required=True)
    parser.add_argument("--n_cores", type=int, default=10)
    args = parser.parse_args()
except:
    base_path = input("Enter the base path: ")
    output_path = input("Enter the output path: ")
    args = argparse.Namespace(
        basePath=base_path,
        outputPath=output_path,
        n_cores=10,
    )

basePath = args.basePath

# %%
df = parameterTable(basePath)

# %%
def get_colony_center(x, y):
    return np.mean(x), np.mean(y)


# doc hack to get the colony radius from n% largest radius cells
def get_colony_radius(x, y, n=0.95):
    cx, cy = get_colony_center(x, y)
    # .sort the data by the radius, in descending order
    radius = np.sort(np.hypot(x - cx, y - cy))[::-1]
    # . find first index where the radius is larger the x% of the largest radius
    idx = np.where(radius <= n * radius[0])[0][0]
    return (cx, cy), radius[0:idx].mean()


def task_def(dataPath):
    _, times = Measurements.extract_times(dataPath)

    if len(times) == 0:
        return None
    data = Measurements(dataPath=dataPath, time=times[-1])
    mask = data.color2 == 1
    center, radius = get_colony_radius(data.x, data.y, 0.95)

    passive_radii = radius - np.hypot(data.x - center[0], data.y - center[1])[mask]
    return passive_radii.mean()


# .iterate through all rows of the data frame
with multiprocessing.Pool(args.n_cores) as pool:
    results = pool.map(task_def, list(df.basePath))

# .replace results elements of none type ot arrays of zeros
for i in range(len(results)):
    if results[i] is None:
        results[i] = 0

# %%
rnd = np.random.randint(0, len(df))
p = df.basePath.values[rnd]
_, times = Measurements.extract_times(p)

data = Measurements(dataPath=p, time=times[-1])

import matplotlib.pyplot as plt

# Plotting the data points
fig, ax = plt.subplots(ncols=1)
fig.tight_layout()

# Setting the aspect ratio to equal
ax.axis("off")
ax.set_aspect("equal")

ax.scatter(data.x, data.y, s=10, c="gray")

mask = data.color2 == 1
ax.scatter(data.x[mask], data.y[mask], s=10, c="r")

center, radius = get_colony_radius(data.x, data.y, 0.95)

passive_radii = radius - np.hypot(data.x - center[0], data.y - center[1])[mask]

circle = plt.Circle(center, radius, color="r", fill=False)
ax.add_artist(circle)

#%%
table = pd.DataFrame(results, columns=["boundary_distance"])
merged_df = pd.concat([df, table], axis=1)
merged_df["boundary_distance"] = merged_df["boundary_distance"] / merged_df["thickness"]

import seaborn as sns
fig, ax = plt.subplots(ncols=1)

max_aspect_ratio_2 = np.sort(df.max_aspect_ratio_2.unique())
print(max_aspect_ratio_2)

_df = merged_df.groupby(["max_aspect_ratio_1", "max_aspect_ratio_2"]).agg({"boundary_distance": "mean"}, {"boundary_distance_std": "std"})

#sns.lineplot(data=merged_df, x="max_aspect_ratio_1", y="boundary_distance", hue="max_aspect_ratio_2", ax=ax, palette="flare")
sns.pointplot(data=merged_df, x="max_aspect_ratio_1", y="boundary_distance", hue="max_aspect_ratio_2", ax=ax, palette="flare", linestyles="", errorbar="se", markersize=3, err_kws={"linewidth": 0.5})


ax.set_xlabel("")
ax.set_ylabel("")
ax.legend(title="Population B", loc="center left", bbox_to_anchor=(0, 0.75))

