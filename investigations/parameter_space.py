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
def task_def(dataPath):
    _, times = Measurements.extract_times(dataPath)

    if len(times) == 0:
        return None

    data = Measurements(dataPath=dataPath, time=times[-1])
    cellnumber = len(data.l)
    return cellnumber


# .iterate through all rows of the data frame
with multiprocessing.Pool(args.n_cores) as pool:
    results = pool.map(task_def, list(df.basePath))

# %%
# .replace results elements of none type ot arrays of nans
for i in range(len(results)):
    if results[i] is None:
        results[i] = 0

cellnumber_table = pd.DataFrame(results, columns=["cellnumber"])
merged_df = pd.concat([df, cellnumber_table], axis=1)

print("these have strange number of cells")
print(merged_df)


# %%
def task_def(dataPath):
    _, times = Measurements.extract_times(dataPath)

    if len(times) == 0:
        return None

    values = np.zeros((10, len(times)), dtype=float)
    mask = np.zeros(len(times), dtype=bool)

    for i, time in enumerate(times[-3:]):
        data = Measurements(dataPath=dataPath, time=time)

        H = data.Heterozygosities()
        radial = data.radial_alignment()
        nematic = data.nematic_global()
        polar = data.polar_global()
        edge_frac = data.edgeFraction(
            data.color2, data.x, data.y, bandwidthFraction=0.1
        )
        cellnumber = len(data.l)
        r_max = np.max(np.hypot(data.x, data.y))

        values[0, i] = np.mean(H[0])
        values[1, i] = np.mean(H[1])
        values[2, i] = np.mean(radial[data.color2 == 0])
        values[3, i] = np.mean(radial[data.color2 == 1])
        values[4, i] = r_max
        values[5, i] = cellnumber
        values[6, i] = edge_frac
        values[7, i] = polar
        values[8, i] = nematic
        values[9, i] = data.time
        mask[i] = True if cellnumber > 5000 else False
    _mean = np.mean(values[:, mask], axis=1)
    _std = np.std(values[:, mask], axis=1)
    return np.stack((_mean, _std), axis=1)


# .iterate through all rows of the data frame
with multiprocessing.Pool(args.n_cores) as pool:
    results = pool.map(task_def, list(df.basePath))

# %%
# .replace results elements of none type ot arrays of nans
for i in range(len(results)):
    if results[i] is None:
        results[i] = np.full((10, 2), np.nan)

data_frames = []
for i in [0, 1]:
    columns = [
        "H_1",
        "H_2",
        "radial_1",
        "radial_2",
        "r_max",
        "cellnumber",
        "edge_frac",
        "polar",
        "nematic",
        "time",
    ]
    if i == 1:
        columns = [col + "_std" for col in columns]

    _results = [result[:,i] for result in results]
    _df = pd.DataFrame(_results, columns=columns)
    data_frames.append(_df)

# %%
if not os.path.exists(args.outputPath):
    os.mkdir(args.outputPath)

merged_df = pd.concat([df, *data_frames], axis=1)
merged_df.to_csv(os.path.join(args.outputPath, "parameterSpaceTable.csv"), index=False)
