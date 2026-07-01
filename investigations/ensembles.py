#%%
from tools.dataAPI.datamodel import DataModel
from routines.parameterTable import parameterTable
import multiprocessing, functools, os
import numpy as np
import tools.dataModels.measurements as _measurements
from importlib import reload
reload(_measurements)
from tools.dataModels.measurements import Measurements
import matplotlib.pyplot as plt

#%%
basePath = input("Enter the path to the data:")
df = parameterTable(basePath)

#%%
df

#%%
# >Heterozygosity, H_0(t)
def task_def(time, dataPath):
    data = Measurements(dataPath=dataPath, time=time)
    size = len(data.l)
    H = data.Heterozygosities()
    H_1, H_2 = np.mean(H[0]), np.mean(H[1])
    H_0 = 0.5 * (H_1 + H_2)
    return (data.time, H_1, H_2, H_0, size)

collection = dict()
for dataPath in df.basePath:
    opts, times = Measurements.extract_times(dataPath)
    labels = np.round(
        np.array([opts["division_length_1"], opts["division_length_2"]]).astype(float), 2
    ).astype(str)

    task = functools.partial(task_def, dataPath=dataPath)
    with multiprocessing.Pool(12) as pool:
        results = pool.map(task, times)

    time, H_1, H_2, H_0, size = list(zip(*results))
    collection[dataPath] = (time, H_1, H_2, H_0, size)

#%%
fig, ax = plt.subplots()
twin = ax.twinx()

for dataPath, (time, H_1, H_2, H_0, size) in collection.items():
    ax.plot(time, H_0, c="blue")
    ax.scatter(time, H_1, c="green", marker=".")
    ax.scatter(time, H_2, c="red", marker="x")
    twin.plot(time, size, c="black")

ax.title.set_text(f"Heterozygosity, $H_0(t)$ vs $t$")
ax.set_xlabel("t")
ax.set_ylabel("H_0(t)")
twin.set_ylabel("Number of Cells")

# ax.legend(loc="upper right")
# twin.legend(loc="upper left")

