# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import pandas as pd
import multiprocessing
import functools
from importlib import reload
import tools.dataModels.measurements

reload(tools.dataModels.measurements)
from tools.dataModels.measurements import Measurements
import tools.graphics.snapshots as snaps
import seaborn as sns

# %%
#dataPath = input()
dataPath = "/home/jgonzaleznunez/Projects/activeMatterCommunities/workspace/simulations/R:1_MP:10000_FS:10000_S:50.0_IT:passive_ring_IC:10_D:0.0_K:100.0_dT:0.005_T:0.1_MAR:4.0,4.0_GR:0.0005,0.0005_EDT:1_ASD:200"

_, times = Measurements.extract_times(dataPath)

# %%
#> snapshot of the grains and their orientations
data = snaps.extract_data(dataPath, times[-15])

grain_angle, grain_id, x_gbp, y_gbp, cluster_id = data.grains(return_angle=True)

values = ((grain_angle + np.pi)% np.pi) / np.pi
cm = get_cmap("hsv")

#%%
fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(10, 10))
axes[0,0].set_aspect('equal')

snaps.iterateRods(
    axes[0,0], cm(values), data.x, data.y, data.ex, data.ey, 0.95 * data.l, data.phi, data.r
)

tmax = max(max(abs(data.x)), max(abs(data.y)))

axes[0,0].set_xlim([-tmax, tmax])
axes[0,0].set_ylim( [-tmax, tmax])

#. grain size distribution
areas = np.zeros(np.max(np.unique(grain_id)) + 1, dtype=np.float64)
for i_cell in range(data.l.size):
    areas[grain_id[i_cell]] += data.l[i_cell] * data.thickness

mask = np.where(areas > 0)
weights, bins = np.histogram(areas[mask], bins=50)

for ax in [axes[0,1]]:
    ax.scatter(bins[:-1], weights, c="blue", s=10)
    ax.set_yscale("log")
    ax.set_ylabel("Frequency", fontsize=14)
    ax.set_xlabel("Area", fontsize=14)

#. domain area vs colony radial position
areas_r = np.zeros(np.max(np.unique(grain_id)) + 1, dtype=np.float64)
grain_counts = np.zeros(np.max(np.unique(grain_id)) + 1, dtype=np.float64)

annulus_size = 5 * data.thickness
center_mass  = np.array([np.mean(data.x), np.mean(data.y)])
cell_r = np.hypot(data.x - center_mass[0], data.y - center_mass[1])

bin_edges = np.arange(0, cell_r.max(), annulus_size)
weights = np.zeros(len(bin_edges) - 1)
for j in range(len(bin_edges) - 1):
    mask = np.where((cell_r >= bin_edges[j]) & (cell_r < bin_edges[j + 1]))
    grain_id_annulus = np.unique(grain_id[mask])
    _areas = dict()

    for id in grain_id_annulus:
        _areas[id] = np.sum(data.l[mask][np.where(grain_id[mask] == id)])

    weights[j] = np.sum(list(_areas.values())) / len(_areas) 


axes[1,0].scatter(bin_edges[:-1], weights, c="blue", s=10)

#%%
#> time-series of the grain properties
def task_def(time, dataPath):
    data = Measurements(dataPath=dataPath, time=time)
    grain_angle, grain_id, *_ = data.grains(return_angle=True)
    element_counts = np.bincount(grain_id)
    grain_size_mean = np.mean(element_counts[np.where(element_counts > 0)])
    avg_grain_angle, _counts = 0, 0
    for i in np.unique(grain_id):
        _angle = np.mean(grain_angle[np.where(grain_id == i)])
        com_x = np.mean(data.x[np.where(grain_id == i)])
        com_y = np.mean(data.y[np.where(grain_id == i)])
        radial_angle = np.arctan2(com_y, com_x)
        avg_grain_angle += np.cos( 2 * (_angle - radial_angle))

        _counts += 1
    return (time, grain_size_mean, avg_grain_angle / _counts)



task = functools.partial(task_def, dataPath=dataPath)
with multiprocessing.Pool(12) as pool:
    results = pool.map(task, times[::2])

#%%
fig, ax = plt.subplots()
twin = ax.twinx()

time, grain_size_mean, avg_grain_angle = list(zip(*results))
ax.plot(time, grain_size_mean, c="blue", label="grain size")

twin.plot(time, avg_grain_angle, c="green", label="grain angle")

fig.suptitle(f"Grain Properties")
ax.set_xlabel("t")
ax.set_ylabel("mean grain size")
twin.set_ylabel("grain angle")

ax.legend(loc="upper right")
twin.legend(loc="upper left")


# %%
angle = np.arctan2(data.ey, data.ex)
