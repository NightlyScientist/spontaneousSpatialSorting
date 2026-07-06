# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import pandas as pd
import multiprocessing
import functools
from importlib import reload
import tools.dataModels.measurements as _measurements

reload(_measurements)
from tools.dataModels.measurements import Measurements

# %%
dataPath = input()

opts, times = Measurements.extract_times(dataPath)

# %%
labels = np.round(
    np.array([opts["division_lengths_1"], opts["division_lengths_2"]]).astype(float), 2
).astype(str)


# %%
# >Heterozygosity, H_0(t)
def task_def(time, dataPath):
    data = Measurements(dataPath=dataPath, time=time)
    size = len(data.l)
    H = data.Heterozygosities()
    H_1, H_2 = np.mean(H[0]), np.mean(H[1])
    H_0 = 0.5 * (H_1 + H_2)
    return (data.time, H_1, H_2, H_0, size)


task = functools.partial(task_def, dataPath=dataPath)
with multiprocessing.Pool(12) as pool:
    results = pool.map(task, times)

fig, ax = plt.subplots()
twin = ax.twinx()

time, H_1, H_2, H_0, size = list(zip(*results))
ax.plot(time, H_0, c="blue", label="H_0")
ax.plot(time, H_1, c="green", label=labels[0])
ax.plot(time, H_2, c="red", label=labels[1])
twin.plot(time, size, c="black", label="Number of Cells")

fig.suptitle(f"Heterozygosity, $H_0(t)$ vs $t$")
ax.set_xlabel("t")
ax.set_ylabel("H_0(t)")
twin.set_ylabel("Number of Cells")

ax.legend(loc="upper right")
twin.legend(loc="upper left")


# %%
# >global nematic order
def task_def(time, dataPath):
    data = Measurements(dataPath=dataPath, time=time)
    size = len(data.l)
    nematic = data.nematic_global()
    polar = data.polar_global()
    return (data.time, nematic, polar, size)


task = functools.partial(task_def, dataPath=dataPath)
with multiprocessing.Pool(12) as pool:
    results = pool.map(task, times)

fig, ax = plt.subplots()
twin = ax.twinx()

time, nematic, polar, size = list(zip(*results))
ax.plot(time, nematic, c="blue", label="nematic")
ax.plot(time, polar, c="red", label="polar")
twin.plot(time, size, c="black", label="Number of Cells")

fig.suptitle("Global nematic order vs t")
ax.set_xlabel("t")
ax.set_ylabel("f(theta)")
ax.legend()


# %%
# >radial alignment timeseries
def task_def(time, dataPath):
    data = Measurements(dataPath=dataPath, time=time)
    radial = data.radial_alignment()
    size = len(data.l)
    r_1 = np.mean(radial[data.color2 == 0])
    r_2 = np.mean(radial[data.color2 == 1])
    return (data.time, r_1, r_2, np.mean(radial), np.std(radial), size)


task = functools.partial(task_def, dataPath=dataPath)
with multiprocessing.Pool(12) as pool:
    results = pool.map(task, times)


fig, ax = plt.subplots(nrows=1, figsize=(6, 6))
twin = ax.twinx()

time, r_1, r_2, rad, rad_std, size = list(zip(*results))

ax.plot(time, rad, c="blue", label="total", marker=".")
ax.plot(time, r_1, c="yellow", label=labels[0], marker=".")
ax.plot(time, r_2, c="red", label=labels[1], marker=".")
twin.plot(time, size, c="black", label="Number of Cells")

ax.set_xlabel("time")
ax.set_ylabel("|(r, e)|")
twin.set_ylabel("Number of Cells")

ax.legend(loc="lower right")
ax.title.set_text("Radial alignment vs t")


# %%
# >population growth rate and aspect ratio
def task_def(time, dataPath):
    data = Measurements(dataPath=dataPath, time=time)
    radial = data.radial_alignment()
    size = len(data.l)
    r_1 = np.mean(radial[data.color2 == 0])
    r_2 = np.mean(radial[data.color2 == 1])
    return (data.time, r_1, r_2, np.mean(radial), np.std(radial), size)


task = functools.partial(task_def, dataPath=dataPath)
with multiprocessing.Pool(12) as pool:
    results = pool.map(task, times)


fig, ax = plt.subplots(nrows=1, figsize=(6, 6))
twin = ax.twinx()

time, r_1, r_2, rad, rad_std, size = list(zip(*results))

ax.plot(time, rad, c="blue", label="total", marker=".")
ax.plot(time, r_1, c="yellow", label=labels[0], marker=".")
ax.plot(time, r_2, c="red", label=labels[1], marker=".")
twin.plot(time, size, c="black", label="Number of Cells")

ax.set_xlabel("time")
ax.set_ylabel("|(r, e)|")
twin.set_ylabel("Number of Cells")

ax.legend(loc="lower right")
ax.title.set_text("Radial alignment vs t")


# %%
# >edge population fraction
def task_def(time, dataPath):
    data = Measurements(dataPath=dataPath, time=time)
    size = len(data.l)
    size = np.max(np.hypot(data.x, data.y))
    edge_frac = data.edgeFraction(data.color2, data.x, data.y, bandwidth=2)
    return (data.time, edge_frac, size)


task = functools.partial(task_def, dataPath=dataPath)
with multiprocessing.Pool(12) as pool:
    results = pool.map(task, times)

fig, ax = plt.subplots()
twin = ax.twinx()

time, edge_fraction, size = list(zip(*results))
ax.plot(time, edge_fraction, c="blue", label="edge fraction")
twin.plot(time, size, c="black", label="Number of Cells")

ax.title.set_text("Edge Population Fraction vs t")
ax.set_xlabel("t")
ax.set_ylabel("edge fraction")
twin.set_ylabel("Number of Cells")

ax.legend(loc="upper right")
twin.legend(loc="upper left")

# %%
import scipy.spatial as spatial
from tools.graphics.nematics import fieldSnapshot, defects_from_Q


def defects_association(data, pixel_number=50, num_particles=10):
    def construct_kdtree(data):
        points = np.column_stack((data.x, data.y))
        kdtree = spatial.KDTree(points)
        return kdtree

    def find_closest_particles(kdtree, x, y, num_particles):
        points = np.column_stack((x, y))
        distances, indices = kdtree.query(points, k=num_particles)
        return distances, indices

    data.nematicField(n=pixel_number)
    neg_x, neg_y, neg_px, neg_py, pos_x, pos_y, pos_px, pos_py = defects_from_Q(
        data.n_x, data.n_y, data.Q11, data.Q12
    )

    kdtree = construct_kdtree(data)

    defect_association = dict()
    for _x, _y, _c, _t in [
        (neg_x, neg_y, "green", "neg"),
        (pos_x, pos_y, "pink", "pos"),
    ]:
        distances, indices = find_closest_particles(
            kdtree, _x, _y, num_particles=num_particles
        )
        defect_association[_t] = indices

    return defect_association, data


def calculate_average_color(defect_association, data):
    average_color = {}
    for defect_type, indices in defect_association.items():
        if len(indices) == 0:
            average_color[defect_type] = -1
        else:
            average_color[defect_type] = np.mean(data.color2[indices])
    return average_color


# > defect association with population color(2)
def task_def(time, dataPath, pixel_number=50, num_particles=10):
    data = Measurements(dataPath=dataPath, time=time)
    size_number = len(data.l)
    size_radius = np.max(np.hypot(data.x, data.y))

    def_association, data = defects_association(
        data=data, pixel_number=50, num_particles=num_particles
    )
    average_color = calculate_average_color(def_association, data)
    return (
        data.time,
        average_color["neg"],
        average_color["pos"],
        size_number,
        size_radius,
    )


task = functools.partial(
    task_def,
    dataPath=dataPath,
    pixel_number=50,
)
with multiprocessing.Pool(12) as pool:
    results = pool.map(task, times)

# %%
fig, ax = plt.subplots()
twin = ax.twinx()

time, avg_neg_color, avg_pos_color, s_num, s_radius = list(zip(*results))
avg_neg_color = np.array(avg_neg_color)
avg_pos_color = np.array(avg_pos_color)
time = np.array(time)

ax.plot(
    time[avg_neg_color > -1],
    avg_neg_color[avg_neg_color > -1],
    c="blue",
    label="neg defect",
)
ax.plot(
    time[avg_pos_color > -1],
    avg_pos_color[avg_pos_color > -1],
    c="red",
    label="pos defect",
)
twin.plot(time, s_num, c="black", label="Number of Cells")

ax.title.set_text("defect correlation")
ax.set_xlabel("t")
ax.set_ylabel("avg color")
twin.set_ylabel("Number of Cells")

ax.legend(loc="upper right")
twin.legend(loc="upper left")
