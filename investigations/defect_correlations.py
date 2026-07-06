# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from matplotlib.cm import get_cmap
import matplotlib as mpl
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tools.dataModels.measurements import Measurements
from numba import jit
from functools import partial
from tools.graphics.nematics import fieldSnapshot, defects_from_Q

from ipywidgets import interact, fixed
import matplotlib.pyplot as plt
from importlib import reload
from tools.dataAPI.datamodel import DataModel
import tools.graphics.snapshots as snaps

reload(snaps)

import scipy.spatial as spatial

plt.style.use("fast")
# %%
dataPath = input()

opts, times = DataModel.extract_times(dataPath)


# %%
def extract_data(dataPath: str, time: int):
    data = Measurements(dataPath=dataPath, time=time)
    return data

def show_figure(pixel_number, data_path, time):
    fig, ax = plt.subplots(figsize=(10, 10))
    fig.tight_layout()
    ax.set_aspect("equal")
    ax.tick_params(axis="both", labelsize=20)

    data = extract_data(data_path, time)
    data.nematicField(n=pixel_number)
    cm = get_cmap("coolwarm")
    values = data.color2.astype(float)

    snaps.iterateRods(
        ax,
        cm(values),
        data.x,
        data.y,
        data.ex,
        data.ey,
        0.95 * data.l,
        data.phi,
        data.r,
    )

    tmax = max(max(abs(data.x)), max(abs(data.y)))
    plt.axis([-tmax, tmax, -tmax, tmax])

    neg_x, neg_y, neg_px, neg_py, pos_x, pos_y, pos_px, pos_py = defects_from_Q(
        data.n_x, data.n_y, data.Q11, data.Q12
    )
    ax.scatter(neg_x, neg_y, c="orange", marker="^", s=200)
    ax.scatter(pos_x, pos_y, c="r", marker="o", s=200)

    def construct_kdtree(data):
        points = np.column_stack((data.x, data.y))
        kdtree = spatial.KDTree(points)
        return kdtree

    def find_closest_particles(kdtree, x, y, num_particles):
        points = np.column_stack((x, y))
        distances, indices = kdtree.query(points, k=num_particles)
        return distances, indices

    kdtree = construct_kdtree(data)

    defect_association = dict()
    for _x, _y, _c, _t in [
        (neg_x, neg_y, "green", "neg"),
        (pos_x, pos_y, "pink", "pos"),
    ]:
        distances, indices = find_closest_particles(kdtree, _x, _y, num_particles=10)
        ax.scatter(data.x[indices], data.y[indices], c=_c, marker="o", s=150)

        defect_association[_t] = indices

    return defect_association, data


defect_association, data = show_figure(pixel_number=50, data_path=dataPath, time=times[100])

#%%
def calculate_average_color(defect_association, data):
    average_color = {}
    for defect_type, indices in defect_association.items():
        average_color[defect_type] = np.mean(data.color2[indices])
    return average_color

average_color = calculate_average_color(defect_association, data)
print(average_color)