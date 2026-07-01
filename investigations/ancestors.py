# %%
import pandas as pd
import numpy as np
import tools.tracking.ancestors as ancestorTracker
from importlib import reload
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from tools.graphics import snapshots as snaps
import matplotlib as mpl
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from routines.parameterTable import parameterTable
import argparse
from tools.dataAPI.datamodel import DataModel


reload(ancestorTracker)

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

# read csv from path
df = parameterTable(basePath)

# %%
reload(ancestorTracker)
sample_path = df.basePath.values[3]
trajectories = ancestorTracker.fetch_trajectories(sample_path)

# %%
def color_map(traj, option, cmap="coolwarm"):
    cm = get_cmap(cmap)
    colors = None
    vrange = [0, 1]

    if option == "radial":
        colors = cm(traj.radial)
    elif option == "population":
        colors = cm([(1 + traj.population) / 2] * traj.x.size)
    elif option == "time":
        colors = cm(traj.time / traj.time.max())
        vrange = [0, traj.time.max()]
    elif option == "front":
        r = traj.front_displacement
        norm = mpl.colors.Normalize(vmin=r.min(), vmax=r.max())
        colors = cm(norm(r))
        vrange = [r.min() / 0.1, r.max() / 0.1]
    else:
        colors = cm(traj.radial)
    return cm, colors, vrange


def apply_colorbar(ax, cm, vrange):
    v_min, v_max = vrange
    norm = mpl.colors.Normalize(vmin=v_min, vmax=v_max)
    sm = plt.cm.ScalarMappable(cmap=cm, norm=norm)
    divider = make_axes_locatable(ax)
    colorbar_axes = divider.append_axes("right", size="5%", pad=0.1)
    cb = plt.colorbar(sm, cax=colorbar_axes, orientation="vertical")
    cb.set_ticks([v_min, v_max])
    return cb


def sample_image(ax, trajectories, option="radial", break_point=25):
    counter = 0
    vrange, cm = None, None
    for trajectory in trajectories:
        x = trajectory.x
        y = trajectory.y
        ex = trajectory.ex
        ey = trajectory.ey
        cm, colors, vrange = color_map(trajectory, option)

        R = 0.1 / 2
        l = 6 * R * np.ones(x.size)
        snaps.iterateRods(ax, colors, x, y, ex, ey, l, None, R, resp=0.01)
        ax.scatter(x, y, c=colors, s=1)

        if counter > break_point:
            break
        counter += 1
    return cm, vrange


fig, (ax, sub) = plt.subplots(ncols=2, figsize=(20, 10))

cm, vrange = sample_image(ax, trajectories, "front", break_point=100)
cb = apply_colorbar(ax, cm, vrange)

opts, times = DataModel.extract_times(sample_path)
_data, _ = ancestorTracker.fetch_data(sample_path, times)
norm = mpl.colors.Normalize(vmin=0, vmax=1)
c = get_cmap("coolwarm")(norm(_data.color2))

sub.scatter(_data.x, _data.y, s=2, c=c)

cb.set_label("Radial Alignment", fontsize=20)
ax.set_ylabel("y", fontsize=20)
ax.set_xlabel("x", fontsize=20)
plt.show()


# %%
def trajectory_radial_order(trajectories, radii):
    avgs = np.empty(len(trajectories))
    for i in range(len(trajectories)):
        #avgs[i] = np.mean(trajectories[i].radial)
        r = np.hypot(trajectories[i].x, trajectories[i].y)
        avgs[i] = np.mean(radii - r) / 0.1
    mask = np.where(np.array([t.population for t in trajectories]) == 0)
    mask_2 = np.where(np.array([t.population for t in trajectories]) == 1)
    return avgs[mask], avgs[mask_2]


avgs = trajectory_radial_order(trajectories, radii)

fig, ax = plt.subplots()
for i in range(2):
    a = df[f"max_aspect_ratio_{i+1}"].values[3]
    ax.hist(
        avgs[i],
        bins=50,
        histtype="barstacked",
        density=False,
        alpha=0.5,
        label=f"Population {i} (a = {a})",
    )

ax.set_xlabel("trajectory avg front distance order $(d_0)$", fontsize=16)
ax.set_ylabel("pdf", fontsize=20)
ax.legend(fontsize=16)
plt.show()
# %%
