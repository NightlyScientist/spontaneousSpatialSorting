# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import pandas as pd
from importlib import reload
import tools.dataModels.measurements as _measurements
import warnings

reload(_measurements)
from tools.dataModels.measurements import Measurements

# %%
dataPath = input()

# %%
opts, times = Measurements.extract_times(dataPath)
labels = np.round(
    np.array([opts["division_lengths_1"], opts["division_lengths_2"]]).astype(float), 2
).astype(str)

# %%
# >Angle Distribution
snapshot = Measurements(dataPath=dataPath, time=times[-1])
angles = snapshot.phi

# .distance from center and segments
center = np.array([snapshot.x.mean(), snapshot.y.mean()])
distance = np.hypot(snapshot.x - center[0], snapshot.y - center[1])
segments = np.array([0, 0.6, 0.85, 1]) * distance.max()

dangle = np.arctan2(snapshot.ey, snapshot.ex) - np.arctan2(snapshot.y, snapshot.x)
rcorr = np.cos(2 * (dangle))

fig, (ax, sub) = plt.subplots(nrows=2, figsize=(7, 10))

for i in np.unique(snapshot.color2):
    bins = np.linspace(0, 2 * np.pi, 20)
    hist, bin_edges = np.histogram(angles[snapshot.color2 == i], bins, density=True)
    ax.plot(bin_edges[:-1], hist, label=labels[i])

for i, label in zip(range(1, len(segments)), ["homeland", "main", "periphery"]):
    bins = np.linspace(-1, 1, 20)
    low = segments[i - 1]
    high = segments[i]
    hist, bin_edges = np.histogram(
        rcorr[np.where((low <= distance) & (distance <= high))], bins, density=True
    )
    sub.plot(bin_edges[:-1], hist, label=label)

ax.title.set_text(f"Angle Distribution")
sub.title.set_text(f"Radial Alignment")
for axis in [ax, sub]:
    axis.set_xlabel(r"$\theta$")
    axis.set_ylabel("1/N")
    axis.legend(loc="upper right")

# %%
# .radial order vs radius
snapshot = Measurements(dataPath=dataPath, time=times[-1])
radii = np.hypot(snapshot.x, snapshot.y)
bins = np.linspace(0, radii.max(), 25)

radial = snapshot.radial_alignment()
for i, label in zip(range(2), labels):
    r_sub = radial[snapshot.color2 == i]
    digitized = np.digitize(radii[snapshot.color2 == i], bins)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        bin_means = [r_sub[digitized == j].mean() for j in range(0, len(bins))]
        sub.plot(bins, bin_means, label=label, marker=".")

sub.set_xlabel("radius")
sub.set_ylabel("|(r, e)|")
sub.legend()

# >auto correlation of nematic alignment
snapshot = Measurements(dataPath=dataPath, time=times[-1])
cnn = snapshot.Cnn(nbins=100)

fig, ax = plt.subplots()

l_0 = np.mean([snapshot.division_length_1, snapshot.division_length_2])

ax.scatter(cnn[:, 0] / l_0, cnn[:, 1], c="blue", label="Cnn", marker=".")

ax.title.set_text(f"Nematic alignment auto-correlation")
ax.set_ylim(0, 1)
ax.set_xlabel("radius / $l_{mean}$")
ax.set_ylabel("$C_{nn}(r)$")
ax.legend(loc="upper right")

# task: aspect ratio vs radius (from the center