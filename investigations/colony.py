# %%
from ipywidgets import interact, fixed
import matplotlib.pyplot as plt
from importlib import reload
from tools.dataAPI.datamodel import DataModel
import tools.graphics.snapshots as snaps
reload(snaps)

plt.style.use("fast")
#%%
import ipympl

# %%
dataPath = input()

opts, times = DataModel.extract_times(dataPath)
colorOptions = [
    "angle",
    "population",
    "allele",
    "smectic",
    "heterozygosity",
    "nematic",
    "splits",
    "radial",
    "velocity",
    "length",
    "grains",
    "flows"
]

# %%
interact(
    snaps.drawRods,
    time=times,
    datapath=fixed(dataPath),
    imgPath=fixed(""),
    colorby=colorOptions,
)

# %%
interact(
    snaps.gridSnapshots,
    time=times,
    datapath=fixed(dataPath),
    imgPath=fixed(""),
    colorby=colorOptions,
)

# %%
interact(
    snaps.flowfield,
    time=times,
    datapath=fixed(dataPath),
    imgPath=fixed(""),
    colorby=colorOptions,
    pxNumber=fixed(100)
)