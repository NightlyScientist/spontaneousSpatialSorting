# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd
from importlib import reload
import tools.dataModels.measurements as _measurements

reload(_measurements)
from tools.dataModels.measurements import Measurements

# %%
dataPath = input()

# %%
opts, times = Measurements.extract_times(dataPath)

# %%
snapshot = Measurements(dataPath=dataPath, time=times[15])
snapshot.nematicField(n=100)

# %%
import ipympl

# %%
import tools.graphics.nematics as nematics

reload(nematics)

nematics.fieldSnapshot(
    snapshot.n_x,
    snapshot.n_y,
    snapshot.Q11,
    snapshot.Q12,
    snapshot.n_vx,
    snapshot.n_vy,
    snapshot.x,
    snapshot.y,
)

# %%
import tools.tracking.defectTracker as Tracker
reload(Tracker)

option = "v"
grid_size = snapshot.grid_size
xyz = np.array([grid_size, grid_size, 1])

#. director field and S field from Q tensor
n_ex, n_ey, s_array = Tracker.directorField(snapshot.Q11, snapshot.Q12)

#. Q-tensor field coverted to 4-D arrays
Q, S, director = Tracker.QTensorSDirector(snapshot.Q11, snapshot.Q12, n_ex, n_ey, xyz, s_array)

#%%
#. two-dimensional D tensor
D = Tracker.calcD_2D(xyz, Q)

#. scalar-splay bend parameter field
ssb = Tracker.calcSSB(xyz, Q)

#. group defect candidates 
cufoffsd = 4.5
defectlistD = Tracker.extractSingularities(xyz, D, cufoffsd)

#. convert defect list to angles
defect_angles = Tracker.findNematicOrientation(xyz, defectlistD, Q)

# .Convert defect list to class defects
defects = Tracker.classAPI(defectlistD)


# %%
#infile,ts,vel = readVelTimeStep(velF, infile, xyz)
#velMean, velMag = fieldMean(vel,[0.],xyz,'z',minmax)

#QCdirect = Tracker.calcQC(xyz, vel)
#VGT = Tracker.calcVGT(xyz, vel)
#QCinv = Tracker.calcInvar2(xyz, VGT)

from tools.tracking.drawings import *

figure, ax = set_up_figure(xyz)
plt.xlim(0, xyz[0])
plt.ylim(0, xyz[1])

#background = plot_background_scalar_field(sca, xyz, ed.plasma, [0, 1], 1)
#plot_director_field(dir, xyz, 2, white)

#for defect in defectlistD:
#    drawNematicDefect(defect, 1.2)

#divider = make_axes_locatable(ax)
#cbar_axes = divider.append_axes("right", size="6%", pad=0.05)
#CB = plt.colorbar(background, cax=cbar_axes)
## CB.set_label('Splay-bend, $S_{\mathrm{SB}}$')
#CB.set_label("Scalar order, $S$")
#CB.remove()
plt.show()

#%%
#######################################################

figure, ax = set_up_figure(xyz)
plt.xlim(0, xyz[0])
plt.ylim(0, xyz[1])
background = plot_background_scalar_field(SSB, xyz, mapNemW, [-0.6, 0.6], 1)
plot_director_field(dir, xyz, 2, saphire)

for defect in defectlistD:
    # abc = 1
    drawNematicDefect(defect, 1.2)

divider = make_axes_locatable(ax)
cbar_axes = divider.append_axes("right", size="6%", pad=0.05)
CB = plt.colorbar(background, cax=cbar_axes)
# CB.set_label('Splay-bend, $S_{\mathrm{SB}}$')
CB.set_label("Splay-bend, $S_{\mathrm{SB}}$")

#######################################################
figure, ax = set_up_figure(xyz)
plt.xlim(0, xyz[0])
plt.ylim(0, xyz[1])
if option == "v":
    velMagPL = np.squeeze(velMag)
    background = plot_background_scalar_field(velMagPL, xyz, ed.viridis, "unknown", 1)
    plot_velocity_field(vel, xyz, 2, silver, 15)
elif option == "f":
    flwMagPL = np.squeeze(flowMag)
    background = plot_background_scalar_field(flwMagPL, xyz, ed.viridis, "unknown", 1)
    plot_velocity_field(flow, xyz, 2, silver, 15)
for defect in defectlistD:
    # abc = 1
    drawNematicDefect(defect, 1.2)

divider = make_axes_locatable(ax)
cbar_axes = divider.append_axes("right", size="6%", pad=0.05)
CB = plt.colorbar(background, cax=cbar_axes)

#######################################################
figure, ax = set_up_figure(xyz)
plt.xlim(0, xyz[0])
plt.ylim(0, xyz[1])
background = plot_background_scalar_field(
    QCdirect,
    xyz,
    mapQC,
    [-0.6 * np.max(abs(QCdirect)), 0.6 * np.max(abs(QCdirect))],
    1,
)
if option == "v":
    plot_velocity_field(vel, xyz, 2, silver, 15)
elif option == "f":
    plot_velocity_field(flow, xyz, 2, silver, 15)
for defect in defectlistD:
    # abc = 1
    drawNematicDefect(defect, 1.2)
# Plot rod
if len(current_rod.monomers[0]) == 1:
    plt.scatter(current_rod.monomers[0], current_rod.monomers[1], color=silver, s=50)
else:
    plt.plot(
        current_rod.monomers[0], current_rod.monomers[1], color=saphire, linewidth=3
    )

divider = make_axes_locatable(ax)
cbar_axes = divider.append_axes("right", size="6%", pad=0.05)
CB = plt.colorbar(background, cax=cbar_axes)
 CB.set_label('Splay-bend, $S_{\mathrm{SB}}$')

if option == "v":
    CB.set_label("$\mathcal{Q}$-criterion")
    plt.savefig(path + "movies/velQCdirectframe%04d.png" % (t), transparent=True)
    # plt.savefig(path+'movies/velQCdirectframe%04d.eps'%(t),format='eps')
elif option == "f":
    CB.set_label("$\mathcal{Q}$-criterion")
    plt.savefig(path + "movies/flwQCdirectframe%04d.png" % (t), transparent=True)
    # plt.savefig(path+'movies/flwQCdirectframe%04d.eps'%(t),format='eps')

plt.close()
CB.remove()

#######################################################
figure, ax = set_up_figure(xyz)
plt.xlim(0, xyz[0])
plt.ylim(0, xyz[1])

background = plot_background_scalar_field(
    QCinv, xyz, mapQC, [-0.6 * np.max(abs(QCinv)), 0.6 * np.max(abs(QCinv))], 1
)

plot_velocity_field(vel, xyz, 2, silver, 15)
plot_velocity_field(flow, xyz, 2, silver, 15)

for defect in defectlistD:
    # abc = 1
    drawNematicDefect(defect, 1.2)

divider = make_axes_locatable(ax)
cbar_axes = divider.append_axes("right", size="6%", pad=0.05)
CB = plt.colorbar(background, cax=cbar_axes)
