import matplotlib.pyplot as plt
from pylab import *
from tools.tracking.colourSheet import *
from colourSheet import *

plt.rcParams.update({"lines.markeredgewidth": 2})


def set_up_figure(xyz):
    fig = plt.figure()
    fig.set_tight_layout("tight")
    ax = fig.add_subplot(111, aspect="equal")
    ax.set_aspect("equal")
    plt.cla()
    plt.xticks([])
    plt.yticks([])
    return fig, ax

def drawNematicDefect(defect, multiplier):
    # Make option for centre circle with white inner circle
    # 3 lines for minus half
    # 1 line for plus half
    # the outer circle and lines are coloured by defect code -- limegreen for +1/2 and -- saphire for -1/2

    def drawLine(coordinate, theta, dtheta, distanceStart, length, colour, linewidth):
        xstart = coordinate[0] + distanceStart * np.cos(theta + dtheta)
        ystart = coordinate[1] + distanceStart * np.sin(theta + dtheta)
        xend = coordinate[0] + (distanceStart + length) * np.cos(theta + dtheta)
        yend = coordinate[1] + (distanceStart + length) * np.sin(theta + dtheta)

        return plt.plot(
            [xstart, xend], [ystart, yend], color=colour, lw=linewidth, zorder=200
        )

    def drawCircle(coordinate, radius, colour, linewidth):
        allAngles = np.linspace(0, 2 * np.pi, 150)

        x = coordinate[0] + radius * np.cos(allAngles)
        y = coordinate[1] + radius * np.sin(allAngles)

        plt.fill(np.ravel(x), np.ravel(y), "w", zorder=201)
        return plt.plot(x, y, color=colour, lw=linewidth, zorder=202)

    length = 1.5 * multiplier
    radius = 0.5 * multiplier
    linewidth = 2 * multiplier  # *multiplier/10
    if multiplier == 2:
        length = 1.2 * multiplier
        radius = 1 * multiplier
        linewidth = 1.5 * multiplier  # *multiplier/10

    if sign(defect[1]) > 0:
        # +1/2 defect
        colour = limegreenLCH

        drawLine(defect[0], defect[3], 0, radius, length, colour, linewidth)
        drawCircle(defect[0], radius, colour, linewidth)
    if sign(defect[1]) < 0:
        # -1/2 defect
        colour = saphire

        drawLine(defect[0], defect[3], 0.0, radius, length, colour, linewidth)
        drawLine(
            defect[0], defect[3], (2 / 3) * np.pi, radius, length, colour, linewidth
        )
        drawLine(
            defect[0], defect[3], (4 / 3) * np.pi, radius, length, colour, linewidth
        )

        drawCircle(defect[0], radius, colour, linewidth)
    

def plot_background_scalar_field(field, xyz, colormap, limits, axesMultiplier):
    if limits == "unknown":
        # Find axes scaling
        mx = np.max(field)
        mn = np.min(field)
        if abs(mn) > mx:
            mx = abs(mn)
        axes_scale = axesMultiplier * mx

        if mn >= 0:
            axes_scale_lower = mn
        if mn < 0:
            axes_scale_lower = -axesMultiplier * mx  # -1.3*mx
    elif limits == "lower":
        axes_scale_lower = 0.0
        axes_scale = axesMultiplier * mx
    else:
        axes_scale = limits[1]
        axes_scale_lower = limits[0]

    # Plot colour-map
    image = imshow(
        field.T,
        cmap=colormap,
        origin="lower",
        aspect="equal",
        vmin=axes_scale_lower,
        vmax=axes_scale,
    )

    return image


def plot_director_field(field, xyz, resolution, color):
    # Make field fully 2D
    minmax = [999999999.0, 0.0]
    MEAN, vMAG = fieldMean(field, [0.0], xyz, "z", minmax)

    # Make mesh
    XYZ = formMesh(xyz)
    XY = makeMesh(XYZ, xyz, "z")

    old_resolution = resolution

    # Director field
    spacing = resolution
    if resolution > 5:
        c = 0.75 * resolution
    else:
        c = 1.5
    if resolution == 2 or resolution == 3:
        resolution = 0
    # for a in range(int((1/spacing)*(xyz[0]+1))):
    for a in range(int((1 / spacing) * (xyz[0]))):
        # for b in range(int((1/spacing)*(xyz[1]+1))):
        for b in range(int((1 / spacing) * (xyz[1]))):
            plot(
                [
                    XY[0][spacing * a][spacing * b]
                    - c * MEAN[0][spacing * a][spacing * b]
                    + 0.5 * resolution,
                    XY[0][spacing * a][spacing * b]
                    + c * MEAN[0][spacing * a][spacing * b]
                    + 0.5 * resolution,
                ],
                [
                    XY[1][spacing * a][spacing * b]
                    - c * MEAN[1][spacing * a][spacing * b]
                    + 0.5 * resolution,
                    XY[1][spacing * a][spacing * b]
                    + c * MEAN[1][spacing * a][spacing * b]
                    + 0.5 * resolution,
                ],
                color=color,
                alpha=0.8,
                linewidth=1.0,
                zorder=0.6,
            )  # previously alpha was 0.2
    resolution = old_resolution


def plot_velocity_field(field, xyz, resolution, color, scale):
    # Make field fully 2D
    minmax = [999999999.0, 0.0]
    vMEAN, MAG = fieldMean(field, [0.0], xyz, "z", minmax)

    # Make mesh
    XYZ = formMesh(xyz)
    XY = makeMesh(XYZ, xyz, "z")

    spacing = resolution
    quiv = quiver(
        XY[0][::spacing, ::spacing],
        XY[1][::spacing, ::spacing],
        vMEAN[0][::spacing, ::spacing],
        vMEAN[1][::spacing, ::spacing],
        scale=scale,
        color=color,
        linewidths=15,
        zorder=0.5,
        alpha=0.5,
    )
    return quiv