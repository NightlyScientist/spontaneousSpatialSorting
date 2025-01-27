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


def colorSelector(colorby, data):
    vrange = np.array([0, 1])

    if colorby == "smectic":
        cutoff = 2 * max(data.division_length_1, data.division_length_2)
        data.set_local_radius(cutoff)
        values = data.smectic()
        cm = get_cmap("coolwarm")
    elif colorby == "angle":
        vrange = np.array([0, 180])
        values = data.nematic_angle() / np.pi
        cm = get_cmap("hsv")
    elif colorby == "heterozygosity":
        values = data.Heterozygosity_local(touching=True)
        cm = get_cmap("coolwarm")
    elif colorby == "nematic":
        values = data.nematic_local()
        cm = get_cmap("coolwarm")
    elif colorby == "splits":
        vrange = [0, np.max(data.splits)]
        values = data.splits / (np.max(data.splits) if np.max(data.splits) > 0 else 1)
        cm = get_cmap("Purples")
    elif colorby == "radial":
        values = data.radial_alignment()
        cm = get_cmap("coolwarm")
    elif colorby == "velocity":
        values = data.velocityProjection()
        cm = get_cmap("coolwarm")
    elif colorby == "length":
        lengths = np.array([data.division_length_1, data.division_length_2])
        vrange = [lengths.min() / 2, lengths.max()]
        values = (data.l - data.l.min()) / (data.l.max() - data.l.min())
        cm = get_cmap("coolwarm")
    elif colorby == "allele":
        allele = data.color
        vrange = [allele.min(), allele.max()]
        values = allele / allele.max()
        cm = get_cmap("hsv")
    elif colorby == "grains":
        grain_id, _, _, _ = data.grains()
        grain_new_id = grain_id

        kcal = 0
        for i in np.unique(grain_id):
            grain_new_id[np.where(grain_id == i)] = kcal
            kcal = kcal + 1

        values = grain_new_id / np.size(np.unique(grain_new_id))
        vrange = [grain_id.min(), grain_id.max()]
        cm = get_cmap("hsv")
    else:
        values = data.color2.astype(float)
        cm = get_cmap("copper")
    return cm, values, vrange


def extract_data(dataPath: str, time: int):
    data = Measurements(dataPath=dataPath, time=time)
    return data


def drawRods(
    time,
    datapath="",
    imgPath="",
    hidedecoration=False,
    saveImg=False,
    colorby="angle",
    show_defects=False,
    pixel_number=100,
):
    fig, ax = plt.subplots(figsize=(10, 10))
    fig.tight_layout()
    ax.set_aspect("equal")
    ax.tick_params(axis='both', labelsize=20)

    data = extract_data(datapath, time)
    cm, values, vrange = colorSelector(colorby, data)

    iterateRods(
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

    tmax = max(max(abs(data.x)), max(abs(data.y))) + 2 * data.thickness

    plt.axis([-tmax, tmax, -tmax, tmax])

    if show_defects:
        data.nematicField(n=pixel_number)

        neg_x, neg_y, neg_px, neg_py, pos_x, pos_y, pos_px, pos_py = defects_from_Q(
            data.n_x, data.n_y, data.Q11, data.Q12
        )
        ax.plot(neg_x, neg_y, "bx", markersize=25)
        ax.plot(pos_x, pos_y, "ro", markersize=25)

    # Normalizer
    v_min, v_max = vrange
    norm = mpl.colors.Normalize(vmin=v_min, vmax=v_max)
    sm = plt.cm.ScalarMappable(cmap=cm, norm=norm)
    divider = make_axes_locatable(ax)
    colorbar_axes = divider.append_axes("right", size="3%", pad=0.1)
    cb = plt.colorbar(sm, cax=colorbar_axes)
    cb.set_ticks([v_min, v_max])
    cb.ax.tick_params(labelsize=22)

    if hidedecoration:
        ax.set_xticks([])
        ax.set_yticks([])

    if imgPath != "" and saveImg:
        F = data.force_constant
        ax.title.set_text(
            f"red: {data.max_aspect_ratio_1}, green:{data.max_aspect_ratio_2}, @F = {F}"
        )
        plt.savefig(f"{imgPath}/{time}.png")
        plt.close(fig)
    else:
        plt.show()


def iterateRods(ax, coloring, x, y, ex, ey, l, phi, R, resp=0.01):
    r = R - resp

    for i in range(0, np.size(x)):
        color = coloring[i]
        x1, y1 = (x[i] - (0.5 * l[i] - r) * ex[i], y[i] - (0.5 * l[i] - r) * ey[i])
        x2, y2 = (x[i] + (0.5 * l[i] - r) * ex[i], y[i] + (0.5 * l[i] - r) * ey[i])

        angle = np.arctan2(ey[i], ex[i])
        epx = -np.sin(angle)
        epy = np.cos(angle)

        c1 = Circle((x1, y1), r, color=color)
        c2 = Circle((x2, y2), r, color=color)

        perp_angle = angle / (2.0 * np.pi) * 360.0

        xr = x[i] - (0.5 * l[i] - r) * ex[i] - epx * r
        yr = y[i] - (0.5 * l[i] - r) * ey[i] - epy * r
        rect = Rectangle(
            (xr, yr), width=l[i] - 2 * r, height=2 * r, angle=perp_angle, color=color
        )
        for artist in [c2, rect, c1]:
            ax.add_artist(artist)


def gridSnapshots(time: int, datapath: str, imgPath="", saveImg=False):
    data = extract_data(datapath, time)

    colorOptions = [
        "population",
        "length",
        "splits",
        "heterozygosity",
        "angle",
        "nematic",
        "radial",
        "angle",
    ]

    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(10, 6))
    fig.tight_layout()

    tmax = max(max(abs(data.x)), max(abs(data.y))) + max(
        data.division_length_1, data.division_length_2
    )

    for i, ax in enumerate(axes.reshape(-1)):
        ax.set_aspect("equal")
        ax.set_ylim([-tmax, tmax])
        ax.set_xlim([-tmax, tmax])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.title.set_text(f"{colorOptions[i]}")

        cm, values, vrange = colorSelector(colorOptions[i], data)

        iterateRods(
            ax, cm(values), data.x, data.y, data.ex, data.ey, data.l, data.phi, data.r
        )

        # Normalizer
        v_min, v_max = vrange
        norm = mpl.colors.Normalize(vmin=v_min, vmax=v_max)
        sm = plt.cm.ScalarMappable(cmap=cm, norm=norm)
        divider = make_axes_locatable(ax)
        colorbar_axes = divider.append_axes("bottom", size="5%", pad=0.1)
        cb = plt.colorbar(sm, cax=colorbar_axes, orientation="horizontal")
        cb.set_ticks([v_min, v_max])

    if imgPath != "" and saveImg:
        plt.savefig(f"{imgPath}/{time}.png")
        plt.close(fig)
    else:
        plt.show()


def surface(
    time,
    datapath="",
    imgPath="",
    hidedecoration=False,
    saveImg=False,
    show_surface=True,
):
    fig, ax = plt.subplots(figsize=(10, 10))
    fig.tight_layout()
    ax.set_aspect("equal")

    data = extract_data(datapath, time)

    cm = get_cmap("coolwarm")
    tmax = max(max(abs(data.x)), max(abs(data.y)))

    plt.axis([-tmax, tmax, -tmax, tmax])

    if show_surface:
        gaussian = partial(surf, amp=data.amplitude, wavelength=data.wavelength)
    else:
        gaussian = partial(K, amp=data.amplitude, wavelength=data.wavelength)
    x = np.linspace(-data.extent, data.extent, 100)
    y = np.linspace(-data.extent, data.extent, 100)
    X, Y = np.meshgrid(x, y)
    Z = gaussian(X, Y)
    ax.contourf(X, Y, Z, cmap=cm)

    ax.plot(
        data.x[np.where(data.defect_charge > 0)],
        data.y[np.where(data.defect_charge > 0)],
        "ro",
        markersize=10,
    )
    ax.plot(
        data.x[np.where(data.defect_charge < 0)],
        data.y[np.where(data.defect_charge < 0)],
        "bx",
        markersize=10,
    )

    # Normalizer
    v_min, v_max = Z.min(), Z.max()
    norm = mpl.colors.Normalize(vmin=v_min, vmax=v_max)
    sm = plt.cm.ScalarMappable(cmap=cm, norm=norm)
    divider = make_axes_locatable(ax)
    colorbar_axes = divider.append_axes("right", size="3%", pad=0.1)
    cb = plt.colorbar(sm, cax=colorbar_axes)
    cb.set_ticks([v_min, v_max])

    if hidedecoration:
        ax.set_xticks([])
        ax.set_yticks([])

    if imgPath != "" and saveImg:
        F = data.force_constant
        ax.title.set_text(
            f"red: {data.max_aspect_ratio_1}, green:{data.max_aspect_ratio_2}, @F = {F}"
        )
        plt.savefig(f"{imgPath}/{time}.png")
        plt.close(fig)
    else:
        plt.show()


def flowfield(
    time,
    datapath="",
    imgPath="",
    hidedecoration=False,
    saveImg=False,
    pxNumber=100,
    streamers=False,
):
    data = extract_data(datapath, time)
    data.nematicField(n=pxNumber)

    fig, ax, sub = fieldSnapshot(
        data.n_x,
        data.n_y,
        data.Q11,
        data.Q12,
        data.n_vx,
        data.n_vy,
        data.x,
        data.y,
        streamers=streamers,
    )
    if hidedecoration:
        for axis in [ax, sub]:
            axis.set_xticks([])
            axis.set_yticks([])

    if imgPath != "" and saveImg:
        F = data.force_constant
        ax.title.set_text(
            f"red: {data.max_aspect_ratio_1}, green:{data.max_aspect_ratio_2}, @F = {F}"
        )
        plt.savefig(f"{imgPath}/{time}.png")
        plt.close(fig)
    else:
        plt.show()


@jit(nopython=True)
def K(x, y, amp, wavelength):
    A = amp
    B = 2 * np.pi / wavelength

    up = A**2 * B**4 * (np.sin(B * x) * np.cos(B * y))
    dwn = 1 + A**2 * B**2 * (np.cos(B * x) ** 2 + np.sin(B * y) ** 2)
    return up / dwn


@jit(nopython=True)
def surf(x, y, amp, wavelength):
    return amp * (
        np.sin(2 * np.pi * x / wavelength) + np.cos(2 * np.pi * y / wavelength)
    )