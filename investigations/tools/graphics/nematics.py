import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as interpolate
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numpy import linalg
from numba import jit
from skimage.feature import peak_local_max
import skimage


@jit(nopython=True, fastmath=True)
def get_field(xg, yg, Q11, Q12):
    nx = np.zeros(xg.size)
    ny = np.zeros(xg.size)
    S = np.zeros(xg.size)

    gridsize = xg.size
    grid = int(np.sqrt(gridsize))
    for i in range(Q11.size):
        v, w = linalg.eig(np.array([[Q11[i], Q12[i]], [Q12[i], -Q11[i]]]))
        S[i] = np.hypot(Q11[i], Q12[i])
        nx[i] = w[np.argmax(v)][0]
        ny[i] = w[np.argmax(v)][1]
    return S, nx, ny


@jit(nopython=True, fastmath=True)
def vorticity(xg, yg, Q11, Q12, vx, vy):
    d1Q11 = np.zeros(Q11.size)
    d2Q11 = np.zeros(Q11.size)
    d1Q12 = np.zeros(Q11.size)
    d2Q12 = np.zeros(Q11.size)
    vort = np.zeros(Q11.size)
    grid = int(np.sqrt(Q11.size))

    for i in range(0, grid):
        for j in range(0, grid):
            ind = i + j * grid
            indpi = (i + 1) % grid + j * grid
            indmi = (i - 1) % grid + j * grid
            indpj = i + ((j + 1) % grid) * grid
            indmj = i + ((j - 1) % grid) * grid

            d1Q11[ind] = -Q11[indmi] + Q11[indpi]
            d2Q11[ind] = -Q11[indmj] + Q11[indpj]
            d1Q12[ind] = -Q12[indmi] + Q12[indpi]
            d2Q12[ind] = -Q12[indmj] + Q12[indpj]
            vort[ind] = (-vy[indmi] + vy[indpi]) - (-vx[indmj] + vx[indpj])
    return vort


def fieldSnapshot(xg, yg, Q11, Q12, vx, vy, xp, yp, streamers=False):

    # .director and nematic order parameter
    S, nx, ny = get_field(xg, yg, Q11, Q12)

    xi = np.linspace(xg.min(), xg.max(), 100)
    yi = np.linspace(yg.min(), yg.max(), 100)
    X, Y = np.meshgrid(xi, yi)

    cmap = plt.colormaps.get_cmap("coolwarm")
    cmap.set_under("w", alpha=0)

    fig, (ax, sub) = plt.subplots(nrows=2, figsize=(6, 12))
    C = interpolate.griddata((xg, yg), S, (X, Y), method="linear")
    im = ax.pcolormesh(X, Y, C, shading="gouraud", cmap=cmap, vmin=0.001, vmax=1.0)

    divider = make_axes_locatable(ax)
    colorbar_axes = divider.append_axes("right", size="3%", pad=0.1)
    cb = plt.colorbar(
        cax=colorbar_axes, mappable=im, label="nematic order parameter $S$"
    )

    ax.quiver(xg, yg, nx, ny, pivot="mid", scale=60.0, headaxislength=0, color="black")

    tmax = np.hypot(xp, yp).max()
    ax.set_xlim([-tmax, tmax])
    ax.set_ylim([-tmax, tmax])

    vort = vorticity(xg, yg, Q11, Q12, vx, vy)

    C = interpolate.griddata((xg, yg), vort, (X, Y), method="cubic")

    vmin, vmax = -max(abs(vort)), max(abs(vort))
    im2 = sub.pcolormesh(
        X, Y, C, shading="gouraud", cmap="seismic", vmin=vmin, vmax=vmax
    )

    neg_x, neg_y, neg_px, neg_py, pos_x, pos_y, pos_px, pos_py = defects_from_Q(
        xg, yg, Q11, Q12
    )

    sub.set_xlim([-tmax, tmax])
    sub.set_ylim([-tmax, tmax])

    fltr = np.where(np.hypot(neg_x, neg_y) < tmax)
    ax.scatter(neg_x[fltr], neg_y[fltr], color="red", s=100, marker="^")
    fltr = np.where(np.hypot(pos_x, pos_y) < tmax)
    ax.scatter(pos_x[fltr], pos_y[fltr], color="blue", s=100, marker="2")

    vabs = np.sqrt(vx * vx + vy * vy)

    divider = make_axes_locatable(sub)
    colorbar_axes = divider.append_axes("right", size="3%", pad=0.1)
    cb = plt.colorbar(
        cax=colorbar_axes, mappable=im2, label="nematic order parameter $S$"
    )

    if streamers:
        Uth = interpolate.griddata((xg, yg), vx, (X, Y), method="linear")
        Vth = interpolate.griddata((xg, yg), vy, (X, Y), method="linear")
        sub.streamplot(
            X, Y, Uth, Vth, linewidth=1.0, density=4.0, color="black", arrowsize=1.0
        )
    return fig, ax, sub


@jit(nopython=True, fastmath=True)
def _QGrid(Q11, Q12, delx):
    grid = int(np.sqrt(Q11.size))

    phi = np.zeros(Q11.size)
    nx = np.zeros(Q11.size)
    ny = np.zeros(Q11.size)
    S = np.zeros(Q11.size)

    q = np.zeros(Q11.size)
    qgrid = np.zeros((grid, grid))
    delta = 1.0 / 2.0 * delx
    polx = np.zeros(Q11.size)
    poly = np.zeros(Q11.size)

    for i in range(Q11.size):
        S[i] = np.sqrt(Q11[i] * Q11[i] + Q12[i] * Q12[i])
        phi[i] = 0.5 * np.arctan2(Q12[i], Q11[i])
        if phi[i] > np.pi:
            phi[i] -= np.pi
        if phi[i] < 0:
            phi[i] += np.pi
        nx[i] = np.cos(phi[i])
        ny[i] = np.sin(phi[i])
        phi[i] = np.arctan2(ny[i], nx[i])

    curln_zcomp = np.zeros(Q11.size)

    for i in range(1, grid - 1):
        for j in range(1, grid - 1):
            ind = i + j * grid
            indpi = i + 1 + j * grid
            indmi = i - 1 + j * grid
            indpj = i + (j + 1) * grid
            indmj = i + (j - 1) * grid
            curln_zcomp[ind] = (ny[indpi] - ny[indmi]) * delta - (
                nx[indpj] - nx[indmj]
            ) * delta
            q[ind] = (
                1.0
                / (2.0 * np.pi)
                * (
                    (Q11[indpi] - Q11[indmi])
                    * delta
                    * (Q12[indpj] - Q12[indmj])
                    * delta
                    - (Q12[indpi] - Q12[indmi])
                    * delta
                    * (Q11[indpj] - Q11[indmj])
                    * delta
                )
            )
            qgrid[i, j] = q[ind]
            l_polx = (Q11[indpi] - Q11[indmi]) * delta + (
                Q12[indpj] - Q12[indmj]
            ) * delta
            l_poly = (Q12[indpi] - Q12[indmi]) * delta - (
                Q11[indpj] - Q11[indmj]
            ) * delta
            l_polabs = np.sqrt(l_polx * l_polx + l_poly * l_poly)
            if l_polabs != 0:
                polx[ind] = -l_polx / l_polabs
                poly[ind] = -l_poly / l_polabs
    return nx, ny, S, phi, curln_zcomp, qgrid, polx, poly, q


def get_peak_coordinates(qgrid, min_distance=1):
    return skimage.feature.peak_local_max(abs(qgrid), min_distance=min_distance)


def defects_from_Q(xg, yg, Q11, Q12, min_distance=1):
    gridsize = xg.size
    grid = int(np.sqrt(gridsize))

    delx = xg[1] - xg[0]

    nx, ny, S, phi, curln_zcomp, qgrid, polx, poly, q = _QGrid(Q11, Q12, delx)

    ind = np.where(S > 0.0)

    S = S[ind]
    nx = nx[ind]
    ny = ny[ind]
    curln_zcomp = curln_zcomp[ind]

    coordinates = get_peak_coordinates(abs(qgrid), min_distance=min_distance)
    single_coord = coordinates[:, 0] + grid * coordinates[:, 1]

    rmcounter = 0
    indlist = [
        (0, -2),
        (1, -2),
        (2, -1),
        (2, 0),
        (2, 1),
        (1, 2),
        (0, 2),
        (-1, 2),
        (-2, 1),
        (-2, 0),
        (-2, -1),
        (-1, -2),
        (0, -2),
    ]
    for k in range(single_coord.size):
        i = coordinates[k, 0]
        j = coordinates[k, 1]
        wind_num = 0
        phiold = phi[i + grid * (j - 2)]
        for m in indlist:
            ik = m[0]
            jk = m[1]
            nowind = i + ik + grid * (j + jk)
            if nowind < gridsize:
                if abs(phi[nowind] - phiold) > 2.0:
                    wind_num += 1
                phiold = phi[nowind]
        if wind_num != 1.0:
            single_coord = np.delete(single_coord, k - rmcounter)
            rmcounter += 1

    xq = xg[single_coord]
    yq = yg[single_coord]
    xg = xg[ind]
    yg = yg[ind]
    phi = phi[ind]

    polxmax = polx[single_coord]
    polymax = poly[single_coord]
    qmax = q[single_coord]
    poscharge = np.where(qmax > 0)
    negcharge = np.where(qmax < 0)
    return (
        xq[negcharge],
        yq[negcharge],
        polxmax[negcharge],
        polymax[negcharge],
        xq[poscharge],
        yq[poscharge],
        polxmax[poscharge],
        polymax[poscharge],
    )


def lyupanov_exponent(nx, ny, Q11, Q12, vx, vy):
    d1vx = np.zeros(np.size(Q11))
    d2vx = np.zeros(np.size(Q11))
    d1vy = np.zeros(np.size(Q11))
    d2vy = np.zeros(np.size(Q11))

    lyapunov_field = np.zeros(np.size(Q11))

    grid = int(np.sqrt(np.size(Q11)))

    for i in range(0, grid):
        for j in range(0, grid):
            ind = i + j * grid
            indpi = (i + 1) % grid + j * grid
            indmi = (i - 1) % grid + j * grid
            indpj = i + ((j + 1) % grid) * grid
            indmj = i + ((j - 1) % grid) * grid

            d1vx[ind] = -vx[indmi] + vx[indpi]
            d2vx[ind] = -vx[indmj] + vx[indpj]
            d1vy[ind] = -vy[indmi] + vy[indpi]
            d2vy[ind] = -vy[indmj] + vy[indpj]

            lyapunov_field[ind] = (
                nx[ind] ** 2 * d1vx[ind]
                + nx[ind] * ny[ind] * (d2vx[ind] + d1vy[ind])
                + ny[ind] ** 2 * d2vy[ind]
            )

    # lyapunov_exps=[]
    # lyapunov_exps = np.hstack((lyapunov_exps, np.mean(lyapunov_field)))
    # return lyapunov_exps
    return lyapunov_field


# def defect_density(xg, yg, Q11, Q12):
# xn, yn, exn, eyn, xp, yp, ex_p, ey_p = defects_from_Q(xg, yg, Q11, Q12)

# def extens(self):
#     gridsize = np.size(xg)
#     grid = int(np.sqrt(gridsize))

#     nx = np.zeros(gridsize)
#     ny = np.zeros(gridsize)
#     S = np.zeros(gridsize)
#     phi = np.zeros(gridsize)

#     for i in range(0, gridsize):
#         S[i] = np.sqrt(self.Q11[i] * self.Q11[i] + self.Q12[i] * self.Q12[i])
#         phi[i] = 0.5 * np.arctan2(self.Q12[i], self.Q11[i])
#         if phi[i] > np.pi:
#             phi[i] -= np.pi
#         if phi[i] < 0:
#             phi[i] += np.pi
#         nx[i] = np.cos(phi[i])
#         ny[i] = np.sin(phi[i])
#         phi[i] = np.arctan2(ny[i], nx[i])

#     ind = np.where(S > 0.0)

#     S = S[ind]
#     rho = rho[ind]
#     nx = nx[ind]
#     ny = ny[ind]
#     xg = xg[ind]
#     yg = yg[ind]
#     phi = phi[ind]

#     Q11 = self.Q11
#     Q12 = self.Q12
#     d1Q11 = np.zeros(np.size(Q11))
#     d2Q11 = np.zeros(np.size(Q11))
#     d1Q12 = np.zeros(np.size(Q11))
#     d2Q12 = np.zeros(np.size(Q11))
#     vort = np.zeros(np.size(Q11))

#     grid = int(np.sqrt(np.size(Q11)))
#     for i in range(0, grid):
#         for j in range(0, grid):
#             ind = i + j * grid
#             indpi = (i + 1) % grid + j * grid
#             indmi = (i - 1) % grid + j * grid
#             indpj = i + ((j + 1) % grid) * grid
#             indmj = i + ((j - 1) % grid) * grid

#             d1Q11[ind] = Q11[indmi] - 2.0 * Q11[ind] + Q11[indpi]
#             d2Q11[ind] = Q11[indmj] - 2.0 * Q11[ind] + Q11[indpj]
#             d1Q12[ind] = Q12[indmi] - 2.0 * Q12[ind] + Q12[indpi]
#             d2Q12[ind] = Q12[indmj] - 2.0 * Q12[ind] + Q12[indpj]
#             vort[ind] = (vy[indmi] - 2.0 * vy[ind] + vy[indpi]) - (
#                 vx[indmj] - 2.0 * vx[ind] + vx[indpj]
#             )
#     fx = d1Q11 + d2Q12
#     fy = d1Q12 - d2Q11

#     ind = np.where((yg > -4.0) & (yg < 4.0) & (xg > -4.0) & (xg < 4.0))

#     fx = fx[ind]
#     fy = fy[ind]
#     vx = vx[ind]
#     vy = vy[ind]
#     return np.sum(fx * vx + fy * vy)


# def velocity(xg, yg, rho, vx, vy, press, time, delx=0.5):
# com_x = np.mean(xg)
# com_y = np.mean(yg)
# rad_dist = np.arange(2.0 * delx, max(xg[np.where(rho > 0)]) - 1.0, delx)

# Press_R = np.zeros(np.size(rad_dist))
# norm = np.zeros(np.size(rad_dist))
# V_r = np.zeros(np.size(rad_dist))
# V_phi = np.zeros(np.size(rad_dist))
# for i in range(0, np.size(xg)):
#     l_x = com_x - xg[i]
#     l_y = com_y - yg[i]
#     dist = np.sqrt((l_x) ** 2 + (l_y) ** 2)
#     angle = np.arctan2(l_y, l_x) + np.pi
#     ind = np.where(((rad_dist - delx / 2) < dist) & ((rad_dist + delx / 2) > dist))
#     if np.size(ind[0]) > 0:
#         Press_R[ind[0][0]] = Press_R[ind[0][0]] + press[i]
#         absv = np.sqrt(vx[i] ** 2 + vy[i] ** 2)
#         V_r[ind[0][0]] = vx[i] * np.cos(angle) + vy[i] * np.sin(angle)
#         V_phi[ind[0][0]] = -vx[i] * np.sin(angle) + vy[i] * np.cos(angle)
#         norm[ind[0][0]] = norm[ind[0][0]] + 1
# Press_R = Press_R / (norm)
# # avr_vel.append(np.mean((V_r)))
# # avr_vel_phi.append(np.mean((V_phi)))
# # the_times.append(time)


# def pressure(xg, yg, rho, press, Q11, Q12, vx, vy, delx=0.5):
# com_x = np.mean(xg)
# com_y = np.mean(yg)
# rad_dist = np.arange(2.0 * delx, max(xg[np.where(rho > 0)]) - 1.0, delx)
# Press_R = np.zeros(np.size(rad_dist))
# norm = np.zeros(np.size(rad_dist))
# V_r = np.zeros(np.size(rad_dist))
# V_phi = np.zeros(np.size(rad_dist))

# for i in range(0, np.size(xg)):
#     l_x = com_x - xg[i]
#     l_y = com_y - yg[i]
#     dist = np.sqrt((l_x) ** 2 + (l_y) ** 2)
#     angle = np.arctan2(l_y, l_x) + np.pi
#     ind = np.where(((rad_dist - delx / 2) < dist) & ((rad_dist + delx / 2) > dist))
#     if np.size(ind[0]) > 0:
#         Press_R[ind[0][0]] = Press_R[ind[0][0]] + press[i]
#         absv = np.sqrt(vx[i] ** 2 + vy[i] ** 2)
#         V_r[ind[0][0]] += vx[i] * np.cos(angle) + vy[i] * np.sin(angle)
#         V_phi[ind[0][0]] += -vx[i] * np.sin(angle) + vy[i] * np.cos(angle)
#         norm[ind[0][0]] = norm[ind[0][0]] + 1
# Press_R = Press_R / (norm)
# V_r = V_r / norm
# V_phi = V_phi / norm
