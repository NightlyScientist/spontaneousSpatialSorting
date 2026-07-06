from tools.dataAPI.datamodel import DataModel
import numpy as np
from numba import jit
import copy


class Alignments(DataModel):

    def smectic(self):
        cutoff = self.local_radius**2

        @jit(nopython=True, fastmath=False)
        def f(x, y, ex, ey):
            smectic = np.zeros(x.size).astype(np.complex64)
            scounter = np.zeros(x.size).astype(np.complex64)

            for i in range(x.size):
                for j in range(x.size):
                    l_dist = (x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2

                    if l_dist < cutoff:
                        smectic[i] = smectic[i] + np.exp(
                            2.0j * np.pi * (ex[j] * ex[i] + ey[j] * ey[i])
                        )
                        scounter[i] += 1
            smectic = np.abs(smectic) / scounter
            return smectic.real

        return f(self.x, self.y, self.ex, self.ey)

    def nematic_angle(self, angles=None):
        if angles is None:
            angles = copy.copy(self.phi)

        angles[np.where(angles > np.pi)] = angles[np.where(angles > np.pi)] - np.pi
        return angles

    def nematic_global(self):
        """Global nematic order parameter"""
        ex = np.cos(self.phi)
        ey = np.sin(self.phi)
        q11 = 2 * ex * ex - 1
        q12 = 2 * ex * ey
        return np.sqrt(sum(q11) ** 2 + sum(q12) ** 2) / len(self.phi)

    def polar_global(self):
        """Global polar order parameter"""
        ex = np.sum(np.cos(self.phi))
        ey = np.sum(np.sin(self.phi))
        return np.sqrt(ex**2 + ey**2) / len(self.phi)

    def radial_alignment(self):
        com_x, com_y = np.mean(self.x), np.mean(self.y)
        rx, ry = self.x - com_x, self.y - com_y
        radii = np.sqrt(rx**2 + ry**2)
        mask = (radii > 0)
        return np.abs((rx * self.ex + ry * self.ey)[mask] / radii[mask])

    def nematic_local(self, local=True):
        cutoff = self.local_radius

        @jit(nopython=True, fastmath=False)
        def f(x, y, ex, ey, l, phi, cutoff, d0, local=False):
            q11 = np.zeros(x.size)
            q12 = np.zeros(x.size)
            ptclCounter = np.zeros(x.size)
            for i in range(len(x)):
                for j in range(i, len(x)):
                    if np.hypot(x[i] - x[j], y[i] - y[j]) >= cutoff:
                        continue

                    hx, hy = 1.0, 1.0
                    if local:
                        _hx, _hy = istouching(
                            x[i],
                            y[i],
                            ex[i],
                            ey[i],
                            x[j],
                            y[j],
                            ex[j],
                            ey[j],
                            l[i],
                            l[j],
                            d0,
                        )
                        hx = _hx
                        hy = _hy

                    if not (abs(hx) > 0 and abs(hy) > 0):
                        continue

                    _ex = np.cos(phi[j])
                    _ey = np.sin(phi[j])
                    q11[i] += 2 * _ex * _ex - 1
                    q12[i] += _ex * _ey
                    ptclCounter[i] += 1

                    if i != j:
                        _ex = np.cos(phi[i])
                        _ey = np.sin(phi[i])
                        q11[j] += 2 * _ex * _ex - 1
                        q12[j] += _ex * _ey
                        ptclCounter[j] += 1

            return np.sqrt(q11**2 + q12**2) / ptclCounter

        return f(
            self.x,
            self.y,
            self.ex,
            self.ey,
            self.l,
            self.phi,
            cutoff,
            self.thickness,
            local=local,
        )

    def g_nn(self, nbins):
        """Pair Nematic correlation function"""

        @jit(nopython=True, fastmath=True)
        def f(phi_1, phi_2, l_1, l_2):
            return 0.5 * (3 * np.cos(phi_1 - phi_2) ** 2 - 1)

        return self._autoCorrelation(
            f, self.phi, self.color2, self.x, self.y, nbins, "bin"
        )

    def g_of_r(self, nbins):
        """radial pair correlation function"""

        @jit(nopython=True, fastmath=True)
        def f(phi_1, phi_2, l_1, l_2):
            return l_1 != l_2

        return self._autoCorrelation(
            f, self.phi, self.color2, self.x, self.y, nbins, "all"
        )

    def Cvv(self, nbins):
        """Instantaneous velocity-velocity correlation function"""

        @jit(nopython=True, fastmath=True)
        def f(phi_1, phi_2, l_1, l_2):
            return np.cos(phi_1) * np.cos(phi_2) + np.sin(phi_1) * np.sin(phi_2)

        return self._autoCorrelation(
            f, self.phi, self.color2, self.x, self.y, nbins, "bin"
        )

    def Cnn(self, nbins):
        """Instantaneous director-director correlation function"""

        @jit(nopython=True, fastmath=True)
        def f(phi_1, phi_2, l_1, l_2):
            return np.abs(np.cos(phi_1) * np.cos(phi_2) + np.sin(phi_1) * np.sin(phi_2))

        return self._autoCorrelation(
            f, self.phi, self.color2, self.x, self.y, nbins, "bin"
        )

    @staticmethod
    @jit(nopython=True, fastmath=True)
    def _iteration(func, x, y, binsize, bin_sums, bin_counts, angles, labels):
        for i in range(len(x)):
            for j in range(i):
                dist = np.hypot(x[i] - x[j], y[i] - y[j])
                bin_num = min(int(dist / binsize), len(bin_sums) - 1)
                bin_sums[bin_num] += 2 * func(
                    angles[i], angles[j], labels[i], labels[j]
                )
                bin_counts[bin_num] += 2

    def _autoCorrelation(self, func, angles, labels, x, y, nbins, norm="bin"):
        r_max = np.hypot(x, y).max()
        binsize = np.sqrt(2) * r_max / nbins

        bin_ranges = np.arange(0, r_max, binsize)
        bin_sums = np.zeros(bin_ranges.shape[0])
        bin_counts = np.zeros_like(bin_sums, dtype=int)

        self._iteration(func, x, y, binsize, bin_sums, bin_counts, angles, labels)

        if norm == "bin":
            # for two-point correlations, averaging over values in each bin
            denominator = np.maximum(bin_counts, 1)
        else:
            # 2 \pi r dr \rho
            number_density = np.sum(bin_counts) / (2 * np.pi * r_max**2)
            denominator = (
                number_density * 2 * np.pi * binsize * (bin_ranges + binsize / 2)
            )
        bin_values = bin_sums / denominator
        answer = np.stack((bin_ranges, bin_values), axis=1)
        return answer[np.where(answer[:, 0] < r_max)]

    def nematic_local_3D(self):
        cutoff = self.local_radius
        if hasattr(self, "bx") and hasattr(self, "by") and hasattr(self, "bz"):
            return _nematic_local_3D(self.x, self.y, self.bx, self.by, self.bz, cutoff)
        else:
            return _nematic_local_3D(self.x, self.y, self.ex, self.ey, self.ez, cutoff)


# .three dimensional stuff
@jit(nopython=True, fastmath=True)
def _nematic_local_3D(x, y, ex, ey, ez, cutoff):
    ptclCounter = np.zeros(x.size)
    S = np.ones(x.size)
    Q = np.zeros((3, 3, x.size), dtype=float)

    for i in range(len(x)):
        for j in range(i, len(x)):
            if np.hypot(x[i] - x[j], y[i] - y[j]) <= cutoff:
                Q[0, 0, i] += 3 * ex[j] * ex[j] - 1
                Q[0, 1, i] += ex[j] * ey[j]
                Q[0, 2, i] += ex[j] * ez[j]

                Q[1, 0, i] += ey[j] * ex[j]
                Q[1, 1, i] += 3 * ey[j] * ey[j] - 1
                Q[1, 2, i] += ey[j] * ez[j]

                Q[2, 0, i] += ez[j] * ex[j]
                Q[2, 1, i] += ez[j] * ey[j]
                Q[2, 2, i] += 3 * ez[j] * ez[j] - 1
                ptclCounter[i] += 1

                if i == j:
                    continue

                Q[0, 0, j] += 3 * ex[i] * ex[i] - 1
                Q[0, 1, j] += ex[i] * ey[i]
                Q[0, 2, j] += ex[i] * ez[i]

                Q[1, 0, j] += ey[i] * ex[i]
                Q[1, 1, j] += 3 * ey[i] * ey[i] - 1
                Q[1, 2, j] += ey[i] * ez[i]

                Q[2, 0, j] += ez[i] * ex[i]
                Q[2, 1, j] += ez[i] * ey[i]
                Q[2, 2, j] += 3 * ez[i] * ez[i] - 1
                ptclCounter[j] += 1

    for i in range(len(x)):
        Q[:, :, i] /= ptclCounter[i]
        w, _ = np.eig(Q[:, :, i])
        S[i] = np.max(w)
    return S


@jit(nopython=True, fastmath=True)
def istouching(x1, y1, ex1, ey1, x2, y2, ex2, ey2, length1, length2, thickness):
    orientation_sign = 0

    l_dx = x1 - x2
    l_dy = y1 - y2
    l_dmin_2 = np.sqrt((l_dx) ** 2 + (l_dy) ** 2)
    thickness_pow2 = thickness * thickness
    thickness_ulim = thickness * thickness * 0.25

    while orientation_sign < 2:
        l_lambda1 = (length1 - thickness) * 0.5
        if orientation_sign == 1:
            l_lambda1 = -l_lambda1

        l_lambda2 = (l_dx - l_lambda1 * ex1) * ex2 + (l_dy - l_lambda1 * ey1) * ey2
        l_hx = l_dx - l_lambda2 * ex2 - l_lambda1 * ex1
        l_hy = l_dy - l_lambda2 * ey2 - l_lambda1 * ey1
        l_h12 = l_hx * l_hx + l_hy * l_hy
        l_lambda2_prime = (length2 - thickness) * 0.5
        if (
            (l_h12 < thickness_pow2)
            & (l_h12 > thickness_ulim)
            & (l_lambda2 * l_lambda2 < l_lambda2_prime * l_lambda2_prime)
        ):
            return -l_lambda1 * ex1 - l_hx * 0.5, -l_lambda1 * ey1 - l_hy * 0.5
        else:
            l_lambda2 = l_lambda2_prime
            l_hx = l_dx - l_lambda2 * ex2 - l_lambda1 * ex1
            l_hy = l_dy - l_lambda2 * ey2 - l_lambda1 * ey1
            l_h12 = l_hx * l_hx + l_hy * l_hy

            if (l_h12 < thickness_pow2) & (l_h12 > thickness_ulim):
                return -l_lambda1 * ex1 - l_hx * 0.5, -l_lambda1 * ey1 - l_hy * 0.5

            else:
                l_lambda2 = -l_lambda2_prime
                l_hx = l_dx - l_lambda2 * ex2 - l_lambda1 * ex1
                l_hy = l_dy - l_lambda2 * ey2 - l_lambda1 * ey1
                l_h12 = l_hx * l_hx + l_hy * l_hy
                if (l_h12 < thickness_pow2) & (l_h12 > thickness_ulim):
                    return -l_lambda1 * ex1 - l_hx * 0.5, -l_lambda1 * ey1 - l_hy * 0.5
        orientation_sign += 1
    return 0.0, 0.0
