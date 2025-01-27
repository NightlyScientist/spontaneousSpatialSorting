import numpy as np
from tools.dataAPI.datamodel import DataModel
from numba import njit, jit


class FlowFields(DataModel):

    def velocityProjection(self):
        mag = np.array([x if x > 0 else 1 for x in np.hypot(self.fx, self.fy)])
        projections = np.abs((self.ex * self.fx + self.ey * self.fy) / mag)
        return projections

    def nematicField(self, n):
        self.grid_size = n
        self.sigma = 2 * self.thickness

        @jit(nopython=True, fastmath=True)
        def f(n, x, y, ex, ey, l, fx, fy, thickness, sigma):
            nf_x = np.zeros(n * n, dtype=float)
            nf_y = np.zeros(n * n, dtype=float)
            rho = np.zeros(n * n, dtype=float)
            Q11 = np.zeros(n * n, dtype=float)
            Q12 = np.zeros(n * n, dtype=float)
            vx = np.zeros(n * n, dtype=float)
            vy = np.zeros(n * n, dtype=float)
            press = np.zeros(n * n, dtype=float)

            n_particle = x.shape[0]

            angle = np.arctan2(ey, ex)
            ex_p = -np.sin(angle)
            ey_p = np.cos(angle)

            r_max = max(max(np.abs(x)), max(np.abs(y))) + 1
            pos = np.linspace(-r_max, r_max, n)
            pos = pos + 0.5*(pos[1] - pos[0])

            for i in range(n):
                for j in range(n):
                    ind = i + j * n
                    nf_x[ind] = pos[i]
                    nf_y[ind] = pos[j]

                    for k in range(n_particle):
                        dx = nf_x[ind] - x[k]
                        dy = nf_y[ind] - y[k]

                        proj = ex[k] * dx + ey[k] * dy
                        proj_p = ex_p[k] * dx + ey_p[k] * dy
                        lhalf = 0.5 * l[k]
                        dhalf = 0.5 * thickness

                        a = np.tanh((proj + lhalf) / sigma)
                        b = np.tanh((proj - lhalf) / sigma)
                        c = np.tanh((proj_p + dhalf) / sigma)
                        d = np.tanh((proj_p - dhalf) / sigma)
                        h_r = 0.25 * (a - b) * (c - d)

                        rho[ind] += h_r
                        Q11[ind] += h_r * (2.0 * ex[k] * ex[k] - 1.0)
                        Q12[ind] += h_r * (2.0 * ex[k] * ey[k])
                        vx[ind] += h_r * fx[k]
                        vy[ind] += h_r * fy[k]
                        press[ind] += (
                            h_r * np.hypot(fx[k], fy[k]) / (2.0 * l[k] + sigma)
                        )
            return {
                "n_x": nf_x,
                "n_y": nf_y,
                "rho": rho,
                "Q11": Q11,
                "Q12": Q12,
                "n_vx": vx,
                "n_vy": vy,
                "press": press,
            }

        result = f(
            n,
            self.x,
            self.y,
            self.ex,
            self.ey,
            self.l,
            self.fx,
            self.fy,
            self.thickness,
            self.sigma,
        )
        for key in result:
            setattr(self, key, result[key])
