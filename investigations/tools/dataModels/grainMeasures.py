from tools.dataAPI.datamodel import DataModel
from tools.dataModels.alignmentMeasures import istouching
import numpy as np
from numba import jit


class Grains(DataModel):

    def grains(self, grain_dist=1.0, grain_boundary_angle=np.pi / 8, return_angle=False):

        @jit(nopython=True, fastmath=True)
        def f(r, x, y, ex, ey, length, angle, grain_dist, grain_boundary_angle):
            thickness = r * 2 + r * 0.3

            grain_id = np.arange(0, x.shape[0]).astype(np.int64)
            grain_angle = angle

            touching_cells = np.full((x.shape[0], x.shape[0], 2), 0.0)

            for i in range(x.shape[0]):
                iw = grain_id[i]
                for j in range(x.shape[0]):
                    l_dist = np.hypot(x[i] - x[j], y[i] - y[j])
                    l_angle = abs(grain_angle[i] - angle[j])
                    hx, hy = istouching(
                        x[i],
                        y[i],
                        ex[i],
                        ey[i],
                        x[j],
                        y[j],
                        ex[j],
                        ey[j],
                        length[i],
                        length[j],
                        thickness,
                    )
                    if abs(hx > 0) | abs(hy > 0):
                        touching_cells[i, j, 0] = hx
                        touching_cells[i, j, 1] = hy
                        if l_angle < grain_boundary_angle:
                            grain_id[np.where(grain_id == iw)] = grain_id[j]
                            iw = grain_id[j]
                            grain_angle[np.where(grain_id == iw)] = np.mean(
                                angle[np.where(grain_id == iw)]
                            )

            x_gbp = np.zeros(0, dtype=np.float64)
            y_gbp = np.zeros(0, dtype=np.float64)
            phi_gbp = np.zeros(0, dtype=np.float64)
            grain_gbp_1 = np.zeros(0, dtype=np.float64)
            grain_gbp_2 = np.zeros(0, dtype=np.float64)

            ptr = 0
            for i in range(x.shape[0]):
                for j in range(x.shape[0]):
                    hx = touching_cells[i, j, 0]
                    hy = touching_cells[i, j, 1]

                    if abs(hx > 0) | abs(hy > 0):
                        l_angle = abs(grain_angle[i] - grain_angle[j])
                        if l_angle > grain_boundary_angle:
                            x_gbp = np.append(x_gbp, x[i] + hx)
                            y_gbp = np.append(y_gbp, y[i] + hy)
                            phi_gbp = np.append(phi_gbp, l_angle)
                            grain_gbp_1 = np.append(grain_gbp_1, grain_id[i])
                            grain_gbp_2 = np.append(grain_gbp_2, grain_id[j])

            clust_id = np.zeros(x_gbp.shape[0], dtype=np.int64)
            for i in range(x_gbp.shape[0]):
                clust_id[i] = i
            #clust_id = np.arange(0, x_gbp.shape[0], dtype=np.int64)

            for i in range(x_gbp.shape[0]):
                iw = clust_id[i]
                for j in range(x_gbp.shape[0]):
                    l_dist = np.hypot(x_gbp[i] - x_gbp[j], y_gbp[i] - y_gbp[j])
                    if l_dist < grain_dist:
                        if (
                            (grain_gbp_1[i] == grain_gbp_1[j])
                            & (grain_gbp_2[i] == grain_gbp_2[j])
                        ) | (
                            (grain_gbp_1[i] == grain_gbp_2[j])
                            & (grain_gbp_2[i] == grain_gbp_1[j])
                        ):
                            clust_id[np.where(clust_id == iw)] = clust_id[j]
                            iw = clust_id[j]
            return grain_angle, grain_id, x_gbp, y_gbp, clust_id

        # set grain_dist by the maxixum length of the cells in the system
        if grain_dist == 1.0:
            grain_dist = np.max(self.l)

        angle = np.arctan2(self.ey, self.ex)
        grain_angle, grain_id, x_gbp, y_gbp, clust_id =  f(
            self.r,
            self.x,
            self.y,
            self.ex,
            self.ey,
            self.l,
            angle,
            grain_dist,
            grain_boundary_angle,
        )
        if return_angle:
            return grain_angle, grain_id, x_gbp, y_gbp, clust_id
        return grain_id, x_gbp, y_gbp, clust_id
