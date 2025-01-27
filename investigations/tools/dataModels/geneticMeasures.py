from tools.dataAPI.datamodel import DataModel
from tools.dataModels.alignmentMeasures import istouching
import numpy as np
from numba import jit


class Genetics(DataModel):

    def Heterozygosities(self, touching=False):
        H_0 = self.Heterozygosity_local(touching=touching)
        H_1 = H_0[self.color2 == 0]
        H_2 = H_0[self.color2 == 1]
        return H_1, H_2

    def Heterozygosity_global(self, touching=False):
        H_1, H_2 = self.Heterozygosities(touching=touching)
        return 0.5 * (np.mean(H_1) + np.mean(H_2))

    def Heterozygosity_local(self, touching=False):
        cutoff = self.local_radius**2
        return self._Heterozygosity(
            self.x,
            self.y,
            self.ex,
            self.ey,
            self.l,
            self.thickness,
            self.color2,
            cutoff,
            touching=touching
        )

    @staticmethod
    @jit(nopython=True, fastmath=True)
    def _Heterozygosity(x, y, ex, ey, l, d0, color2, cutoff, touching=False):
        values = np.zeros(x.size)
        ptclCounter = np.zeros(x.size)
        if touching:
            _cutoff = np.max(l) ** 2
        else:
            _cutoff = cutoff

        for i in range(len(x)):
            for j in range(i + 1, len(x)):
                dist = (x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2

                if dist > _cutoff:
                    continue

                hx = 1
                hy = 1
                if touching:
                    hx, hy = istouching(
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

                if abs(hx) > 0 or abs(hy) > 0:
                    ptclCounter[i] += 1
                    ptclCounter[j] += 1
                    if color2[i] != color2[j]:
                        values[i] += 1
                        values[j] += 1
        return values / (ptclCounter + (ptclCounter == 0))

    def Ancestor_Distance(self, mindist=0.5):
        nrecord = len(self.ancestors[0].split(":").astype(int))
        the_index_list = np.zeros((nrecord, len(self.x)))

        for i in range(len(self.x)):
            the_index_list[:, i] = self.ancestors[i].split(":").astype(int)

        ancestor = self.ancestor

        ij_ansdist = np.zeros((ancestor.shape[0], ancestor.shape[0]))

        ans_dist = np.zeros(ancestor.shape[0])
        n_ans_avr = np.zeros(ancestor.shape[0])

        for i in range(ancestor.shape[0]):
            i_ind_list = the_index_list[:, i]
            i_ind_list = i_ind_list[np.where(i_ind_list > 0)]
            i_ind_list = np.flipud(i_ind_list)

            for j in range(ancestor.shape[0]):
                if i != j:
                    dist = np.hypot(self.x[i] - self.x[j], self.y[i] - self.y[j])

                    equaltoken = False

                    i_counter = 0
                    j_counter = 0

                    j_ind_list = the_index_list[:, j]
                    j_ind_list = j_ind_list[np.where(j_ind_list > 0)]
                    j_ind_list = np.flipud(j_ind_list)

                    if (np.size(i_ind_list) > 1) and (np.size(j_ind_list) > 1):
                        i_ans_ind = i_ind_list[0]
                        j_ans_ind = j_ind_list[0]

                        while equaltoken == False:
                            while (j_counter < np.size(j_ind_list) - 1) & (
                                equaltoken == False
                            ):
                                if i_ans_ind == j_ans_ind:
                                    equaltoken = True
                                else:
                                    j_counter += 1
                                    j_ans_ind = j_ind_list[j_counter]

                            if i_ans_ind == j_ans_ind:
                                equaltoken = True
                            else:
                                j_counter = 0
                                i_counter += 1
                                if i_counter < np.size(i_ind_list):
                                    j_ans_ind = j_ind_list[j_counter]
                                    i_ans_ind = i_ind_list[i_counter]
                                else:
                                    equaltoken = True
                                    i_counter = np.size(
                                        i_ind_list[np.where(i_ind_list > 0)]
                                    )
                                    j_counter = i_counter
                    ij_ansdist[i, j] = min([i_counter, j_counter])

                    if dist < mindist:
                        ans_dist[i] += min([i_counter, j_counter])
                        n_ans_avr[i] += 1

        ans_dist[np.where(n_ans_avr > 0)] = (
            ans_dist[np.where(n_ans_avr > 0)] / n_ans_avr[np.where(n_ans_avr > 0)]
        )
        return ans_dist
