import numpy as np
from numba import jit


@staticmethod
@jit(nopython=True, fastmath=True)
def edgeFraction(labels, x, y, bandwidth=1, bandwidthFraction=None):
    center = np.array([x.mean(), y.mean()])
    distance = np.hypot(x - center[0], y - center[1])
    r_max = distance.max()
    if bandwidthFraction is not None:
        r_lower = r_max * max([1.0 - bandwidthFraction, 0])
    else:
        r_lower = max([0, r_max - bandwidth])

    band = labels[np.where((r_lower <= distance) & (distance<= r_max))]
    return np.sum(band == 1) / len(band)