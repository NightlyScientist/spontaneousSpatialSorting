import numpy as np


def legacy(datapath, time):
    maxind = 47
    data = np.fromfile("%s/t-%d.pos" % (datapath, time), dtype=np.double)
    x = data[0 : np.size(data) : maxind]
    y = data[1 : np.size(data) : maxind]
    z = data[2 : np.size(data) : maxind]
    ex = data[3 : np.size(data) : maxind]
    ey = data[4 : np.size(data) : maxind]
    ez = data[5 : np.size(data) : maxind]
    length = data[6 : np.size(data) : maxind]
    active = data[7 : np.size(data) : maxind].astype(int)
    ancestor = data[8 : np.size(data) : maxind].astype(int)
    color = data[9 : np.size(data) : maxind].astype(int)
    color2 = data[10 : np.size(data) : maxind].astype(int)

    vx = data[11 : np.size(data) : maxind]
    vy = data[12 : np.size(data) : maxind]
    vz = data[13 : np.size(data) : maxind]

    ind = np.where((active == 1))
    x = x[ind]
    y = y[ind]
    ex = ex[ind]
    ey = ey[ind]
    length = length[ind]
    ancestor = ancestor[ind]