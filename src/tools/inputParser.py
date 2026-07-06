import numpy as np

def readBinary(file, numVariables=51):
    maxind = numVariables
    data = np.fromfile(file, dtype=np.double)
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

    splits = data[46 : np.size(data) : maxind]
    phi = data[47 : np.size(data) : maxind]
    u = data[48 : np.size(data) : maxind]
    v = data[49 : np.size(data) : maxind]
    b_idx = data[50 : np.size(data) : maxind]

    # get only those where active
    ind = np.where((active == 1))

    return {
        "x" : x[ind],
        "y" : y[ind],
        "z" : z[ind],
        "ex" : ex[ind],
        "ey" : ey[ind],
        "ez" : ez[ind],
        "length" : length[ind],
        "ancestor" : ancestor[ind],
        "color" : color[ind],
        "color2" : color2[ind],
        "vx" : vx[ind],
        "vy" : vy[ind],
        "vz" : vz[ind],
        "phi" : phi[ind],
        "u" : u[ind],
        "v" : v[ind],
        "box_idx" : b_idx[ind],
    }

def readOpts(file):
    s = { (i.strip()) for i in open(file,'r').readlines() }
    return {i.split(' ')[0]:i.split(' ')[1] for i in s}

# if __name__ == "__main__":
#     data = readBinary(file=f"{datapath}/t-{time}.pos")
#     opts = readOpts(file=f"{datapath}/input.dat")
