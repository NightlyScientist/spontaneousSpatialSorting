import numpy as np
import math
from numpy import linalg


def classAPI(defects):
    nem_defects = []

    for defect in defects:
        # .Remove defect if in the experimental channel
        # .coordinate is defect[0], theta (radians) is defect[3], sign is defect[1]
        nem_defects.append(
            Defect(
                charge=np.sign(defect[1]) * 0.5,
                orientation=defect[3],
                position=defect[0],
            )
        )
    return nem_defects


def directorField(Q11, Q12):
    nx = np.zeros(Q11.size)
    ny = np.zeros(Q11.size)
    S = np.zeros(Q11.size)

    for i in range(Q11.size):
        v, w = linalg.eig(np.array([[Q11[i], Q12[i]], [Q12[i], -Q11[i]]]))
        S[i] = np.hypot(Q11[i], Q12[i])
        nx[i] = w[np.argmax(v)][0]
        ny[i] = w[np.argmax(v)][1]
    return nx, ny, S


# doc convert 1-D array to 4-D array
def QTensorSDirector(Q11, Q12, nx, ny, grid_shape, _s):
    lx, ly, lz = grid_shape
    S = np.zeros(shape=(lx, ly, lz), dtype=float)
    Q = np.zeros(shape=(3, 3, lx, ly, lz), dtype=float)
    director = np.zeros(shape=(3, lx, ly, lz), dtype=float)

    # .convert 1-D array to 4-D array
    for i in range(lx):
        for j in range(ly):
            _index = i + j * lx
            S[i, j, 0] = _s[_index]
            director[0, i, j, 0] = nx[_index]
            director[1, i, j, 0] = ny[_index]

            #Q[0, 0, i, j, 0] = Q11[_index]
            #Q[1, 1, i, j, 0] = -Q11[_index]
            #Q[0, 1, i, j, 0] = Q12[_index]
            #Q[1, 1, i, j, 0] = Q12[_index]

    # .calculate S-order and Q-tensor
    z = 0
    for x in range(lx):
        for y in range(ly):
            for i in range(3): 
                for j in range(3):
                    Q[i, j, x, y, z] = 3.0 * director[i, x, y, z] * director[j, x, y, z]

                    # .Diagonal elements
                    if i == j:
                        Q[i, j, x, y, z] -= 1.0
                    Q[i, j, x, y, z] *= 0.5 * S[x, y, z]
    return Q, S, director


def fieldMean(field, scalar, xyz, avDIM, minmax):
    # Set up dimensions
    if avDIM == "x":
        d1 = 1
        d2 = 2
        d3 = 0
    elif avDIM == "y":
        d1 = 0
        d2 = 2
        d3 = 1
    elif avDIM == "z":
        d1 = 0
        d2 = 1
        d3 = 2

    # Initialise XY array
    MEAN = np.zeros(shape=(3, xyz[d1], xyz[d2]), dtype=float)
    MAG = np.zeros(shape=(xyz[d1], xyz[d2]), dtype=float)
    if len(scalar) != 1:
        AVS = np.zeros(shape=(xyz[d1], xyz[d2]), dtype=float)

    # sum
    for a in range(xyz[d1]):
        for b in range(xyz[d2]):
            for c in range(xyz[d3]):
                if avDIM == "x":
                    for i in range(3):
                        MEAN[i][a][b] += field[i][c][a][b]
                    if len(scalar) != 1:
                        AVS[a][b] += scalar[c][a][b]
                if avDIM == "y":
                    for i in range(3):
                        MEAN[i][a][b] += field[i][a][c][b]
                    if len(scalar) != 1:
                        AVS[a][b] += scalar[a][c][b]
                if avDIM == "z":
                    for i in range(3):
                        MEAN[i][a][b] += field[i][a][b][c]
                    if len(scalar) != 1:
                        AVS[a][b] += scalar[a][b][c]

    # average
    for a in range(xyz[d1]):
        for b in range(xyz[d2]):
            for i in range(3):
                MEAN[i][a][b] / xyz[d3]
                if len(scalar) != 1:
                    AVS[a][b] / xyz[d3]

    # update min and max field magnitude?
    for a in range(xyz[d1]):
        for b in range(xyz[d2]):
            MAG[a][b] = math.sqrt(
                MEAN[0][a][b] ** 2 + MEAN[1][a][b] ** 2 + MEAN[2][a][b] ** 2
            )
            if MAG[a][b] > minmax[1]:
                minmax[1] = MAG[a][b]  # max value is minmax[1]
            elif MAG[a][b] < minmax[0]:
                minmax[0] = MAG[a][b]  # min value is minmax[0]
    if len(scalar) != 1:
        return MEAN, MAG, AVS
    else:
        return MEAN, MAG


def formMesh(xyz):
    XYZ = np.zeros(shape=(3, xyz[0], xyz[1], xyz[2]))

    for x in range(xyz[0]):
        for y in range(xyz[1]):
            for z in range(xyz[2]):
                XYZ[0][x][y][z] = float(x) + 0.5
                XYZ[1][x][y][z] = float(y) + 0.5
                XYZ[2][x][y][z] = float(z) + 0.5
    return XYZ


def makeMesh(XYZ, xyz, avDIM):
    # convert to 2D

    # Set up dimensions
    if avDIM == "x":
        d1 = 1
        d2 = 2
    elif avDIM == "y":
        d1 = 0
        d2 = 2
    elif avDIM == "z":
        d1 = 0
        d2 = 1

    # Initialise XY array
    XY = np.zeros(shape=(2, xyz[d1], xyz[d2]), dtype=float)

    # Convert XYZ to a 2D mesh
    if avDIM == "x":
        for y in range(xyz[1]):
            for z in range(xyz[2]):
                XY[0][y][z] = XYZ[d1][0][y][z]
                XY[1][y][z] = XYZ[d2][0][y][z]
    elif avDIM == "y":
        for x in range(xyz[0]):
            for z in range(xyz[2]):
                XY[0][x][z] = XYZ[d1][x][0][z]
                XY[1][x][z] = XYZ[d2][x][0][z]
    elif avDIM == "z":
        for x in range(xyz[0]):
            for y in range(xyz[1]):
                XY[0][x][y] = XYZ[d1][x][y][0]
                XY[1][x][y] = XYZ[d2][x][y][0]
    return XY


def calcCM(posMono, numMono, dim):
    # convert monomer positions to polymer centre of mass
    cmRod = np.zeros(dim, dtype=float)
    for d in range(dim):
        for k in range(numMono):
            cmRod[d] += posMono[d][k]
        cmRod[d] /= numMono
    return cmRod.tolist()


def shiftCMPeriodic(cm, dim, xyz):
    # if centre of mass position is outside box, shift it
    for d in range(dim):
        if cm[d] > xyz[d]:
            cm[d] -= xyz[d]
        if cm[d] < 0.0:
            cm[d] += xyz[d]
    return 0


def identifyCell(position):
    # not specifically polymers
    cell = np.zeros(shape=(3), dtype=int)
    for d in range(3):
        cell[d] = math.floor(position[d])
    return cell.tolist()


def shiftMonoPeriodic(posMono, numMono, xyz):
    # shift monomer positions if outside box
    for d in range(2):
        periodic = False
        for k in range(numMono - 1):
            if posMono[d][k + 1] - posMono[d][k] > 0.5 * xyz[d]:
                periodic = True
            if posMono[d][k + 1] - posMono[d][k] < -0.5 * xyz[d]:
                periodic = True
        if periodic == True:
            for k in range(numMono):
                if posMono[d][k] <= 2 * numMono:
                    posMono[d][k] += xyz[d]
    return 0


def distance2D(pos1, pos2, xyz, boundarylist):
    sumsqrs = 0

    for i in range(0, 2):
        separation = pos1[i] - pos2[i]

        if boundarylist[i] == "p":
            if separation > 0.5 * xyz[i]:
                separation -= xyz[i]
            if separation < -0.5 * xyz[i]:
                separation += xyz[i]

        sumsqrs += separation * separation

    sumsqrs = math.sqrt(sumsqrs)

    return sumsqrs


# doc Calculating the D22 element of the nematic disclination tensor
def calcD_2D(xyz, Q):
    D22 = np.zeros(shape=(xyz[0], xyz[1]))
    z = 0

    for x in range(xyz[0]):
        for y in range(xyz[1]):
            # . Neighbouring cells
            U = y + 1  # up
            D = y - 1  # down
            R = x + 1  # right
            L = x - 1  # left

            # . Applying PBC
            if R >= xyz[0]:
                R -= xyz[0]
            if L < 0:
                L += xyz[0]

            # . x-derivatives
            dxQxx = 0.5 * (Q[0][0][R][y][z] - Q[0][0][L][y][z])
            dxQyy = 0.5 * (Q[1][1][R][y][z] - Q[1][1][L][y][z])
            dxQxy = 0.5 * (Q[0][1][R][y][z] - Q[0][1][L][y][z])
            dxQyx = dxQxy

            # . y- derivatives
            if U >= xyz[1]:
                U -= xyz[1]
            if D < 0:
                D += xyz[1]

            dyQxx = 0.5 * (Q[0][0][x][U][z] - Q[0][0][x][D][z])
            dyQyy = 0.5 * (Q[1][1][x][U][z] - Q[1][1][x][D][z])
            dyQxy = 0.5 * (Q[0][1][x][U][z] - Q[0][1][x][D][z])
            dyQyx = dyQxy

            D22[x][y] = (
                dxQxx * dyQyx
                + dxQxy * dyQyy
                - dxQyx * dyQxx
                - dxQyy * dyQxy
                + dyQyx * dxQxx
                + dyQyy * dxQxy
                - dyQxx * dxQyx
                - dyQxy * dxQyy
            )
    return D22


def calcSSB(xyz, Q):
    # . Initialise the scalar-splay bend parameter
    ssb = np.zeros(shape=(xyz[0], xyz[1]), dtype=float)

    # . Initialise first derivative array
    djQij = np.zeros((2, xyz[0], xyz[1]), dtype=float)

    # . Loop over field
    z = 0
    for x in range(xyz[0]):
        for y in range(xyz[1]):
            xylist = [x, y]

            # . Directionality of the derivative
            xL = [x, x, x]
            yL = [y, y, y]

            # . Loop over indices
            for i in range(2):
                for j in range(2):
                    if j == 0:
                        xL = [x - 1, x, x + 1]
                    if j == 1:
                        yL = [y - 1, y, y + 1]

                    # . Calculate derivative wrt j
                    if xylist[j] == 0:
                        djQij[i][x][y] += (
                            Q[i][j][xL[2]][yL[2]][z] - Q[i][j][xL[1]][yL[1]][z]
                        )  # forward
                    elif xylist[j] == xyz[j] - 1:
                        djQij[i][x][y] += (
                            Q[i][j][xL[1]][yL[1]][z] - Q[i][j][xL[0]][yL[0]][z]
                        )  # backward
                    else:
                        djQij[i][x][y] += 0.5 * (
                            Q[i][j][xL[2]][yL[2]][z] - Q[i][j][xL[0]][yL[0]][z]
                        )  # central

                    # Reset
                    if j == 0:
                        xL = [x, x, x]
                    if j == 1:
                        yL = [y, y, y]

    # . Loop over field
    for x in range(xyz[0]):
        for y in range(xyz[1]):
            xylist = [x, y]

            # Directionality of the derivative
            xL = [x, x, x]
            yL = [y, y, y]

            # Initialise second derivative array
            didjQij = 0.0

            # Loop over indices
            for i in range(2):
                if i == 0:
                    xL = [x - 1, x, x + 1]
                if i == 1:
                    yL = [y - 1, y, y + 1]

                # Calculate derivative wrt i
                if xylist[i] == 0:
                    didjQij += (
                        djQij[i][xL[2]][yL[2]] - djQij[i][xL[1]][yL[1]]
                    )  # forward
                elif xylist[i] == xyz[i] - 1:
                    didjQij += (
                        djQij[i][xL[1]][yL[1]] - djQij[i][xL[0]][yL[0]]
                    )  # backward
                else:
                    didjQij += 0.5 * (
                        djQij[i][xL[2]][yL[2]] - djQij[i][xL[0]][yL[0]]
                    )  # central

                # Reset
                if i == 0:
                    xL = [x, x, x]
                if i == 1:
                    yL = [y, y, y]

            ssb[x][y] = didjQij
    return ssb


def extractSingularities(xyz, scalarfield, cutoffsd):
    # . Extract and group all singularities from just the scalarfield
    def findSingularities(xyz, scalarfield, cutoffsd):
        # . Find singularities of a field by cutting out values within 2 s.d. of the mode

        binnum = 60  # number of bins - for resolution of mode

        cell = []  # list of singularity cells
        value = []  # list of values

        # Make histogram
        hist, binval = np.histogram(scalarfield, binnum, density=True)
        binwidth = binval[1] - binval[0]

        # Make centre values -- seems correct
        centre = np.zeros(binnum)
        for l in range(binnum):
            centre[l] = 0.5 * (binval[l + 1] - binval[l]) + binval[l]

        # Find peak
        peak = np.max(hist)
        listhist = hist.tolist()
        peakindex = listhist.index(peak)
        peakcentre = centre[peakindex]

        # Standard deviation
        sd = np.std(scalarfield)

        # Loop through scalar field and cut
        for x in range(xyz[0]):
            for y in range(xyz[1]):
                if scalarfield[x][y] > (peakcentre + cutoffsd * sd) or scalarfield[x][
                    y
                ] < (peakcentre - cutoffsd * sd):
                    cell.append([x, y])
                    value.append(scalarfield[x][y])

        # Tuple the cells and values together
        singlist = list(zip(cell, value))
        return singlist

    def neighbourcheck(xyz, celli, cellj, boundarylist=["p", "p", "p"]):
        # Flag for a neighbouring cell
        check = False

        # Define directions
        lxr = [celli[0] - 2, celli[0] - 1, celli[0], celli[0] + 1, celli[0] + 2]
        dyu = [celli[1] - 2, celli[1] - 1, celli[1], celli[1] + 1, celli[1] + 2]

        # . Apply PBC shifts
        if (lxr[0] < 0) and (boundarylist[0] == "p"):
            lxr[0] += xyz[0]
        if (lxr[1] < 0) and (boundarylist[0] == "p"):
            lxr[1] += xyz[0]
        if (lxr[3] >= xyz[0]) and (boundarylist[0] == "p"):
            lxr[3] -= xyz[0]
        if (lxr[4] >= xyz[0]) and (boundarylist[0] == "p"):
            lxr[4] -= xyz[0]

        if (dyu[0] < 0) and (boundarylist[1] == "p"):
            dyu[0] += xyz[1]
        if (dyu[1] < 0) and (boundarylist[1] == "p"):
            dyu[1] += xyz[1]
        if (dyu[3] >= xyz[1]) and (boundarylist[1] == "p"):
            dyu[3] -= xyz[1]
        if (dyu[4] >= xyz[1]) and (boundarylist[1] == "p"):
            dyu[4] -= xyz[1]

        # Minimum and maximum for neighbour loop
        xminmax = [0, 4]
        yminmax = [0, 4]

        # Find if neighbours are in the walls or past the walls
        if lxr[4] >= xyz[0]:
            xminmax[1] = 3
        if lxr[3] >= xyz[0]:
            xminmax[1] = 2
        if lxr[0] < 0:
            xminmax[0] = 1
        if lxr[1] < 0:
            xminmax[0] = 2

        if dyu[4] >= xyz[1]:
            yminmax[1] = 3
        if dyu[3] >= xyz[1]:
            yminmax[1] = 2
        if dyu[0] < 0:
            yminmax[0] = 1
        if dyu[1] < 0:
            yminmax[0] = 2

        positionList = []
        # Does j share a position with i's neighbours
        for i in range(xminmax[0], xminmax[1] + 1):
            for j in range(yminmax[0], yminmax[1] + 1):
                positionList.append([lxr[i], dyu[j]])
                if (cellj[0] == lxr[i]) and (cellj[1] == dyu[j]):
                    check = True

        return check

    def groupSingularities(xyz, singList):
        # Number of singularity cells,
        N = len(singList)

        # Assign group index (-1 to initialise)
        for n in range(N):
            singList[n] = list(singList[n] + (-1,))

        # Index value to assign to groups
        newindex = 0

        for i in range(N):
            for j in range(i, N):
                if i != j:
                    if np.sign(singList[i][1]) == np.sign(singList[j][1]):

                        # Check if neighbours
                        ci = [singList[i][0][0], singList[i][0][1]]
                        cj = [singList[j][0][0], singList[j][0][1]]
                        check = neighbourcheck(xyz, ci, cj)
                        if check == True:

                            ### Options ###

                            # 1) both indices -1? -- unsorted
                            if singList[i][2] == -1 and singList[j][2] == -1:

                                # Give unassigned index to form group
                                singList[i][2] = newindex
                                singList[j][2] = newindex
                                newindex += 1

                            # 2) one index -1?
                            elif singList[i][2] == -1 and singList[j][2] != -1:
                                # Give group index to unassigned cell
                                singList[i][2] = singList[j][2]
                            elif singList[j][2] == -1 and singList[i][2] != -1:
                                # Give group index to unassigned cell
                                singList[j][2] = singList[i][2]

                            # 3) neither index -1 and they don't have the same index
                            elif (
                                (singList[i][2] != -1)
                                and (singList[j][2] != -1)
                                and (singList[i][2] != singList[j][2])
                            ):

                                # Find all other cells with the same value as j and assign all i's index
                                for k in range(N):
                                    if singList[k][2] == singList[j][2]:
                                        singList[k][2] = singList[i][2]

        groupList = []
        # . find all the index values
        for ind in range(newindex + 1):
            groupCandidates = []

            # . Loop through cells and find which ones have that index value
            for j in range(N):
                if singList[j][2] == ind:
                    groupCandidates.append(singList[j])

            if len(groupCandidates) > 0:
                groupList.append(groupCandidates)

        return groupList

    def findmaxposition(group):
        xmax = 0
        ymax = 0
        for k in range(len(group)):

            if xmax < group[k][0][0]:
                xmax = group[k][0][0]
            if ymax < group[k][0][1]:
                ymax = group[k][0][1]

        return [xmax, ymax]

    def combineSingularityGroups(xyz, groupList):
        defectList = []

        for group in groupList:
            meanx = 0
            meany = 0

            # For whether to apply PBC
            maxpos = findmaxposition(group)

            for k in range(len(group)):
                # Sort x and y components
                meanx += group[k][0][0]
                meany += group[k][0][1]

                # Apply PBC shift
                if maxpos[0] == xyz[0] - 1:
                    if group[k][0][0] > xyz[0] * 0.7:
                        meanx -= xyz[0]
                if maxpos[1] == xyz[1] - 1:
                    if group[k][0][1] > xyz[1] * 0.7:
                        meany -= xyz[1]

            # Convert sum to mean
            meanx /= len(group)
            meany /= len(group)

            # Apply PBC back
            if meanx < 0:
                meanx += xyz[0]
            if meany < 0:
                meany += xyz[1]
            if meanx >= xyz[0]:
                meanx -= xyz[0]
            if meany >= xyz[1]:
                meany -= xyz[1]

            defectList.append([[meanx, meany], group[0][1], np.sqrt(len(group))])

        return defectList

    singlist = findSingularities(xyz, scalarfield, cutoffsd)
    grouplist = groupSingularities(xyz, singlist)
    defectlist = combineSingularityGroups(xyz, grouplist)
    return defectlist


def defectAxis(xyz, Q, sign, x, y, corediameter):
    # x and y need to be integers - round them
    k = 0.5 * sign  # charge
    multiplier = k / (1 - k)

    # Initialise numerator and denominator for angle calculation
    numerator = 0.0
    denominator = 0.0

    # Core radius
    coreradii = round(0.5 * corediameter)

    # Finding elements of array that are in the loop
    loopList = []
    spread = 1 + 2 * coreradii

    for j in range(y - coreradii + 1, y + coreradii):
        # x = L-2	y takes all values except L-2
        # X – L+2	y takes all values except L-2
        loopList.append([x - coreradii, j])
        loopList.append([x + coreradii, j])
    for i in range(x - coreradii, x + coreradii + 1):
        # y = L-2 	x takes all values
        # Y = L+2 	x takes all values
        loopList.append([i, y - coreradii])
        loopList.append([i, y + coreradii])

    # refactor: loopList seems to be ok, but there might be a problem with below
    # Loop over loop
    for loopItem in loopList:
        # Find neighbours to the loop
        r = loopItem[0] + 1
        l = loopItem[0] - 1
        u = loopItem[1] + 1
        d = loopItem[1] - 1

        # Apply PBC anyway
        if r < 0:
            r += xyz[0]
        if r >= xyz[0]:
            r -= xyz[0]
        if u >= xyz[1]:
            u -= xyz[1]
        if u < 0:
            u += xyz[1]
        if l < 0:
            l += xyz[0]
        if l >= xyz[0]:
            l -= xyz[0]
        if d < 0:
            d += xyz[1]
        if d >= xyz[1]:
            d -= xyz[1]
        if loopItem[0] < 0:
            loopItem[0] += xyz[0]
        if loopItem[0] >= xyz[0]:
            loopItem[0] -= xyz[0]
        if loopItem[1] < 0:
            loopItem[1] += xyz[1]
        if loopItem[1] >= xyz[1]:
            loopItem[1] -= xyz[1]

        dxQxy = 0.5 * (Q[0][1][r][loopItem[1]][0] - Q[0][1][l][loopItem[1]][0])
        dyQxx = 0.5 * (Q[0][0][loopItem[0]][u][0] - Q[0][0][loopItem[0]][d][0])
        dxQxx = 0.5 * (Q[0][0][r][loopItem[1]][0] - Q[0][0][l][loopItem[1]][0])
        dyQxy = 0.5 * (Q[0][1][loopItem[0]][u][0] - Q[0][1][loopItem[0]][d][0])

        numerator += sign * dxQxy - dyQxx
        denominator += dxQxx + sign * dyQxy

    # Calculate angle
    angle = multiplier * np.arctan2(numerator, denominator)
    return angle


# doc Angle in radians
def findNematicOrientation(xyz, defectList, Q):
    # Loop over defects
    for defect in defectList:
        theta = defectAxis(
            xyz,
            Q,
            np.sign(defect[1]),
            round(defect[0][0]),
            round(defect[0][1]),
            defect[2],
        )
        defect.append(theta)
    return defectList


# doc Calculating the Q-criterion
def calcQC(xyz, vel):
    QC = np.zeros(shape=(xyz[0], xyz[1]))
    z = 0

    # .Calculate derivatives
    for x in range(xyz[0]):
        for y in range(xyz[1]):
            if x == 0:
                dudx = vel[0][x + 1][y][z] - vel[0][x][y][z]
                dvdx = vel[1][x + 1][y][z] - vel[1][x][y][z]
            elif x == xyz[0] - 1:
                dudx = vel[0][x][y][z] - vel[0][x - 1][y][z]
                dvdx = vel[1][x][y][z] - vel[1][x - 1][y][z]
            else:
                dudx = 0.5 * (vel[0][x + 1][y][z] - vel[0][x - 1][y][z])
                dvdx = 0.5 * (vel[1][x + 1][y][z] - vel[1][x - 1][y][z])

            if y == 0:
                dudy = vel[0][x][y + 1][z] - vel[0][x][y][z]
                dvdy = vel[1][x][y + 1][z] - vel[1][x][y][z]
            elif y == xyz[1] - 1:
                dudy = vel[0][x][y][z] - vel[0][x][y - 1][z]
                dvdy = vel[1][x][y][z] - vel[1][x][y - 1][z]
            else:
                dudy = 0.5 * (vel[0][x][y + 1][z] - vel[0][x][y - 1][z])
                dvdy = 0.5 * (vel[1][x][y + 1][z] - vel[1][x][y - 1][z])

            QC[x][y] = (dvdx - dudy) ** 2 - 0.5 * (
                ((2 * dudx) * (2 * dudx))
                + ((2 * dvdy) * (2 * dvdy))
                + 2 * ((dudy + dvdx) * (dudy + dvdx))
            )
    return QC


# doc Calculate the velocity gradient tensor
def calcVGT(xyz, vel):
    VGT = np.zeros(shape=(2, 2, xyz[0], xyz[1]))

    # .Calculate derivatives
    for x in range(xyz[0]):
        for y in range(xyz[1]):
            z = 0
            if x == 0:
                dudx = vel[0][x + 1][y][z] - vel[0][x][y][z]
                dvdx = vel[1][x + 1][y][z] - vel[1][x][y][z]
            elif x == xyz[0] - 1:
                dudx = vel[0][x][y][z] - vel[0][x - 1][y][z]
                dvdx = vel[1][x][y][z] - vel[1][x - 1][y][z]
            else:
                dudx = 0.5 * (vel[0][x + 1][y][z] - vel[0][x - 1][y][z])
                dvdx = 0.5 * (vel[1][x + 1][y][z] - vel[1][x - 1][y][z])

            if y == 0:
                dudy = vel[0][x][y + 1][z] - vel[0][x][y][z]
                dvdy = vel[1][x][y + 1][z] - vel[1][x][y][z]
            elif y == xyz[1] - 1:
                dudy = vel[0][x][y][z] - vel[0][x][y - 1][z]
                dvdy = vel[1][x][y][z] - vel[1][x][y - 1][z]
            else:
                dudy = 0.5 * (vel[0][x][y + 1][z] - vel[0][x][y - 1][z])
                dvdy = 0.5 * (vel[1][x][y + 1][z] - vel[1][x][y - 1][z])

            VGT[0][0][x][y] = dudx
            VGT[0][1][x][y] = dudy
            VGT[1][0][x][y] = dvdx
            VGT[1][1][x][y] = dvdy
    return VGT


# doc Calculate second invariant
def calcInvar2(xyz, Tensor):
    inv2 = np.zeros(shape=(xyz[0], xyz[1]))

    for x in range(xyz[0]):
        for y in range(xyz[1]):
            inv2[x][y] = (
                Tensor[0][0][x][y] * Tensor[1][1][x][y]
                - Tensor[0][1][x][y] * Tensor[1][0][x][y]
            )
    return inv2


class Defect:
    def __init__(self, charge, orientation, position):
        self.charge = charge
        self.orientation = orientation
        self.position = position
