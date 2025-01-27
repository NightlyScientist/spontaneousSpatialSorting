def get_defect(time, datapath):
    for i in range(0, gridsize):
        S[i] = np.sqrt(Q11[i] * Q11[i] + Q12[i] * Q12[i])
        phi[i] = 0.5 * np.arctan2(Q12[i], Q11[i])
        if phi[i] > np.pi:
            phi[i] -= np.pi
        if phi[i] < 0:
            phi[i] += np.pi
        nx[i] = np.cos(phi[i])
        ny[i] = np.sin(phi[i])
        phi[i] = np.arctan2(ny[i], nx[i])

    curln_zcomp = np.zeros(gridsize)
    q = np.zeros(gridsize)
    qgrid = np.zeros((grid, grid))
    delx = maxx / grid
    delta = 1.0 / 2.0 * delx
    polx = np.zeros(gridsize)
    poly = np.zeros(gridsize)

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
            polx[ind] = -l_polx / l_polabs
            poly[ind] = -l_poly / l_polabs

    ind = np.where(S > 0.01)

    S = S[ind]
    rho = rho[ind]
    nx = nx[ind]
    ny = ny[ind]
    curln_zcomp = curln_zcomp[ind]

    coordinates = peak_local_max(abs(qgrid))

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
    for k in range(0, np.size(single_coord)):
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
    outfile = "%s/neg_defect-%d.pos" % (datapath, time)
    np.save(
        outfile, (xq[negcharge], yq[negcharge], polxmax[negcharge], polymax[negcharge])
    )
    outfile = "%s/pos_defect-%d.pos" % (datapath, time)
    np.save(
        outfile, (xq[poscharge], yq[poscharge], polxmax[poscharge], polymax[poscharge])
    )