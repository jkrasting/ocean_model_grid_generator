#!/usr/bin/env python

from __future__ import print_function

import numpypi.numpypi_series as np

from oceangrid.util import mdist, lagrange_interp
from oceangrid.constants import _default_Re, PI_180, HUGE

def bipolar_projection(lamg, phig, lon_bp, rp, metrics_only=False):
    """Makes a stereographic bipolar projection of the input coordinate mesh (lamg,phig)
    Returns the projected coordinate mesh and their metric coefficients (h^-1).
    The input mesh must be a regular spherical grid capping the pole with:
        latitudes between 2*arctan(rp) and 90  degrees
        longitude between lon_bp       and lonp+360
    """
    ### symmetry meridian resolution fix
    phig = 90 - 2 * np.arctan(np.tan(0.5 * (90 - phig) * PI_180) / rp) / PI_180
    tmp = mdist(lamg, lon_bp) * PI_180
    sinla = np.sin(tmp)  # This makes phis symmetric
    sphig = np.sin(phig * PI_180)
    alpha2 = (np.cos(tmp)) ** 2  # This makes dy symmetric
    beta2_inv = (np.tan(phig * PI_180)) ** 2
    rden = 1.0 / (1.0 + alpha2 * beta2_inv)

    if not metrics_only:
        B = sinla * np.sqrt(rden)  # Actually two equations  +- |B|
        # Deal with beta=0
        B = np.where(np.abs(beta2_inv) > HUGE, 0.0, B)
        lamc = np.arcsin(B) / PI_180
        ##But this equation accepts 4 solutions for a given B, {l, 180-l, l+180, 360-l }
        ##We have to pickup the "correct" root.
        ##One way is simply to demand lamc to be continuous with lam on the equator phi=0
        ##I am sure there is a more mathematically concrete way to do this.
        lamc = np.where((lamg - lon_bp > 90) & (lamg - lon_bp <= 180), 180 - lamc, lamc)
        lamc = np.where(
            (lamg - lon_bp > 180) & (lamg - lon_bp <= 270), 180 + lamc, lamc
        )
        lamc = np.where((lamg - lon_bp > 270), 360 - lamc, lamc)
        # Along symmetry meridian choose lamc
        lamc = np.where(
            (lamg - lon_bp == 90), 90, lamc
        )  # Along symmetry meridian choose lamc=90-lon_bp
        lamc = np.where(
            (lamg - lon_bp == 270), 270, lamc
        )  # Along symmetry meridian choose lamc=270-lon_bp
        lams = lamc + lon_bp

    ##Project back onto the larger (true) sphere so that the projected equator shrinks to latitude \phi_P=lat0_tp
    ##then we have tan(\phi_s'/2)=tan(\phi_p'/2)tan(\phi_c'/2)
    A = sinla * sphig
    chic = np.arccos(A)
    phis = 90 - 2 * np.arctan(rp * np.tan(chic / 2)) / PI_180
    ##Calculate the Metrics
    rden2 = 1.0 / (1 + (rp * np.tan(chic / 2)) ** 2)
    M_inv = rp * (1 + (np.tan(chic / 2)) ** 2) * rden2
    M = 1 / M_inv
    chig = (90 - phig) * PI_180
    rden2 = 1.0 / (1 + (rp * np.tan(chig / 2)) ** 2)
    N = rp * (1 + (np.tan(chig / 2)) ** 2) * rden2
    N_inv = 1 / N
    cos2phis = (np.cos(phis * PI_180)) ** 2

    h_j_inv = (
        cos2phis * alpha2 * (1 - alpha2) * beta2_inv * (1 + beta2_inv) * (rden ** 2)
        + M_inv * M_inv * (1 - alpha2) * rden
    )
    # Deal with beta=0. Prove that cos2phis/alpha2 ---> 0 when alpha, beta  ---> 0
    h_j_inv = np.where(np.abs(beta2_inv) > HUGE, M_inv * M_inv, h_j_inv)
    h_j_inv = np.sqrt(h_j_inv) * N_inv

    h_i_inv = (
        cos2phis * (1 + beta2_inv) * (rden ** 2)
        + M_inv * M_inv * alpha2 * beta2_inv * rden
    )
    # Deal with beta=0
    h_i_inv = np.where(np.abs(beta2_inv) > HUGE, M_inv * M_inv, h_i_inv)
    h_i_inv = np.sqrt(h_i_inv)

    if not metrics_only:
        return lams, phis, h_i_inv, h_j_inv
    else:
        return h_i_inv, h_j_inv


def generate_bipolar_cap_mesh(Ni, Nj_ncap, lat0_bp, lon_bp, ensure_nj_even=True):
    # Define a (lon,lat) coordinate mesh on the Northern hemisphere of the globe sphere
    # such that the resolution of latg matches the desired resolution of the final grid along the symmetry meridian
    print("Generating bipolar grid bounded at latitude ", lat0_bp)
    if Nj_ncap % 2 != 0 and ensure_nj_even:
        print("   Supergrid has an odd number of area cells!")
        if ensure_nj_even:
            print("   The number of j's is not even. Fixing this by cutting one row.")
            Nj_ncap = Nj_ncap - 1

    lon_g = lon_bp + np.arange(Ni + 1) * 360.0 / float(Ni)
    lamg = np.tile(lon_g, (Nj_ncap + 1, 1))
    latg0_cap = lat0_bp + np.arange(Nj_ncap + 1) * (90 - lat0_bp) / float(Nj_ncap)
    phig = np.tile(latg0_cap.reshape((Nj_ncap + 1, 1)), (1, Ni + 1))
    rp = np.tan(0.5 * (90 - lat0_bp) * PI_180)
    lams, phis, h_i_inv, h_j_inv = bipolar_projection(lamg, phig, lon_bp, rp)
    h_i_inv = h_i_inv[:, :-1] * 2 * np.pi / float(Ni)
    h_j_inv = h_j_inv[:-1, :] * PI_180 * (90 - lat0_bp) / float(Nj_ncap)
    print("   number of js=", phis.shape[0])
    return lams, phis, h_i_inv, h_j_inv


def bipolar_cap_ij_array(i, j, Ni, Nj_ncap, lat0_bp, lon_bp, rp):
    long = lon_bp + i * 360.0 / float(Ni)
    latg = lat0_bp + j * (90 - lat0_bp) / float(Nj_ncap)
    lamg = np.tile(long, (latg.shape[0], 1))
    phig = np.tile(latg.reshape((latg.shape[0], 1)), (1, long.shape[0]))
    h_i_inv, h_j_inv = bipolar_projection(lamg, phig, lon_bp, rp, metrics_only=True)
    h_i_inv = h_i_inv * 2 * np.pi / float(Ni)
    h_j_inv = h_j_inv * (90 - lat0_bp) * PI_180 / float(Nj_ncap)
    return h_i_inv, h_j_inv


def bipolar_cap_metrics_quad_fast(order, nx, ny, lat0_bp, lon_bp, rp, Re=_default_Re):
    print("   Calculating bipolar cap metrics via quadrature ...")
    a, b = quad_positions(order)
    daq = np.zeros([ny + 1, nx + 1])
    dxq = np.zeros([ny + 1, nx + 1])
    dyq = np.zeros([ny + 1, nx + 1])

    j1d = np.empty([0])
    for j in range(0, ny + 1):
        j_s = b * j + a * (j + 1)
        if j_s[-1] == ny:
            j_s[-1] = ny - 0.001  # avoid phi=90 as this will cause errore.
        # Niki:Find a way to avoid this properly.
        # This could be a sign that there is still something
        # wrong with the h_j_inv calculations at phi=90 (beta=0).
        j1d = np.append(j1d, j_s)

    i1d = np.empty([0])
    for i in range(0, nx + 1):
        i_s = b * i + a * (i + 1)
        i1d = np.append(i1d, i_s)

    # dx,dy = bipolar_cap_ij_array(i1d,j1d,nx,ny,lat0_bp,lon_bp,rp)
    # Or to make it faster:
    nj, ni = j1d.shape[0], i1d.shape[0]  # Shape of results
    dj = min(nj, max(32 * 1024 // ni, 1))  # Stride to use that fits in memory
    lams, phis, dx, dy = (
        np.zeros((nj, ni)),
        np.zeros((nj, ni)),
        np.zeros((nj, ni)),
        np.zeros((nj, ni)),
    )
    for j in range(0, nj, dj):
        je = min(nj, j + dj)
        dx[j:je], dy[j:je] = bipolar_cap_ij_array(
            i1d, j1d[j:je], nx, ny, lat0_bp, lon_bp, rp
        )

    # reshape to send for quad averaging
    dx_r = dx.reshape(ny + 1, order, nx + 1, order)
    dy_r = dy.reshape(ny + 1, order, nx + 1, order)
    # area element
    dxdy_r = dx_r * dy_r

    for j in range(0, ny + 1):
        for i in range(0, nx + 1):
            daq[j, i] = quad_average_2d(dxdy_r[j, :, i, :])
            dxq[j, i] = quad_average(dx_r[j, 0, i, :])
            dyq[j, i] = quad_average(dy_r[j, :, i, 0])
    daq = daq[:-1, :-1] * Re * Re
    dxq = dxq[:, :-1] * Re
    dyq = dyq[:-1, :] * Re
    return dxq, dyq, daq


def quad_positions(n=3):
    """Returns weights wa and wb so that the element [xa,xb] is sampled at positions
    x=wa(xa+xb*xb)."""
    if n == 2:
        return np.array([0.0, 1.0]), np.array([1.0, 0.0])
    if n == 3:
        return np.array([0.0, 0.5, 1.0]), np.array([1.0, 0.5, 0.0])
    if n == 4:
        r5 = 0.5 / np.sqrt(5.0)
        return np.array([0.0, 0.5 - r5, 0.5 + r5, 1.0]), np.array(
            [1.0, 0.5 + r5, 0.5 - r5, 0.0]
        )
    if n == 5:
        r37 = 0.5 * np.sqrt(3.0 / 7.0)
        return np.array([0.0, 0.5 - r37, 0.5, 0.5 + r37, 1.0]), np.array(
            [1.0, 0.5 + r37, 0.5, 0.5 - r37, 0.0]
        )
    raise Exception("Uncoded order")


def quad_average(y):
    """Returns the average value found by quadrature at order n.
    y is a list of values in order from x=-1 to x=1."""
    if len(y) == 2:  # 1, 1
        d = 1.0 / 2.0
        return d * (y[0] + y[1])
    if len(y) == 3:  # 1/3, 4/3, 1/3
        d = 1.0 / 6.0
        return d * (4.0 * y[1] + (y[0] + y[2]))
    if len(y) == 4:  # 1/6, 5/6, 5/6, 1/6
        d = 1.0 / 12.0
        return d * (5.0 * (y[1] + y[2]) + (y[0] + y[3]))
    if len(y) == 5:  # 9/10, 49/90, 64/90, 49/90, 9/90
        d = 1.0 / 180.0
        return d * (64.0 * y[2] + (49.0 * (y[1] + y[3])) + 9.0 * (y[0] + y[4]))
    raise Exception("Uncoded order")


def quad_average_2d(y):
    """Returns the average value found by quadrature at order n.
    y is a list of values in order from x1=-1 to x1=1 and x2=-1 to x2=1."""
    if y.shape[0] != y.shape[1]:
        raise Exception("Input array is not squared!")

    if y.shape[0] == 2:  # 1, 1
        d = 1.0 / 2.0
        return d * d * (y[0, 0] + y[0, 1] + y[1, 0] + y[1, 1])
    if y.shape[0] == 3:  # 1/3, 4/3, 1/3
        d = 1.0 / 6.0
        return (
            d
            * d
            * (
                y[0, 0]
                + y[0, 2]
                + y[2, 0]
                + y[2, 2]
                + 4.0 * (y[0, 1] + y[1, 0] + y[1, 2] + y[2, 1] + 4.0 * y[1, 1])
            )
        )
    if y.shape[0] == 4:  # 1/6, 5/6, 5/6, 1/6
        d = 1.0 / 12.0
        #       return d * ( 5. * ( y[1] + y[2] ) + ( y[0] + y[3] ) )
        w = np.array([1.0, 5.0, 5.0, 1.0])
        ysum = 0.0
        for j in range(0, y.shape[0]):
            for i in range(0, y.shape[1]):
                ysum = ysum + w[i] * w[j] * y[j, i]
        return d * d * ysum
    if y.shape[0] == 5:  # 9/10, 49/90, 64/90, 49/90, 9/90
        d = 1.0 / 180.0
        # return d * ( 64.* y[2] + ( 49. * ( y[1] + y[3] ) )  + 9. * ( y[0] + y[4] ) )
        w = np.array([9.0, 49.0, 64.0, 49.0, 9.0])
        ysum = 0.0
        for j in range(0, y.shape[0]):
            for i in range(0, y.shape[1]):
                ysum = ysum + w[i] * w[j] * y[j, i]
        return d * d * ysum

    raise Exception("Uncoded order")


