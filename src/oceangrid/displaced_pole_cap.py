#!/usr/bin/env python

from __future__ import print_function

import numpypi.numpypi_series as np
from oceangrid.constants import _default_Re

###
# Displaced pole cap functions
###
def displacedPoleCap_projection(lon_grid, lat_grid, z_0, r_joint):
    r = np.tan((90 + lat_grid) * PI_180) / r_joint
    # Find the theta that has matching resolution at the unit circle with longitude at the joint
    # This is a conformal transformation of the unit circle (inverse to the one below)
    e2itheta = np.cos(lon_grid * PI_180) + 1j * np.sin(lon_grid * PI_180)
    e2ithetaprime = (e2itheta - z_0) / (1.0 - np.conj(z_0) * e2itheta)
    # Conformal map to displace pole from r=0 to r=r_dispole
    z = r * e2ithetaprime
    w = (z + z_0) / (1 + np.conj(z_0) * z)
    # Inverse projection from tangent plane back to sphere
    lamcDP = np.angle(w, deg=True)
    # lamcDP = np.arctan2(np.imag(w), np.real(w))/PI_180
    # np.angle returns a value in the interval (-180,180)
    # However the input grid longitude is in (-lon0,-lon0+360), e.g., (-300,60)
    # We should shift the angle to be in that interval
    ##But we should also be careful to produce a monotonically increasing longitude, starting from lon0.
    lamcDP = monotonic_bounding(lamcDP, lon_grid[0, 0])
    #
    rw = np.absolute(w)
    phicDP = -90 + np.arctan(rw * r_joint) / PI_180
    return lamcDP, phicDP


def monotonic_bounding(x, x_0):
    x_im1 = x[:, 0] * 0 + x_0  # Initial value
    for i in range(0, x.shape[1]):
        x[:, i] = np.where(x[:, i] - x_im1[:] > 100, x[:, i] - 360, x[:, i])
        x_im1[:] = x[:, i]
    return x


def displacedPoleCap_baseGrid(i, j, ni, nj, lon0, lat0):
    u = lon0 + i * 360.0 / float(ni)
    a = -90.0
    b = lat0
    v = a + j * (b - a) / float(nj)
    du = np.roll(u, shift=-1, axis=0) - u
    dv = np.roll(v, shift=-1, axis=0) - v
    return u, v, du, dv


def displacedPoleCap_mesh(
    i, j, ni, nj, lon0, lat0, lam_pole, r_pole, excluded_fraction=None
):

    long, latg, du, dv = displacedPoleCap_baseGrid(i, j, ni, nj, lon0, lat0)
    lamg = np.tile(long, (latg.shape[0], 1))
    phig = np.tile(latg.reshape((latg.shape[0], 1)), (1, long.shape[0]))
    # Projection from center of globe to plane tangent at south pole
    r_joint = np.tan((90 + lat0) * PI_180)
    z_0 = r_pole * (np.cos(lam_pole * PI_180) + 1j * np.sin(lam_pole * PI_180))
    lams, phis = displacedPoleCap_projection(lamg, phig, z_0, r_joint)
    londp = lams[0, 0]
    latdp = phis[0, 0]
    if excluded_fraction is not None:
        ny, nx = lamg.shape
        jmin = np.ceil(excluded_fraction * ny)
        jmin = jmin + np.mod(jmin, 2)
        jmint = int(jmin)
        return lams[jmint:, :], phis[jmint:, :], londp, latdp
    else:
        return lams, phis, londp, latdp


def generate_displaced_pole_grid(Ni, Nj_scap, lon0, lat0, lon_dp, r_dp):
    print("Generating displaced pole grid bounded at latitude ", lat0)
    print("   requested displaced pole lon,rdp=", lon_dp, r_dp)
    i_s = np.arange(Ni + 1)
    j_s = np.arange(Nj_scap + 1)
    x, y, londp, latdp = displacedPoleCap_mesh(
        i_s, j_s, Ni, Nj_scap, lon0, lat0, lon_dp, r_dp
    )
    print("   generated displaced pole lon,lat=", londp, latdp)
    return x, y, londp, latdp


# numerical approximation of metrics coefficients h_i and h_j
def great_arc_distance(j0, i0, j1, i1, nx, ny, lon0, lat0, lon_dp, r_dp):
    """Returns great arc distance between nodes (j0,i0) and (j1,i1)"""
    # https://en.wikipedia.org/wiki/Great-circle_distance
    lam0, phi0, x, y = displacedPoleCap_mesh(i0, j0, nx, ny, lon0, lat0, lon_dp, r_dp)
    lam1, phi1, x, y = displacedPoleCap_mesh(i1, j1, nx, ny, lon0, lat0, lon_dp, r_dp)
    lam0, phi0 = lam0 * PI_180, phi0 * PI_180
    lam1, phi1 = lam1 * PI_180, phi1 * PI_180
    dphi, dlam = phi1 - phi0, lam1 - lam0
    # Haversine formula
    d = np.sin(0.5 * dphi) ** 2 + np.sin(0.5 * dlam) ** 2 * np.cos(phi0) * np.cos(phi1)
    return 2.0 * np.arcsin(np.sqrt(d))


def numerical_hi(j, i, nx, ny, lon0, lat0, lon_dp, r_dp, eps, order=6):
    """Returns a numerical approximation to h_lambda"""
    reps = 1.0 / eps
    ds2 = great_arc_distance(j, i + eps, j, i - eps, nx, ny, lon0, lat0, lon_dp, r_dp)
    if order == 2:
        return 0.5 * ds2 * reps
    ds4 = great_arc_distance(
        j, i + 2.0 * eps, j, i - 2.0 * eps, nx, ny, lon0, lat0, lon_dp, r_dp
    )
    if order == 4:
        return (8.0 * ds2 - ds4) * (1.0 / 12.0) * reps
    ds6 = great_arc_distance(
        j, i + 3.0 * eps, j, i - 3.0 * eps, nx, ny, lon0, lat0, lon_dp, r_dp
    )
    if order == 6:
        return (45.0 * ds2 - 9.0 * ds4 + ds6) * (1.0 / 60.0) * reps
    raise Exception("order not coded")


def numerical_hj(j, i, nx, ny, lon0, lat0, lon_dp, r_dp, eps, order=6):
    """Returns a numerical approximation to h_phi"""
    reps = 1.0 / eps
    ds2 = great_arc_distance(j + eps, i, j - eps, i, nx, ny, lon0, lat0, lon_dp, r_dp)
    if order == 2:
        return 0.5 * ds2 * reps
    ds4 = great_arc_distance(
        j + 2.0 * eps, i, j - 2.0 * eps, i, nx, ny, lon0, lat0, lon_dp, r_dp
    )
    if order == 4:
        return (8.0 * ds2 - ds4) * (1.0 / 12.0) * reps
    ds6 = great_arc_distance(
        j + 3.0 * eps, i, j - 3.0 * eps, i, nx, ny, lon0, lat0, lon_dp, r_dp
    )
    if order == 6:
        return (45.0 * ds2 - 9.0 * ds4 + ds6) * (1.0 / 60.0) * reps
    raise Exception("order not coded")


def displacedPoleCap_metrics_quad(
    order, nx, ny, lon0, lat0, lon_dp, r_dp, Re=_default_Re
):
    print("   Calculating displaced pole cap metrics via quadrature ...")
    a, b = quad_positions(order)
    # Note that we need to include the index of the last point of the grid to do the quadrature correctly.
    daq = np.zeros([ny + 1, nx + 1])
    dxq = np.zeros([ny + 1, nx + 1])
    dyq = np.zeros([ny + 1, nx + 1])

    j1d = np.empty([0])
    for j in range(0, ny + 1):
        j_s = b * j + a * (j + 1)
        j1d = np.append(j1d, j_s)

    i1d = np.empty([0])
    for i in range(0, nx + 1):
        i_s = b * i + a * (i + 1)
        i1d = np.append(i1d, i_s)
    # numerical approximation to h_i_in and h_j_inv at quadrature points
    dx = numerical_hi(j1d, i1d, nx, ny, lon0, lat0, lon_dp, r_dp, eps=1e-3, order=order)
    dy = numerical_hj(j1d, i1d, nx, ny, lon0, lat0, lon_dp, r_dp, eps=1e-3, order=order)
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
