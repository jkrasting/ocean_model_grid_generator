#!/usr/bin/env python

from __future__ import print_function

import numpypi.numpypi_series as np

from oceangrid.constants import _default_Re, PI_180

def chksum(x, lbl):
    import hashlib

    if type(x) in (float, int, np.float64):
        y = np.array(x)
    else:
        y = np.zeros(x.shape)
        y[:] = x
    ymin, ymax, ymean = y.min(), y.max(), y.mean()
    ysd = np.sqrt(((y - ymean) ** 2).mean())
    print(
        hashlib.sha256(y).hexdigest(),
        "%10s" % lbl,
        "min = %.15f" % ymin,
        "max = %.15f" % ymax,
        "mean = %.15f" % ymean,
        "sd = %.15f" % ysd,
    )

def lagrange_interp(x, y, q):
    """Lagrange polynomial interpolation. Retruns f(q) which f(x) passes through four data
    points at x[0..3], y[0..3]."""
    # n - numerator, d - denominator
    n0 = (q - x[1]) * (q - x[2]) * (q - x[3])
    d0 = (x[0] - x[1]) * (x[0] - x[2]) * (x[0] - x[3])
    n1 = (q - x[0]) * (q - x[2]) * (q - x[3])
    d1 = (x[1] - x[0]) * (x[1] - x[2]) * (x[1] - x[3])
    n2 = (q - x[0]) * (q - x[1]) * (q - x[3])
    d2 = (x[2] - x[0]) * (x[2] - x[1]) * (x[2] - x[3])
    n3 = (q - x[0]) * (q - x[1]) * (q - x[2])
    d3 = (x[3] - x[0]) * (x[3] - x[1]) * (x[3] - x[2])
    return ((n0 / d0) * y[0] + (n3 / d3) * y[3]) + ((n1 / d1) * y[1] + (n2 / d2) * y[2])

def mdist(x1, x2):
    """Returns positive distance modulo 360."""
    return np.minimum(np.mod(x1 - x2, 360.0), np.mod(x2 - x1, 360.0))


def generate_grid_metrics_MIDAS(
    x, y, axis_units="degrees", Re=_default_Re, latlon_areafix=True
):
    nytot, nxtot = x.shape
    if axis_units == "m":
        metric = 1.0
    if axis_units == "km":
        metric = 1.0e3
    if axis_units == "degrees":
        metric = Re * PI_180
    lv = (0.5 * (y[:, 1:] + y[:, :-1])) * PI_180
    dx_i = mdist(x[:, 1:], x[:, :-1]) * PI_180
    dy_i = (y[:, 1:] - y[:, :-1]) * PI_180
    dx = Re * np.sqrt(dy_i ** 2 + (dx_i * np.cos(lv)) ** 2)
    lu = (0.5 * (y[1:, :] + y[:-1, :])) * PI_180
    dx_j = mdist(x[1:, :], x[:-1, :]) * PI_180
    dy_j = (y[1:, :] - y[:-1, :]) * PI_180
    dy = Re * np.sqrt(dy_j ** 2 + (dx_j * np.cos(lu)) ** 2)

    ymid_j = 0.5 * (y + np.roll(y, shift=-1, axis=0))
    ymid_i = 0.5 * (y + np.roll(y, shift=-1, axis=1))
    dy_j = np.roll(y, shift=-1, axis=0) - y
    dy_i = np.roll(y, shift=-1, axis=1) - y
    dx_i = mdist(np.roll(x, shift=-1, axis=1), x)
    dx_j = mdist(np.roll(x, shift=-1, axis=0), x)
    if latlon_areafix:
        sl = np.sin(lv)
        dx_i = mdist(x[:, 1:], x[:, :-1]) * PI_180
        area = (Re ** 2) * (
            (0.5 * (dx_i[1:, :] + dx_i[:-1, :])) * (sl[1:, :] - sl[:-1, :])
        )
    else:
        area = 0.25 * ((dx[1:, :] + dx[:-1, :]) * (dy[:, 1:] + dy[:, :-1]))
    return dx, dy, area


def angle_x(x, y):
    """Returns the orientation angle of the grid box"""
    if x.shape != y.shape:
        raise Exception("Input arrays do not have the same shape!")
    angle_dx = np.zeros(x.shape)
    # The corrected version of angle_dx, in addition to including spherical metrics, is centered in the interior and one-sided at the grid edges
    angle_dx[:, 1:-1] = np.arctan2(
        y[:, 2:] - y[:, :-2], (x[:, 2:] - x[:, :-2]) * np.cos(y[:, 1:-1] * PI_180)
    )
    angle_dx[:, 0] = np.arctan2(
        y[:, 1] - y[:, 0], (x[:, 1] - x[:, 0]) * np.cos(y[:, 0] * PI_180)
    )
    angle_dx[:, -1] = np.arctan2(
        y[:, -1] - y[:, -2], (x[:, -1] - x[:, -2]) * np.cos(y[:, -1] * PI_180)
    )
    angle_dx = angle_dx / PI_180
    return angle_dx


def metrics_error(
    dx_,
    dy_,
    area_,
    Ni,
    lat1,
    lat2=90,
    Re=_default_Re,
    bipolar=False,
    displaced_pole=-999,
    excluded_fraction=None,
):
    exact_area = (
        2 * np.pi * (Re ** 2) * np.abs(np.sin(lat2 * PI_180) - np.sin(lat1 * PI_180))
    )
    exact_lat_arc_length = np.abs(lat2 - lat1) * PI_180 * Re
    exact_lon_arc_length = np.cos(lat1 * PI_180) * 2 * np.pi * Re
    grid_lat_arc_length = np.sum(dy_[:, Ni // 4])
    grid_lon_arc_length = np.sum(dx_[0, :])
    if lat1 > lat2:
        grid_lon_arc_length = np.sum(dx_[-1, :])
    if bipolar:
        # length of the fold
        grid_lon_arc_length2 = np.sum(dx_[-1, :])
        # This must be 4*grid_lat_arc_length
        lon_arc2_error = (
            100
            * (grid_lon_arc_length2 / 4 - exact_lat_arc_length)
            / exact_lat_arc_length
        )
    area_error = 100 * (np.sum(area_) - exact_area) / exact_area
    lat_arc_error = (
        100 * (grid_lat_arc_length - exact_lat_arc_length) / exact_lat_arc_length
    )
    lon_arc_error = (
        100 * (grid_lon_arc_length - exact_lon_arc_length) / exact_lon_arc_length
    )
    if displaced_pole != -999:
        antipole = displaced_pole + Ni // 2
        if displaced_pole > Ni // 2:
            antipole = displaced_pole - Ni // 2
        grid_lat_arc_length = np.sum(dy_[:, displaced_pole]) + np.sum(dy_[:, antipole])
        lat_arc_error = (
            100
            * (grid_lat_arc_length - 2.0 * exact_lat_arc_length)
            / exact_lat_arc_length
        )
    if excluded_fraction:
        print(
            "   Cannot estimate area and dy accuracies with excluded_fraction (doughnut)! "
        )
    if bipolar:
        return area_error, lat_arc_error, lon_arc_error, lon_arc2_error
    else:
        return area_error, lat_arc_error, lon_arc_error


def write_nc(
    x,
    y,
    dx,
    dy,
    area,
    angle_dx,
    axis_units="degrees",
    fnam=None,
    format="NETCDF3_64BIT",
    description=None,
    history=None,
    source=None,
    no_changing_meta=None,
    debug=False,
):
    import netCDF4 as nc

    if fnam is None:
        fnam = "supergrid.nc"
    fout = nc.Dataset(fnam, "w", clobber=True, format=format)

    if debug:
        chksum(x, "x")
        chksum(y, "y")
        chksum(dx, "dx")
        chksum(dy, "dy")
        chksum(area, "area")
        chksum(angle_dx, "angle_dx")

    ny = area.shape[0]
    nx = area.shape[1]
    nyp = ny + 1
    nxp = nx + 1
    print("   Writing netcdf file with ny,nx= ", ny, nx)

    nyp = fout.createDimension("nyp", nyp)
    nxp = fout.createDimension("nxp", nxp)
    ny = fout.createDimension("ny", ny)
    nx = fout.createDimension("nx", nx)
    string = fout.createDimension("string", 255)
    tile = fout.createVariable("tile", "S1", ("string"))
    yv = fout.createVariable("y", "f8", ("nyp", "nxp"))
    xv = fout.createVariable("x", "f8", ("nyp", "nxp"))
    yv.units = "degrees"
    xv.units = "degrees"
    yv[:] = y
    xv[:] = x
    stringvals = np.empty(1, "S" + repr(len(tile)))
    stringvals[0] = "tile1"
    tile[:] = nc.stringtochar(stringvals)
    dyv = fout.createVariable("dy", "f8", ("ny", "nxp"))
    dyv.units = "meters"
    dyv[:] = dy
    dxv = fout.createVariable("dx", "f8", ("nyp", "nx"))
    dxv.units = "meters"
    dxv[:] = dx
    areav = fout.createVariable("area", "f8", ("ny", "nx"))
    areav.units = "m2"
    areav[:] = area
    anglev = fout.createVariable("angle_dx", "f8", ("nyp", "nxp"))
    anglev.units = "degrees"
    anglev[:] = angle_dx
    # global attributes
    if not no_changing_meta:
        fout.history = history
        fout.description = description
        fout.source = source

    fout.sync()
    fout.close()

