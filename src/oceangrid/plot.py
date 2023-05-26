#!/usr/bin/env python

from __future__ import print_function

import numpypi.numpypi_series as np


def cut_below(lam, phi, lowerlat):
    nj, ni = lam.shape
    for j in range(0, nj):
        if phi[j, 0] > lowerlat:
            break
    jmin = j
    #    print("jmin",jmin)
    return lam[jmin:, :], phi[jmin:, :]


def cut_above(lam, phi, upperlat):
    nj, ni = lam.shape
    for j in range(0, nj):
        if phi[j, 0] > upperlat:
            break
    jmax = j
    #    print("jmax",jmax)
    return lam[0:jmax, :], phi[0:jmax, :]


# utility function to plot grids
def plot_mesh_in_latlon(
    lam,
    phi,
    stride=1,
    phi_color="k",
    lam_color="r",
    newfig=True,
    title=None,
    axis=None,
    block=False,
):
    import matplotlib.pyplot as plt
#    import cartopy

    if phi.shape != lam.shape:
        raise Exception("Ooops: lam and phi should have same shape")
    nj, ni = lam.shape
    if newfig:
        plt.figure(figsize=(10, 10))
    if axis is None:
        for i in range(0, ni, stride):
            plt.plot(lam[:, i], phi[:, i], lam_color)
        for j in range(0, nj, stride):
            plt.plot(lam[j, :], phi[j, :], phi_color)
    else:
        for i in range(0, ni, stride):
            axis.plot(lam[:, i], phi[:, i], lam_color)#if cartopy is available add argument transform=cartopy.crs.Geodetic()
        for j in range(0, nj, stride):
            axis.plot(lam[j, :], phi[j, :], phi_color)#if cartopy is available add argument transform=cartopy.crs.Geodetic()

    if title is not None:
        plt.title(title)
    if not block:
        plt.show()


def plot_mesh_in_xyz(
    lam,
    phi,
    stride=1,
    phi_color="k",
    lam_color="r",
    lowerlat=None,
    upperlat=None,
    newfig=True,
    title=None,
    axis=None,
    block=False,
):
    if lowerlat is not None:
        lam, phi = cut_below(lam, phi, lowerlat=lowerlat)
    if upperlat is not None:
        lam, phi = cut_above(lam, phi, upperlat=upperlat)
    x = np.cos(phi * PI_180) * np.cos(lam * PI_180)
    y = np.cos(phi * PI_180) * np.sin(lam * PI_180)
    z = np.sin(phi * PI_180)
    plot_mesh_in_latlon(
        x,
        y,
        stride=stride,
        phi_color=phi_color,
        lam_color=lam_color,
        newfig=newfig,
        title=title,
        axis=None,
        block=False,
    )


def displacedPoleCap_plot(
    x_s, y_s, lon0, lon_dp, lat0, stride=40, block=False, dplat=None
):
    #import cartopy.crs as ccrs
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 10))
    ax = plt.axes(projection="polar")
    #if cartopy is available one could use the following for projection for a more accurate plot
    #ccrs.NearsidePerspective(central_longitude=0.0, central_latitude=-90, satellite_height=3578400)
    #ax.stock_img()
    #ax.gridlines(draw_labels=True)
    plot_mesh_in_latlon(x_s, y_s, stride=stride, newfig=False, axis=ax, block=block)
    if dplat is not None:
        ax.plot(lon_dp, dplat, color="r", marker="*") #if cartopy is available add argument transform=ccrs.Geodetic()

    return ax

