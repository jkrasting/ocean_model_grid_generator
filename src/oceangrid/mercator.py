#!/usr/bin/env python

from __future__ import print_function

import numpypi.numpypi_series as np

from oceangrid.util import lagrange_interp
from oceangrid.constants import _default_Re, PI_180

def y_mercator(Ni, phi):
    """Equation (1)"""
    R = Ni / (2 * np.pi)
    return R * (np.log((1.0 + np.sin(phi)) / np.cos(phi)))


def phi_mercator(Ni, y):
    """Equation (2)"""
    R = Ni / (2 * np.pi)
    return np.arctan(np.sinh(y / R)) * (180 / np.pi)  # Converted to degrees


def y_mercator_rounded(Ni, phi):
    y_float = y_mercator(Ni, phi)
    return (np.sign(y_float) * np.round_(np.abs(y_float))).astype(int)


def generate_mercator_grid(
    Ni,
    phi_s,
    phi_n,
    lon0_M,
    lenlon_M,
    refineR,
    shift_equator_to_u_point=True,
    ensure_nj_even=True,
    enhanced_equatorial=0,
):
    print("Requesting Mercator grid with phi range: phi_s,phi_n=", phi_s, phi_n)
    # Diagnose nearest integer y(phi range)
    y_star = y_mercator_rounded(Ni, np.array([phi_s * PI_180, phi_n * PI_180]))
    print("   y*=", y_star, "nj=", y_star[1] - y_star[0] + 1)
    # Ensure that the equator (y=0) is a u-point
    if y_star[0] % 2 == 0:
        print("  *Equator may not be a u-point!")
        # There is another check for this for the whole grid.
        if shift_equator_to_u_point:
            print("  *Fixing this by shifting the bounds!")
            y_star[0] = y_star[0] - 1
            y_star[1] = y_star[1] - 1
            print("   y*=", y_star, "nj=", y_star[1] - y_star[0] + 1)
    if (y_star[1] - y_star[0] + 1) % 2 == 0:
        print("  *Supergrid has an odd number of area cells!")
        if ensure_nj_even:
            print("  *Fixing this by shifting the y_star[1] ")
            y_star[1] = y_star[1] - 1
    Nj = y_star[1] - y_star[0]
    print(
        "   Generating Mercator grid with phi range: phi_s,phi_n=",
        phi_mercator(Ni, y_star),
    )
    phi_M = phi_mercator(Ni, np.arange(y_star[0], y_star[1] + 1))

    # Ensure that the equator (y=0) is included and is a u-point
    equator = 0.0
    equator_index = np.searchsorted(phi_M, equator)
    if equator_index == 0:
        raise Exception("   Ooops: Equator is not in the grid")
    else:
        print("   Equator is at j=", equator_index)
    # Ensure that the equator (y=0) is a u-point
    if equator_index % 2 == 0:
        print("  *Equator is not going to be a u-point of this grid patch.")

    if enhanced_equatorial:
        print("   Enhancing the equator region resolution")
        # Enhance the lattitude resolution between 30S and 30N
        # Set a constant high res lattitude grid spanning 10 degrees centered at the Equator.
        # This construction makes the whole Mercator subgrid symmetric around the Equator.
        #
        # MIDAS parameters. Where does this come from and how should it change with resolution?
        phi_enh_d = -5.0  # Starting lattitude of enhanced resolution grid
        phi_cub_d = -30  # Starting lattitude of cubic interpolation

        N_cub = (
            132 * refineR / 2
        )  # Number of points in the cubic interpolation for one shoulder
        # MIDAS has 130, but 132 produces a result closer to 1/2 degree MIDAS grid
        dphi_e = 0.13 * 2 / refineR  # Enhanced resolution 10 degrees around the equator
        N_enh = (
            40 * refineR / 2
        )  # Number of points in the enhanced resolution below equator

        if refineR == 1 and enhanced_equatorial:  # Closest to SPEAR grid
            phi_enh_d = -10
            phi_cub_d = -20
            N_cub = 29
            N_enh = 55
            dphi_e = -phi_enh_d / N_enh / 0.981

        if refineR == 4 and enhanced_equatorial==8:
            #1/8 degree refine 
            phi_enh_d = -10
            N_enh = 2*enhanced_equatorial * abs(phi_enh_d)+1 #161 
            phi_cub_d = -20
            N_cub = 101
            dphi_e = -phi_enh_d / N_enh

        if refineR == 4 and enhanced_equatorial==6:
            #1/6 degree refine 
            phi_enh_d = -10
            N_enh = 2*enhanced_equatorial * abs(phi_enh_d)+1 #121 
            phi_cub_d = -20
            N_cub = 101  #What determines this?
            dphi_e = -phi_enh_d / N_enh

        j_c0d = np.where(phi_M < phi_enh_d)[0][-1]        # The last index with phi_M<phi_enh_d
        j_phi_cub_d = np.where(phi_M < phi_cub_d)[0][-1]  # The last index with phi_M<phi_cub_d
        dphi = phi_M[1:] - phi_M[0:-1]

        cubic_lagrange_interp = True
        cubic_scipy = False

        phi1 = phi_M[0:j_phi_cub_d]
        phi_s = phi_M[j_phi_cub_d - 1]
        dphi_s = phi_M[j_phi_cub_d] - phi_M[j_phi_cub_d - 1]
        phi_e = phi_enh_d

        nodes = [0, 1, N_cub - 2, N_cub - 1]
        phi_nodes = [phi_s, phi_s + dphi_s, phi_e - dphi_e, phi_e]
        q = np.arange(N_cub)

        #cubic_lagrange_interp:
        phi2 = lagrange_interp(nodes, phi_nodes, q)

        print(
            "   Meridional range of pure Mercator=(",
            phi1[0],
            ",",
            phi1[-2],
            ") U (",
            -phi1[-2],
            ",",
            -phi1[0],
            ").",
        )
        print(
            "   Meridional range of cubic interpolation=(",
            phi2[0],
            ",",
            phi2[-2],
            ") U (",
            -phi2[-2],
            ",",
            -phi2[0],
            ").",
        )
        phi3 = np.concatenate((phi1[0:-1], phi2))

        phi_s = phi3[-1]
        phi4 = np.linspace(phi_s, 0, int(N_enh))
        print(
            "   Meridional range of enhanced resolution=(", phi4[0], ",", -phi4[0], ")."
        )
        print("   Meridional value of enhanced resolution=", phi4[1] - phi4[0])
        phi5 = np.concatenate((phi3[0:-1], phi4))
        # Make the grid symmetric around the equator!!!!
        phi_M = np.concatenate((phi5[0:-1], -phi5[::-1]))

        # limit the upper lattitude by the requested phi_n
        j_phi_n = np.where(phi_M < phi_n)[0][-1]  # The last index with phi_M<phi_n
        phi_M = phi_M[0:j_phi_n]
        Nj = phi_M.shape[0] - 1

    y_grid_M = np.tile(phi_M.reshape(Nj + 1, 1), (1, Ni + 1))
    lam_M = lon0_M + np.arange(Ni + 1) * lenlon_M / float(Ni)
    x_grid_M = np.tile(lam_M, (Nj + 1, 1))
    # Double check is necessary for enhanced_equatorial
    if y_grid_M.shape[0] % 2 == 0 and ensure_nj_even:
        print(
            "   The number of j's is not even. Fixing this by cutting one row at south."
        )
        y_grid_M = np.delete(y_grid_M, 0, 0)
        x_grid_M = np.delete(x_grid_M, 0, 0)
    print("   Final Mercator grid range=", y_grid_M[0, 0], y_grid_M[-1, 0])
    print("   number of js=", y_grid_M.shape[0])

    return x_grid_M, y_grid_M

