#!/usr/bin/env python

from __future__ import print_function

import numpypi.numpypi_series as np

# import numpy as np
import argparse
import sys
import datetime, subprocess

from oceangrid import *

def usage():
    print(
        "ocean_grid_generator.py -f <output_grid_filename> -r <inverse_degrees_resolution> [--rdp=<displacement_factor/0.2> --south_cutoff_ang=<degrees_south_to_start> --south_cutoff_row=<rows_south_to_cut> --match_dy --even_j --plot --write_subgrid_files --enhanced_equatorial --no-metrics --gridlist=sc]"
    )


def main(
    inverse_resolution,
    gridfilename="ocean_hgrid.nc",
    r_dp=0.0,
    lon_dp=80.0,
    lat_dp=-99.0,
    exfracdp=None,
    south_cutoff_row=0,
    south_cutoff_ang=-90.0,
    reproduce_MIDAS_grids=False,
    match_dy=False,
    write_subgrid_files=False,
    plotem=False,
    no_changing_meta=False,
    enhanced_equatorial=0,
    debug=False,
    grids=["bipolar", "mercator", "so", " sc", "all"],
    skip_metrics=False,
    ensure_nj_even=False,
    shift_equator_to_u_point=True,
    south_cap_lat=-99.0,
    south_ocean_upper_lat=-99.0,
    no_south_cap=False,
):

    # fraction of dp grid to be excluded/cut, this particular value was used for the OM4 1/4 degree grid
    doughnut = 0.28 * 7 / 4
    doughnut = exfracdp if (exfracdp is not None) else doughnut

    south_cap = not no_south_cap
    calculate_metrics = not skip_metrics
    degree_resolution_inverse = inverse_resolution

    hasBP = False
    hasMerc = False
    hasSO = False
    hasSC = False

    # Exit if mutually exclusive arguments are provided
    if r_dp != 0.0 and lat_dp > -90.0:
        print("Cannot specify both --rdp and --latdp for the displaced pole!")
        usage()
        sys.exit(2)

    # Information to write in file as metadata
    if not no_changing_meta:
        import socket

        host = str(socket.gethostname())
        scriptpath = sys.argv[0]
        scriptbasename = (
            subprocess.check_output("basename " + scriptpath, shell=True)
            .decode("ascii")
            .rstrip("\n")
        )
        scriptdirname = (
            subprocess.check_output("dirname " + scriptpath, shell=True)
            .decode("ascii")
            .rstrip("\n")
        )
        scriptgithash = (
            subprocess.check_output(
                "cd " + scriptdirname + ";git rev-parse HEAD; exit 0",
                stderr=subprocess.STDOUT,
                shell=True,
            )
            .decode("ascii")
            .rstrip("\n")
        )
        scriptgitMod = (
            subprocess.check_output(
                "cd "
                + scriptdirname
                + ";git status --porcelain "
                + scriptbasename
                + " | awk '{print $1}' ; exit 0",
                stderr=subprocess.STDOUT,
                shell=True,
            )
            .decode("ascii")
            .rstrip("\n")
        )
        if "M" in str(scriptgitMod):
            scriptgitMod = " , But was localy Modified!"

    hist = "This grid file was generated via command " + " ".join(sys.argv)
    if not no_changing_meta:
        hist = hist + " on " + str(datetime.date.today()) + " on platform " + host

    desc = (
        "This is an orthogonal coordinate grid for the Earth with a nominal resoution of "
        + str(1 / degree_resolution_inverse)
        + " degrees along the equator. "
    )

    source = ""
    if not no_changing_meta:
        source = source + scriptpath + " had git hash " + scriptgithash + scriptgitMod
        source = (
            source
            + ". To obtain the grid generating code do: git clone  https://github.com/nikizadehgfdl/grid_generation.git ; cd grid_generation;  git checkout "
            + scriptgithash
        )

    import time

    start_time = time.time()
    # Specify the default grid properties
    refineS = 2  # factor 2 is for supergrid
    refineR = degree_resolution_inverse
    lenlon = 360  # global longitude range
    lon0 = -300.0  # Starting longitude of the map
    Ni = int(refineR * refineS * lenlon)
    ###
    ###Mercator grid
    ###
    # MIDAS has nominal starting latitude for Mercator grid = -65 for 1/4 degree, -70 for 1/2 degree
    # MIDAS has nominal latitude range of Mercator grid     = 125 for 1/4 degree, 135 for 1/2 degree
    # Instead we use:
    phi_s_Merc, phi_n_Merc = -66.85954725, 64.05895973
    if refineR == 2:
        # phi_s_Merc, phi_n_Merc = -68.05725376601046, 65.0 #These give a 1/2 degree enhanced equatorial close to MIDAS result
        # shift_equator_to_u_point= False
        phi_s_Merc, phi_n_Merc = -68.0, 65.0
    if refineR == 1 and enhanced_equatorial:  # Closest to SPEAR grid
        # shift_equator_to_u_point=True
        phi_s_Merc, phi_n_Merc = -77.8, 60.0
    ###
    # Southern Ocean grid
    ###
    lat0_SO = -78.0  # Starting lower lat of Southern Ocean grid
    if south_cap_lat > -90:
        lat0_SO = south_cap_lat
    latUp_SO = phi_s_Merc
    if south_ocean_upper_lat > -90:
        latUp_SO = south_ocean_upper_lat
    lenlat_SO = latUp_SO - lat0_SO
    deltaPhiSO = 1.0 / refineR / refineS
    # To get the same number of points as existing 1/2 and 1/4 degree grids that were generated with MIDAS
    Nj_SO = int(refineR * 55)
    if refineR == 2:
        Nj_SO = 54 * refineS + 1
    if refineR == 1 and enhanced_equatorial:
        Nj_SO = 0
    ###
    # Bipolar cap
    ###
    lon_bp = lon0  # longitude of the bipole(s)
    # To get the same number of points as existing 1/2 and 1/4 degree grids that were generated with MIDAS
    Nj_ncap = int(60 * refineR * refineS)
    if refineR == 2:
        Nj_ncap = 119 * refineS
    if refineR == 1 and enhanced_equatorial:
        Nj_ncap = 154  # SPEAR
    ###
    # South cap
    ###
    # To get the same number of points as existing 1/4 degree grids that were generated with MIDAS
    Nj_scap = int(refineR * 40)
    if no_south_cap:
       Nj_scap = 0
    if refineR == 1 and enhanced_equatorial:  # SPEAR
        Nj_scap = 0
    # Refine the grid by a factor and then exclude the inner circle corresponding to that factor
    # These factors are heuristic and we adjust them to get a grid with the same number of points
    # as the existing 1/4 degree grid of OM4p25
    Nj_scap = Nj_scap * 7 // 4

    lat0_SC = lat0_SO

    if "mercator" in grids or "all" in grids:
        lamMerc, phiMerc = generate_mercator_grid(
            Ni,
            phi_s_Merc,
            phi_n_Merc,
            lon0,
            lenlon,
            refineR,
            shift_equator_to_u_point=shift_equator_to_u_point,
            ensure_nj_even=ensure_nj_even,
            enhanced_equatorial=enhanced_equatorial,
        )
        angleMerc = angle_x(lamMerc, phiMerc)
        dxMerc = -np.ones([lamMerc.shape[0], lamMerc.shape[1] - 1])
        dyMerc = -np.ones([lamMerc.shape[0] - 1, lamMerc.shape[1]])
        areaMerc = -np.ones([lamMerc.shape[0] - 1, lamMerc.shape[1] - 1])
        hasMerc = True
        if calculate_metrics:
            # For spherical grids we can safely use the MIDAS algorithm for calculating the metrics
            dxMerc, dyMerc, areaMerc = generate_grid_metrics_MIDAS(lamMerc, phiMerc)
            print(
                "   CHECK_metrics: % errors in (area, lat arc, lon arc)",
                metrics_error(
                    dxMerc, dyMerc, areaMerc, Ni, phiMerc[0, 0], phiMerc[-1, 0]
                ),
            )
        
        if write_subgrid_files:
            write_nc(
                lamMerc,
                phiMerc,
                dxMerc,
                dyMerc,
                areaMerc,
                angleMerc,
                axis_units="degrees",
                fnam=gridfilename + "Merc.nc",
                description=desc,
                history=hist,
                source=source,
                debug=debug,
            )

        # The phi resolution in the first and last row of Mercator grid along the symmetry meridian
        DeltaPhiMerc_so = phiMerc[1, Ni // 4] - phiMerc[0, Ni // 4]
        DeltaPhiMerc_no = phiMerc[-1, Ni // 4] - phiMerc[-2, Ni // 4]
        # Start lattitude from dy above the last Mercator grid
        lat0_bp = phiMerc[-1, Ni // 4] + DeltaPhiMerc_no
        if refineR == 1 and enhanced_equatorial:
            lat0_bp = phiMerc[-1, Ni // 4]  # SPEAR
        if match_dy:
            # The bopolar grid should start from the same lattitude that Mercator ends.
            # Then when we combine the two grids we should drop the x,y,dx,angle from one of the two.
            # This way we get a continous dy and area.
            lat0_bp = phiMerc[-1, Ni // 4]
            # Determine the number of bipolar cap grid point in the y direction such that the y resolution
            # along symmetry meridian is a constant and is equal to (continuous with) the last Mercator dy.
            # Note that int(0.5+x) is used to return the nearest integer to a float with deterministic
            # behavior for middle points.
            # Note that int(0.5+x) is equivalent to math.floor(0.5+x)
            Nj_ncap = int(
                0.5 + (90.0 - lat0_bp) / DeltaPhiMerc_no
            )  # Impose boundary condition for smooth dy

            # Make the last SO grid point a (Mercator) step below the first Mercator lattitude.
            # lenlat_SO = phiMerc[0,Ni//4] - DeltaPhiMerc_so - lat0_SO #Start from a lattitude to smooth out dy.
            # Niki: I think this is wrong!
            # The SO grid should end at the same lattitude that Mercator starts.
            # Then when we combine the two grids we should drop the x,y,dx,angle from one of the two.
            # This way we get a continous dy and area.
            lenlat_SO = phiMerc[0, Ni // 4] - lat0_SO
            # Determine the number of grid point in the y direction such that the y resolution is equal to
            # (continuous with) the first Mercator dy.
            Nj_SO = int(
                0.5 + lenlat_SO / DeltaPhiMerc_so
            )  # Make the resolution continious with the Mercator at joint

    ###
    ###Northern bipolar cap
    ###
    if "bipolar" in grids or "all" in grids:
        # Generate the bipolar grid
        lamBP, phiBP, dxBP_h, dyBP_h = generate_bipolar_cap_mesh(
            Ni, Nj_ncap, lat0_bp, lon_bp, ensure_nj_even=ensure_nj_even
        )
        # Metrics via quadratue of h's
        rp = np.tan(0.5 * (90 - lat0_bp) * PI_180)
        dxBP = -np.ones([lamBP.shape[0], lamBP.shape[1] - 1])
        dyBP = -np.ones([lamBP.shape[0] - 1, lamBP.shape[1]])
        areaBP = -np.ones([lamBP.shape[0] - 1, lamBP.shape[1] - 1])
        hasBP = True
        if calculate_metrics:
            dxBP, dyBP, areaBP = bipolar_cap_metrics_quad_fast(
                5, phiBP.shape[1] - 1, phiBP.shape[0] - 1, lat0_bp, lon_bp, rp
            )
            print(
                "   CHECK_metrics_hquad: % errors in (area, lat arc, lon arc1, lon arc2)",
                metrics_error(dxBP, dyBP, areaBP, Ni, lat0_bp, 90.0, bipolar=True),
            )
        angleBP = angle_x(lamBP, phiBP)

        if write_subgrid_files:
            write_nc(
                lamBP,
                phiBP,
                dxBP,
                dyBP,
                areaBP,
                angleBP,
                axis_units="degrees",
                fnam=gridfilename + "BP.nc",
                description=desc,
                history=hist,
                source=source,
                debug=debug,
            )

    if (Nj_SO != 0) and ("so" in grids or "all" in grids):
        hasSO=True
        ###
        ###Southern Ocean grid
        ###
        lamSO, phiSO = generate_latlon_grid(
            Ni,
            Nj_SO,
            lon0,
            lenlon,
            lat0_SO,
            lenlat_SO,
            ensure_nj_even=ensure_nj_even,
        )
        dxSO = -np.ones([lamSO.shape[0], lamSO.shape[1] - 1])
        dySO = -np.ones([lamSO.shape[0] - 1, lamSO.shape[1]])
        areaSO = -np.ones([lamSO.shape[0] - 1, lamSO.shape[1] - 1])
        if calculate_metrics:
            # For spherical grids we can safely use the MIDAS algorithm for calculating the metrics
            dxSO, dySO, areaSO = generate_grid_metrics_MIDAS(lamSO, phiSO)
        angleSO = angle_x(lamSO, phiSO)
        print(
            "   CHECK_metrics_MIDAS: % errors in (area, lat arc, lon arc)",
            metrics_error(dxSO, dySO, areaSO, Ni, phiSO[0, 0], phiSO[-1, 0]),
        )

        deltaPhiSO = phiSO[1, Ni // 4] - phiSO[0, Ni // 4]
        lat0_SC = phiSO[0, Ni // 4] - deltaPhiSO
        # The above heuristics produce a displaced pole grid with a nominal resolution
        # different from the rest of the grid!
        # To get the nominal resolution right  we must instead make the resolution continuous across the joint
        # fullArc = lat0_SC+90.
        # if(match_dy):
        #    Nj_scap = int(fullArc/deltaPhiSO)

        if write_subgrid_files:
            write_nc(
                lamSO,
                phiSO,
                dxSO,
                dySO,
                areaSO,
                angleSO,
                axis_units="degrees",
                fnam=gridfilename + "SO.nc",
                description=desc,
                history=hist,
                source=source,
                debug=debug,
            )

    if (Nj_scap != 0) and ("sc" in grids or "all" in grids):
        ###
        ###Southern cap
        ###
        hasSC=True
        if r_dp == 0.0 and lat_dp < -90.0:
            fullArc = lat0_SC + 90.0
            Nj_scap = int(fullArc / deltaPhiSO)
            lamSC, phiSC = generate_latlon_grid(
                Ni,
                Nj_scap,
                lon0,
                lenlon,
                -90.0,
                90 + lat0_SO,
                ensure_nj_even=ensure_nj_even,
            )
            angleSC = angle_x(lamSC, phiSC)
            dxSC = -np.ones([lamSC.shape[0], lamSC.shape[1] - 1])
            dySC = -np.ones([lamSC.shape[0] - 1, lamSC.shape[1]])
            areaSC = -np.ones([lamSC.shape[0] - 1, lamSC.shape[1] - 1])
            if calculate_metrics:
                # For spherical grids we can safely use the MIDAS algorithm for calculating the metrics
                dxSC, dySC, areaSC = generate_grid_metrics_MIDAS(lamSC, phiSC)
                print(
                    "   CHECK_metrics_MIDAS: % errors in (area, lat arc, lon arc)",
                    metrics_error(
                        dxSC, dySC, areaSC, Ni, phiSC[-1, 0], phiSC[0, 0]
                    ),
                )
        else:
            if match_dy:
                # The SC grid should end at the same lattitude that SO starts.
                # Then when we combine the two grids we should drop the x,y,dx,angle from one of the two.
                # This way we get a continous dy and area.
                lat0_SC = lat0_SO

            if lat_dp > -90:
                r_dp = np.tan((90 + lat_dp) * PI_180) / np.tan(
                    (90 + lat0_SC) * PI_180
                )

            lamSC, phiSC, londp, latdp = generate_displaced_pole_grid(
                Ni, Nj_scap, lon0, lat0_SC, lon_dp, r_dp
            )
            angleSC = angle_x(lamSC, phiSC)

            # Quadrature metrics using great circle approximations for the h's
            dxSC = -np.ones([lamSC.shape[0], lamSC.shape[1] - 1])
            dySC = -np.ones([lamSC.shape[0] - 1, lamSC.shape[1]])
            areaSC = -np.ones([lamSC.shape[0] - 1, lamSC.shape[1] - 1])
            if calculate_metrics:
                dxSC, dySC, areaSC = displacedPoleCap_metrics_quad(
                    4, Ni, Nj_scap, lon0, lat0_SC, lon_dp, r_dp
                )
                poles_i = int(Ni * np.mod(lon_dp - lon0, 360) / 360.0)
                print(
                    "   CHECK_metrics_hquad: % errors in (area, lat arc, lon arc)",
                    metrics_error(
                        dxSC,
                        dySC,
                        areaSC,
                        Ni,
                        lat1=lat0_SC,
                        lat2=-90.0,
                        displaced_pole=poles_i,
                        excluded_fraction=doughnut,
                    ),
                )
            # Cut the unused portion of the grid
            # Choose the doughnut factor to keep the number of j's the same as in existing OM4p25 grid
            if doughnut != 0.0:
                jmin = np.ceil(doughnut * Nj_scap)
                jmin = jmin + np.mod(jmin, 2)
                jmint = int(jmin)
                lamSC = lamSC[jmint:, :]
                phiSC = phiSC[jmint:, :]
                dxSC = dxSC[jmint:, :]
                dySC = dySC[jmint:, :]
                areaSC = areaSC[jmint:, :]
                angleSC = angleSC[jmint:, :]

            if phiSC.shape[0] % 2 == 0 and ensure_nj_even:
                print(
                    "   The number of j's is not even. Fixing this by cutting one row at south."
                )
                lamSC = np.delete(lamSC, 0, 0)
                phiSC = np.delete(phiSC, 0, 0)
                dxSC = np.delete(dxSC, 0, 0)
                dySC = np.delete(dySC, 0, 0)
                areaSC = np.delete(areaSC, 0, 0)
                angleSC = np.delete(angleSC, 0, 0)

            print("   number of js=", lamSC.shape[0])

        if grids == 'sc':  # if only "sc" was requested cut it according to args
            # Cut the grid at south according to the options!
            # We may need to cut the whole SC grid and some of the SO
            cut = False
            jcut = 0
            cats = 0
            if south_cutoff_row > 0:
                cut = True
                jcut = south_cutoff_row - 1
            elif south_cutoff_ang > -90:
                cut = True
                jcut = 1 + np.nonzero(phiSC[:, 0] < south_cutoff_ang)[0][-1]

            if cut:
                print("   SC: shape[0], jcut", lamSC.shape[0], jcut)
                if jcut < lamSC.shape[0]:  # only SC needs to be cut
                    if (phiSC.shape[0] - jcut) % 2 == 0 and ensure_nj_even:
                        # if((areaSC.shape[0]-jcut-1)%2 == 0 and ensure_nj_even):
                        print(
                            "   SC: The number of j's is not even. Fixing this by cutting one row at south."
                        )
                        jcut = jcut + 1
                    print("   Cutting SC grid rows 0 to ", jcut)
                    lamSC = lamSC[jcut:, :]
                    phiSC = phiSC[jcut:, :]
                    dxSC = dxSC[jcut:, :]
                    dySC = dySC[jcut:, :]
                    areaSC = areaSC[jcut:, :]
                    angleSC = angleSC[jcut:, :]

        if write_subgrid_files:
            write_nc(
                lamSC,
                phiSC,
                dxSC,
                dySC,
                areaSC,
                angleSC,
                axis_units="degrees",
                fnam=gridfilename + "SC.nc",
                description=desc,
                history=hist,
                source=source,
                debug=debug,
            )
        if plotem:
            ax = displacedPoleCap_plot(lamSC,phiSC,lon0,lon_dp,lat0_SC,stride=int(refineR * 10),block=True,dplat=lat_dp)

            if "so" in grids or "all" in grids:
                plot_mesh_in_latlon(
                    lamSO, phiSO, stride=int(refineR * 10), newfig=False, axis=ax
                )

    # Concatenate to generate the whole grid
    stitch = False
    if(hasSC and hasSO): 
        stitch=True
        print("hasSC and hasSO")
    if(hasSO and hasMerc): 
        stitch=True
        print("hasMerc and hasSO")
    if(hasBP and hasMerc): 
        stitch=True
        print("hasMerc and hasBP")
    if stitch:
        # Start from displaced southern cap and join the southern ocean grid
        print("Stitching the grids together...")

        # Note that x,y,dx,angle_dx have a shape[0]=nyp1 and should be cut by one (total 3) for the merged variables
        # to have the right shape.
        # But y and area have a shape[0]=ny and should not be cut.
        # Niki: This cut can be done in a few ambigous ways:
        #    1.  MIDAS way (above)
        #    2.  this way  : x1=np.concatenate((lamSC[:-1,:],lamSO),axis=0)
        #                    y1=np.concatenate((phiSC[:-1,:],phiSO),axis=0)
        #                    dx1=np.concatenate((dxSC[:-1,:],dxSO),axis=0)
        #                    angle1=np.concatenate((angleSC[:-1,:],angleSO),axis=0)
        #                    #
        #                    dy1=np.concatenate((dySC,dySO),axis=0)
        #                    area1=np.concatenate((areaSC,areaSO),axis=0)
        #     3.  at the very end by restricting to x3[2:,:], ...
        #      Which way is "correct"?
        #      If the sub-grids are disjoint, 1 and 2 introduce a jump in y1 at the joint,
        #      as a result y1 and dy1 may become inconsistent?
        #
        # Cut the grid at south according to the options!
        # We may need to cut the whole SC grid and some of the SO
        cut = False
        jcut = 0
        cats = 0
        if south_cutoff_row > 0:
            cut = True
            jcut = south_cutoff_row - 1
        elif south_cutoff_ang > -90:
            cut = True
            jcut = 1 + np.nonzero(phiSC[:, 0] < south_cutoff_ang)[0][-1]

        if cut:
            if hasSC and jcut < lamSC.shape[0]:  # only SC needs to be cut
                print("   SC: shape[0], jcut", lamSC.shape[0], jcut)
                if (phiSC.shape[0] - jcut) % 2 == 0 and ensure_nj_even:
                    # if((areaSC.shape[0]-jcut-1)%2 == 0 and ensure_nj_even):
                    print(
                        "   SC: The number of j's is not even. Fixing this by cutting one row at south."
                    )
                    jcut = jcut + 1
                print("   Cutting SC grid rows 0 to ", jcut)
                lamSC = lamSC[jcut:, :]
                phiSC = phiSC[jcut:, :]
                dxSC = dxSC[jcut:, :]
                dySC = dySC[jcut:, :]
                areaSC = areaSC[jcut:, :]
                angleSC = angleSC[jcut:, :]
            elif hasSO:
                print("   Whole SC and some of SO need to be cut!")
                print("   SO: shape[0], jcut", lamSO.shape[0], jcut)
                hasSC = False
                jcut_SO = jcut - lamSC.shape[0]
                #                    jcut_SO = max(jcut-lamSC.shape[0], 1 + np.nonzero(phiSO[:,0] < south_cutoff_ang)[0][-1])
                if (areaSO.shape[0] - jcut_SO - 1) % 2 == 0 and ensure_nj_even:
                    print(
                        "   SO: The number of j's is not even. Fixing this by cutting one row at south."
                    )
                    jcut_SO = jcut_SO + 1
                print("   No SC grid remained. Cutting SO grid rows 0 to ", jcut_SO)
                lamSO = lamSO[jcut_SO:, :]
                phiSO = phiSO[jcut_SO:, :]
                dxSO = dxSO[jcut_SO:, :]
                dySO = dySO[jcut_SO:, :]
                areaSO = areaSO[jcut_SO:, :]
                angleSO = angleSO[jcut_SO:, :]

        if match_dy or not "all" in grids:
            if hasSC and hasSO:
                x1 = np.concatenate((lamSC[:-1, :], lamSO), axis=0)
                y1 = np.concatenate((phiSC[:-1, :], phiSO), axis=0)
                dx1 = np.concatenate((dxSC[:-1, :], dxSO), axis=0)
                angle1 = np.concatenate((angleSC[:-1, :], angleSO), axis=0)
                #
                dy1 = np.concatenate((dySC, dySO), axis=0)
                area1 = np.concatenate((areaSC, areaSO), axis=0)
                cats = cats + 1
            elif hasSO:  # if the whole SC was cut
                x1 = lamSO
                y1 = phiSO
                dx1 = dxSO
                dy1 = dySO
                area1 = areaSO
                angle1 = angleSO

            # Join the Mercator grid
            if hasSO and hasMerc:
                x2 = np.concatenate((x1[:-1, :], lamMerc), axis=0)
                y2 = np.concatenate((y1[:-1, :], phiMerc), axis=0)
                dx2 = np.concatenate((dx1[:-1, :], dxMerc), axis=0)
                angle2 = np.concatenate((angle1[:-1, :], angleMerc), axis=0)
                #
                dy2 = np.concatenate((dy1, dyMerc), axis=0)
                area2 = np.concatenate((area1, areaMerc), axis=0)
            elif hasMerc:
                x2 = lamMerc
                y2 = phiMerc
                dx2 = dxMerc
                dy2 = dyMerc
                angle2 = angleMerc
                area2 = areaMerc
            else:
                x2 = x1
                y2 = y1
                dx2 = dx1
                dy2 = dy1
                angle2 = angle1
                area2 = area1

            cats = cats + 1
            # Join the norhern bipolar cap grid
            if hasBP:
                x3 = np.concatenate((x2[:-1, :], lamBP), axis=0)
                y3 = np.concatenate((y2[:-1, :], phiBP), axis=0)
                dx3 = np.concatenate((dx2[:-1, :], dxBP), axis=0)
                angle3 = np.concatenate((angle2[:-1, :], angleBP), axis=0)
                #
                dy3 = np.concatenate((dy2, dyBP), axis=0)
                area3 = np.concatenate((area2, areaBP), axis=0)
                dy3_ = np.roll(y3[:, Ni // 4], shift=-1, axis=0) - y3[:, Ni // 4]
                if np.any(dy3_ == 0):
                    raise Exception(
                        "lattitude array has repeated values along symmetry meridian!"
                    )
            else:
                x3 = x2
                y3 = y2
                dx3 = dx2
                dy3 = dy2
                angle3 = angle2
                area3 = area2

        else:
            # Note: angle variable has the same shape as x1,y1,dx1 and should be treated just like them.
            #      I think the way is treated below unlike them is a bug, but fixing the bug will change the produced grids.
            if hasSC and hasSO:
                x1 = np.concatenate((lamSC, lamSO[1:, :]), axis=0)
                y1 = np.concatenate((phiSC, phiSO[1:, :]), axis=0)
                dx1 = np.concatenate((dxSC, dxSO[1:, :]), axis=0)
                dy1 = np.concatenate((dySC, dySO), axis=0)
                area1 = np.concatenate((areaSC, areaSO), axis=0)
                angle1 = np.concatenate((angleSC[:-1, :], angleSO[:-1, :]), axis=0)
                # Join the Mercator grid
                x2 = np.concatenate((x1, lamMerc[1:, :]), axis=0)
                y2 = np.concatenate((y1, phiMerc[1:, :]), axis=0)
                dx2 = np.concatenate((dx1, dxMerc[1:, :]), axis=0)
                dy2 = np.concatenate((dy1, dyMerc), axis=0)
                area2 = np.concatenate((area1, areaMerc), axis=0)
                angle2 = np.concatenate((angle1, angleMerc[:-1, :]), axis=0)
            else:
                x2 = lamMerc
                y2 = phiMerc
                dx2 = dxMerc
                dy2 = dyMerc
                angle2 = angleMerc[:-1, :]
                area2 = areaMerc
            # Join the norhern bipolar cap grid
            x3 = np.concatenate((x2, lamBP[1:, :]), axis=0)
            y3 = np.concatenate((y2, phiBP[1:, :]), axis=0)
            dx3 = np.concatenate((dx2, dxBP[1:, :]), axis=0)
            dy3 = np.concatenate((dy2, dyBP), axis=0)
            area3 = np.concatenate((area2, areaBP), axis=0)
            angle3 = np.concatenate((angle2, angleBP), axis=0)

        dy3_ = np.roll(y3[:, Ni // 4], shift=-1, axis=0) - y3[:, Ni // 4]
        if np.any(dy3_ == 0):
            print(
                "WARNING: lattitude array has repeated values along symmetry meridian! Try option --match_dy"
            )

        if write_subgrid_files:
            if hasSC:
                write_nc(
                    lamSC,
                    phiSC,
                    dxSC,
                    dySC,
                    areaSC,
                    angleSC,
                    axis_units="degrees",
                    fnam=gridfilename + "SC.nc",
                    description=desc,
                    history=hist,
                    source=source,
                    debug=debug,
                )
            elif Nj_scap != 0:
                print(
                    "There remained no South Pole cap grid because of the number of rows cut= ",
                    jcut,
                    lamSC.shape[0],
                )

        # write the whole grid file
        desc = desc + "It consists of; "
        if hasMerc:
            desc = (
                desc
                + "a Mercator grid spanning "
                + str(phiMerc[0, 0])
                + " to "
                + str(phiMerc[-1, 0])
                + " degrees; "
            )
        if hasBP:
            desc = (
                desc
                + "a bipolar northern cap north of "
                + str(phiMerc[-1, 0])
                + " degrees; "
            )
        if hasSO:
            desc = (
                desc
                + "a regular lat-lon grid spanning "
                + str(latUp_SO)
                + " to "
                + str(lat0_SO)
                + " degrees; "
            )
        if hasSC:
            desc = desc + "a "
            if r_dp != 0.0:
                desc = desc + "displaced pole "
            else:
                desc = desc + "regular "
            desc = desc + "southern cap south of " + str(lat0_SO) + " degrees."

        if south_cutoff_ang > -90:
            desc = desc + " It is cut south of " + str(south_cutoff_ang) + " degrees."

        if south_cutoff_row > 0:
            desc = (
                desc
                + " The first "
                + str(south_cutoff_row)
                + " rows at south are deleted."
            )

        # Ensure that the equator (y=0) is still a u-point
        equator = 0.0
        equator_index = np.searchsorted(y3[:, Ni // 4], equator)
        if equator_index == 0:
            raise Exception("   Ooops: Equator is not in the grid")
        else:
            print("   Equator is at j=", equator_index)
        # Ensure that the equator (y=0) is a u-point
        if equator_index % 2 == 0:
            raise Exception(
                "Ooops: Equator is not going to be a u-point. Use option --south_cutoff_row to one more or on less row from south."
            )
        if y3.shape[0] % 2 == 0:
            raise Exception(
                "Ooops: The number of j's in the supergrid is not even. Use option --south_cutoff_row to one more or on less row from south."
            )

        print(
            "shapes: ",
            x3.shape,
            y3.shape,
            dx3.shape,
            dy3.shape,
            area3.shape,
            angle3.shape,
        )
        write_nc(
            x3,
            y3,
            dx3,
            dy3,
            area3,
            angle3,
            axis_units="degrees",
            fnam=gridfilename,
            description=desc,
            history=hist,
            source=source,
            no_changing_meta=no_changing_meta,
            debug=debug,
        )

        print("Wrote the whole grid to file ", gridfilename)

        # Visualization
        if plotem:
            plot_mesh_in_xyz(
                x2, y2, stride=30, upperlat=-40, title="Grid south of -40 degrees"
            )
            plot_mesh_in_xyz(
                x3, y3, stride=30, lowerlat=40, title="Grid north of 40 degrees"
            )

    print("runtime(secs)  %s" % (time.time() - start_time))
    print("FINISHED.")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="create ocean hgrid")

    parser.add_argument(
        "-r",
        "--inverse_resolution",
        type=float,
        required=True,
        help="inverse of the horizontal resolution (e.g. 4 for 1/4 degree)",
    )

    parser.add_argument(
        "-f",
        "--gridfilename",
        type=str,
        required=False,
        default="ocean_hgrid.nc",
        help="name for output grid file",
    )

    parser.add_argument(
        "--r_dp",
        type=float,
        required=False,
        default=0.0,
        help="displacement factor/0.2",
    )

    parser.add_argument(
        "--lon_dp",
        type=float,
        required=False,
        default=80.0,
        help="",
    )

    parser.add_argument(
        "--lat_dp",
        type=float,
        required=False,
        default=-99.0,
        help="",
    )

    parser.add_argument(
        "--exfracdp",
        type=float,
        required=False,
        default=None,
        help="",
    )

    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help="debug mode",
    )

    parser.add_argument(
        "--south_cutoff_ang",
        type=float,
        required=False,
        default=-90.0,
        help="degrees south to start",
    )

    parser.add_argument(
        "--south_cutoff_row",
        type=int,
        required=False,
        default=0,
        help="rows south to cut",
    )

    parser.add_argument(
        "--south_cap_lat",
        type=float,
        required=False,
        default=-99.0,
        help="",
    )

    parser.add_argument(
        "--south_ocean_upper_lat",
        type=float,
        required=False,
        default=-99.0,
        help="",
    )

    parser.add_argument(
        "--no_south_cap",
        action="store_true",
        help="",
    )

    parser.add_argument(
        "--match_dy",
        action="store_true",
        help="",
    )

    parser.add_argument(
        "--ensure_nj_even",
        action="store_true",
        help="",
    )

    parser.add_argument(
        "--plotem",
        action="store_true",
        help="",
    )

    parser.add_argument(
        "--skip_metrics",
        action="store_true",
        help="",
    )

    parser.add_argument(
        "--write_subgrid_files",
        action="store_true",
        help="",
    )

    parser.add_argument(
        "--no_changing_meta",
        action="store_true",
        help="",
    )

    parser.add_argument(
        "--enhanced_equatorial",
        type=int,
        required=False,
        default=0,
        help="",
    )

    parser.add_argument(
        "--shift_equator_to_u_point",
        action="store_false",
        help="",
    )

    parser.add_argument(
        "--grids",
        type=str,
        nargs="+",
        required=False,
        default="all",
        help="choices are bipolar, mercator, so, sc, all. Default is all",
    )

    args = vars(parser.parse_args())

    main(**args)
