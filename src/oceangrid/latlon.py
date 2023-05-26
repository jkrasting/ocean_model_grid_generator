#!/usr/bin/env python

from __future__ import print_function

import numpypi.numpypi_series as np

def generate_latlon_grid(
    lni, lnj, llon0, llen_lon, llat0, llen_lat, ensure_nj_even=True
):
    print("Generating regular lat-lon grid between latitudes ", llat0, llat0 + llen_lat)
    llonSP = llon0 + np.arange(lni + 1) * llen_lon / float(lni)
    llatSP = llat0 + np.arange(lnj + 1) * llen_lat / float(lnj)
    if llatSP.shape[0] % 2 == 0 and ensure_nj_even:
        print(
            "   The number of j's is not even. Fixing this by cutting one row at south."
        )
        llatSP = np.delete(llatSP, 0, 0)

    llamSP = np.tile(llonSP, (llatSP.shape[0], 1))
    lphiSP = np.tile(llatSP.reshape((llatSP.shape[0], 1)), (1, llonSP.shape[0]))

    print(
        "   generated regular lat-lon grid between latitudes ",
        lphiSP[0, 0],
        lphiSP[-1, 0],
    )
    print("   number of js=", lphiSP.shape[0])

    #    h_i_inv=llen_lon*PI_180*np.cos(lphiSP*PI_180)/lni
    #    h_j_inv=llen_lat*PI_180*np.ones(lphiSP.shape)/lnj
    #    delsin_j = np.roll(np.sin(lphiSP*PI_180),shift=-1,axis=0) - np.sin(lphiSP*PI_180)
    #    dx_h=h_i_inv[:,:-1]*_default_Re
    #    dy_h=h_j_inv[:-1,:]*_default_Re
    #    area=delsin_j[:-1,:-1]*_default_Re*_default_Re*llen_lon*PI_180/lni

    return llamSP, lphiSP

