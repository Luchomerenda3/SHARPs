import numpy as np


def img2heliovec(bxImg, byImg, bzImg, lon, lat, lonc, latc, pAng):
    """
    Perform transformation of a vector from image location (lon, lat) to
    heliographic center. The formula is from Hagyard (1987), and further
    developed by Gary & Hagyard (1990).
    Copied/translated from SSWIDL/SDO/HMI/IDL code by Xudong Sun (xudongs@hawaii.edu).

    Arguments:
        bxImg, byImg, bzImg: Three components of vector magnetic field on image
                             coordinates (scalar or numpy array).
        lon, lat:            Heliographic coordinates of the location where the
                             vector field was measured. They are in radians.
        lonc, latc:          Heliographic coordinates of the image disk center.
                             They are in radians.
        pAng:                Position angle of the heliographic north pole, measured
                             eastward from the north. It's in radians.

    Returns:
        bxHelio, byHelio, bzHelio: Transformed magnetic field components.
    """
    a11 = -np.sin(latc) * np.sin(pAng) * np.sin(lon - lonc) + np.cos(pAng) * np.cos(lon - lonc)
    a12 = np.sin(latc) * np.cos(pAng) * np.sin(lon - lonc) + np.sin(pAng) * np.cos(lon - lonc)
    a13 = -np.cos(latc) * np.sin(lon - lonc)

    a21 = -np.sin(lat) * (np.sin(latc) * np.sin(pAng) * np.cos(lon - lonc) + np.cos(pAng) * np.sin(lon - lonc)) - np.cos(lat) * np.cos(latc) * np.sin(pAng)
    a22 = np.sin(lat) * (np.sin(latc) * np.cos(pAng) * np.cos(lon - lonc) - np.sin(pAng) * np.sin(lon - lonc)) + np.cos(lat) * np.cos(latc) * np.cos(pAng)
    a23 = -np.cos(latc) * np.sin(lat) * np.cos(lon - lonc) + np.sin(latc) * np.cos(lat)

    a31 = np.cos(lat) * (np.sin(latc) * np.sin(pAng) * np.cos(lon - lonc) + np.cos(pAng) * np.sin(lon - lonc)) - np.sin(lat) * np.cos(latc) * np.sin(pAng)
    a32 = -np.cos(lat) * (np.sin(latc) * np.cos(pAng) * np.cos(lon - lonc) - np.sin(pAng) * np.sin(lon - lonc)) + np.sin(lat) * np.cos(latc) * np.cos(pAng)
    a33 = np.cos(lat) * np.cos(latc) * np.cos(lon - lonc) + np.sin(lat) * np.sin(latc)

    bxHelio = a11 * bxImg + a12 * byImg + a13 * bzImg
    byHelio = a21 * bxImg + a22 * byImg + a23 * bzImg
    bzHelio = a31 * bxImg + a32 * byImg + a33 * bzImg

    return bxHelio, byHelio, bzHelio
