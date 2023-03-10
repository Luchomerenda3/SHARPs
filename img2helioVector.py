import numpy as np


def img2helioVector(bxImg, byImg, bzImg, lon, lat, lonc, latc, pAng):

    """
    Perform transformation of a vector from image location (lon, lat) to
    heliographic center. The formula is from Hagyard (1987), and further
    developed by Gary & Hagyard (1990).

    Arguments:
    bxImg, byImg, bzImg: three components of vector magnetic field on image
                         coordinates.
    lon, lat:            heliographic coordinates of the location where the vector field
                         measured. They are in radians.
    lonc, latc:          heliographic coordinates of the image disk center. They are in
                         radians.
    pAng:                position angle of the heliographic north pole, measured eastward
                         from the north. It's in radians.
    """

    raddeg = np.pi / 180.
    a11 = -np.sin(latc) * np.sin(pAng) * np.sin(lon - lonc) + np.cos(pAng) * np.cos(
        lon - lonc)
    a12 = np.sin(latc) * np.cos(pAng) * np.sin(lon - lonc) + np.sin(pAng) * np.cos(
        lon - lonc)
    a13 = -np.cos(latc) * np.sin(lon - lonc)
    a21 = -np.sin(lat) * (np.sin(latc) * np.sin(pAng) * np.cos(lon - lonc) + np.cos(
        pAng) * np.sin(lon - lonc)) - np.cos(lat) * np.cos(latc) * np.sin(pAng)
    a22 = np.sin(lat) * (np.sin(latc) * np.cos(pAng) * np.cos(lon - lonc) - np.sin(
        pAng) * np.sin(lon - lonc)) + np.cos(lat) * np.cos(latc) * np.cos(pAng)
    a23 = -np.cos(latc) * np.sin(lat) * np.cos(lon - lonc) + np.sin(latc) * np.cos(lat)
    a31 = np.cos(lat) * (np.sin(latc) * np.sin(pAng) * np.cos(lon - lonc) + np.cos(
        pAng) * np.sin(lon - lonc)) - np.sin(lat) * np.cos(latc) * np.sin(pAng)
    a32 = -np.cos(lat) * (np.sin(latc) * np.cos(pAng) * np.cos(lon - lonc) - np.sin(
        pAng) * np.sin(lon - lonc)) + np.sin(lat) * np.cos(latc) * np.cos(pAng)
    a33 = np.cos(lat) * np.cos(latc) * np.cos(lon - lonc) + np.sin(lat) * np.sin(latc)

    A = np.array([[a11, a12, a13], [a21, a22, a23], [a31, a32, a33]])
    B = np.array([bxImg, byImg, bzImg])

    H = np.dot(A, B)
    bxHelio, byHelio, bzHelio = H[0], H[1], H[2]

    return bxHelio, byHelio, bzHelio
