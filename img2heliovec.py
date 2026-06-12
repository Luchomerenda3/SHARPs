import numpy as np

def img2heliovec(bxImg, byImg, bzImg, lon, lat, lonc, latc, pAng):
    # Convert angles from degrees to radians for computation
    lon, lat, lonc, latc, pAng = np.radians([lon, lat, lonc, latc, pAng])

    # Compute the transformation matrix components
    a11 = -np.sin(latc) * np.sin(pAng) * np.sin(lon - lonc) + np.cos(pAng) * np.cos(lon - lonc)
    a12 = np.sin(latc) * np.cos(pAng) * np.sin(lon - lonc) + np.sin(pAng) * np.cos(lon - lonc)
    a13 = -np.cos(latc) * np.sin(lon - lonc)
    a21 = -np.sin(lat) * (np.sin(latc) * np.sin(pAng) * np.cos(lon - lonc) + np.cos(pAng) * np.sin(lon - lonc)) - np.cos(lat) * np.cos(latc) * np.sin(pAng)
    a22 = np.sin(lat) * (np.sin(latc) * np.cos(pAng) * np.cos(lon - lonc) - np.sin(pAng) * np.sin(lon - lonc)) + np.cos(lat) * np.cos(latc) * np.cos(pAng)
    a23 = -np.cos(latc) * np.sin(lat) * np.cos(lon - lonc) + np.sin(latc) * np.cos(lat)
    a31 = np.cos(lat) * (np.sin(latc) * np.sin(pAng) * np.cos(lon - lonc) + np.cos(pAng) * np.sin(lon - lonc)) - np.sin(lat) * np.cos(latc) * np.sin(pAng)
    a32 = -np.cos(lat) * (np.sin(latc) * np.cos(pAng) * np.cos(lon - lonc) - np.sin(pAng) * np.sin(lon - lonc)) + np.sin(lat) * np.cos(latc) * np.cos(pAng)
    a33 = np.cos(lat) * np.cos(latc) * np.cos(lon - lonc) + np.sin(lat) * np.sin(latc)

    # Apply the transformation to the vector components
    bxHelio = a11 * bxImg + a12 * byImg + a13 * bzImg
    byHelio = a21 * bxImg + a22 * byImg + a23 * bzImg
    bzHelio = a31 * bxImg + a32 * byImg + a33 * bzImg

    return bxHelio, byHelio, bzHelio
