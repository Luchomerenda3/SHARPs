"""
cartography.py <--- cartography.c / SSWIDL cartography routines

Functions for mapping between plate, heliographic, and various map coordinate systems.
Copied/translated from SSWIDL/SDO/HMI/IDL code by Xudong Sun (xudongs@hawaii.edu)
and Rick Bogart (RBogart@solar.Stanford.EDU).
"""

import numpy as np

RECTANGULAR = 0
CASSINI = 1
MERCATOR = 2
CYLEQA = 3
SINEQA = 4
GNOMONIC = 5
POSTEL = 6
STEREOGRAPHIC = 7
ORTHOGRAPHIC = 8
LAMBERT = 9


def arc_distance(lat, lon, latc, lonc):
    cosa = np.sin(lat) * np.sin(latc) + np.cos(lat) * np.cos(latc) * np.cos(lon - lonc)
    return np.arccos(np.clip(cosa, -1.0, 1.0))


def img2sphere(x, y, ang_r, latc, lonc, pa):
    """
    Map projected coordinates (x, y) to (lon, lat) and (rho | sig, chi).
    Copied/translated from SSWIDL/SDO/HMI/IDL code by Xudong Sun (xudongs@hawaii.edu).
    """
    sinang_r = np.sin(ang_r)
    tanang_r = np.tan(ang_r)
    sinlatc = np.sin(latc)
    coslatc = np.cos(latc)

    chi = np.arctan2(x, y) + pa
    while np.any(chi > 2 * np.pi):
        chi = np.where(chi > 2 * np.pi, chi - 2 * np.pi, chi)
    while np.any(chi < 0):
        chi = np.where(chi < 0, chi + 2 * np.pi, chi)

    sig = np.arctan(np.hypot(x, y) * tanang_r)
    sinsig = np.sin(sig)
    rho = np.arcsin(sinsig / sinang_r) - sig

    if np.any(sig > ang_r):
        if np.isscalar(sig):
            return -1

    mu = np.cos(rho + sig)
    sinr = np.sin(rho)
    cosr = np.cos(rho)

    sinlat = sinlatc * cosr + coslatc * sinr * np.cos(chi)
    coslat = np.sqrt(np.maximum(0.0, 1.0 - sinlat * sinlat))
    lat = np.arcsin(np.clip(sinlat, -1.0, 1.0))
    sinlon = np.where(coslat == 0.0, 0.0, sinr * np.sin(chi) / np.where(coslat == 0.0, 1.0, coslat))
    lon = np.arcsin(np.clip(sinlon, -1.0, 1.0))

    lon = np.where(cosr < (sinlat * sinlatc), np.pi - lon, lon)
    lon = lon + lonc

    while np.any(lon < 0):
        lon = np.where(lon < 0, lon + 2 * np.pi, lon)
    while np.any(lon >= 2 * np.pi):
        lon = np.where(lon >= 2 * np.pi, lon - 2 * np.pi, lon)

    return rho, lat, lon, sinlat, coslat, sig, mu, chi


def plane2sphere(x, y, latc, lonc, projection=CYLEQA, return_status=False):
    """
    Perform the inverse mapping from rectangular coordinates x, y on a map
    in a particular projection to heliographic (or geographic) coordinates
    latitude and longitude (in radians).

    Compatible with both cartography.c and SDO/HMI plane2sphere.pro.
    Copied/translated from SSWIDL/SDO/HMI/IDL code by Xudong Sun (xudongs@hawaii.edu).

    Arguments:
        x, y: Map coordinates in radians (scalar or numpy array).
        latc: Latitude of map center in radians.
        lonc: Longitude of map center in radians.
        projection: Map projection code or name (default: CYLEQA = 3).
        return_status: If True, returns (lat, lon, status). If False, returns (lat, lon).
    """
    is_scalar = np.isscalar(x) and np.isscalar(y)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    coslatc = np.cos(latc)
    sinlatc = np.sin(latc)

    if isinstance(projection, str):
        proj_map = {
            "RECTANGULAR": RECTANGULAR, "CASSINI": CASSINI, "MERCATOR": MERCATOR,
            "CYLEQA": CYLEQA, "SINEQA": SINEQA, "GNOMONIC": GNOMONIC,
            "POSTEL": POSTEL, "STEREOGRAPHIC": STEREOGRAPHIC,
            "ORTHOGRAPHIC": ORTHOGRAPHIC, "LAMBERT": LAMBERT
        }
        proj_code = proj_map.get(projection.upper(), -1)
    else:
        proj_code = int(projection)

    if proj_code == CYLEQA:
        invalid = np.abs(y) > 1.0
        y_safe = np.where(invalid, np.copysign(1.0, y), y)\

        cosphi = np.sqrt(np.maximum(0.0, 1.0 - y_safe * y_safe))
        lat_arg = np.clip(y_safe * coslatc + cosphi * np.cos(x) * sinlatc, -1.0, 1.0)
        lat = np.arcsin(lat_arg)

        cos_lat = np.cos(lat)
        safe_cos_lat = np.where(cos_lat == 0.0, 1.0, cos_lat)
        test = np.where(cos_lat == 0.0, 0.0, cosphi * np.sin(x) / safe_cos_lat)
        test = np.clip(test, -1.0, 1.0)
        lon = np.arcsin(test) + lonc

        # Branch cut adjustment for |x| > pi/2
        x0 = np.copy(x)
        mask_pos = x0 > (np.pi / 2.0)
        while np.any(mask_pos):
            lon = np.where(mask_pos, np.pi - lon, lon)
            x0 = np.where(mask_pos, x0 - np.pi, x0)
            mask_pos = x0 > (np.pi / 2.0)

        mask_neg = x0 < (-np.pi / 2.0)
        while np.any(mask_neg):
            lon = np.where(mask_neg, -np.pi - lon, lon)
            x0 = np.where(mask_neg, x0 + np.pi, x0)
            mask_neg = x0 < (-np.pi / 2.0)

        # Points outside the cylinder (|y| > 1) are NaN in plane2sphere.pro
        lat = np.where(invalid, np.nan, lat)
        lon = np.where(invalid, np.nan, lon)

        if return_status:
            status = np.where(invalid, -1, 0)
            status = np.where(arc_distance(lat, lon, latc, lonc) > (np.pi / 2.0), 1, status)
            if is_scalar:
                return float(lat), float(lon), int(status)
            return lat, lon, status

        if is_scalar:
            return float(lat), float(lon)
        return lat, lon

    elif proj_code == RECTANGULAR:
        lon = lonc + x
        lat = latc + y
        if return_status:
            status = np.where(arc_distance(lat, lon, latc, lonc) > (np.pi / 2.0), 1, 0)
            status = np.where((np.abs(x) > np.pi) | (np.abs(y) > (np.pi / 2.0)), -1, status)
            if is_scalar:
                return float(lat), float(lon), int(status)
            return lat, lon, status
        if is_scalar:
            return float(lat), float(lon)
        return lat, lon

    elif proj_code == CASSINI:
        sinx = np.sin(x)
        cosy = np.cos(y + latc)
        siny = np.sin(y + latc)
        lat = np.arccos(np.sqrt(np.clip(cosy * cosy + siny * siny * sinx * sinx, 0.0, 1.0)))
        lat = np.where(y < -latc, -lat, lat)
        cos_lat = np.cos(lat)
        safe_cos = np.where(np.abs(lat) < (np.pi / 2.0), np.where(cos_lat == 0.0, 1.0, cos_lat), 1.0)
        lon = np.where(np.abs(lat) < (np.pi / 2.0), lonc + np.arcsin(np.clip(sinx / safe_cos, -1.0, 1.0)), lonc)
        cond_branch = (y > (np.pi / 2.0 - latc)) | (y < (-np.pi / 2.0 - latc))
        lon = np.where(cond_branch, 2 * lonc + np.pi - lon, lon)
        lon = np.where(lon < -np.pi, lon + 2 * np.pi, lon)
        lon = np.where(lon > np.pi, lon - 2 * np.pi, lon)
        if return_status:
            status = np.where(arc_distance(lat, lon, latc, lonc) > (np.pi / 2.0), 1, 0)
            status = np.where((np.abs(x) > np.pi) | (np.abs(y) > (np.pi / 2.0)), -1, status)
            if is_scalar:
                return float(lat), float(lon), int(status)
            return lat, lon, status
        if is_scalar:
            return float(lat), float(lon)
        return lat, lon

    elif proj_code == SINEQA:
        cosphi = np.cos(y)
        safe_cosphi = np.where(cosphi == 0.0, 1.0, cosphi)
        lat = np.where(cosphi <= 0.0, y, np.arcsin(np.clip(np.sin(y) * coslatc + cosphi * np.cos(x / safe_cosphi) * sinlatc, -1.0, 1.0)))
        coslat = np.cos(lat)
        safe_coslat = np.where(coslat <= 0.0, 1.0, coslat)
        test = np.where(coslat <= 0.0, 0.0, cosphi * np.sin(x / safe_cosphi) / safe_coslat)
        lon = np.where(cosphi <= 0.0, lonc, np.where(coslat <= 0.0, lonc, np.arcsin(np.clip(test, -1.0, 1.0)) + lonc))

        x0 = np.copy(x)
        mask_pos = x0 > (np.pi / 2.0)
        while np.any(mask_pos):
            lon = np.where(mask_pos, np.pi - lon, lon)
            x0 = np.where(mask_pos, x0 - np.pi, x0)
            mask_pos = x0 > (np.pi / 2.0)

        mask_neg = x0 < (-np.pi / 2.0)
        while np.any(mask_neg):
            lon = np.where(mask_neg, -np.pi - lon, lon)
            x0 = np.where(mask_neg, x0 + np.pi, x0)
            mask_neg = x0 < (-np.pi / 2.0)

        if return_status:
            status = np.where(cosphi < 0.0, -1, 0)
            status = np.where(coslat < 0.0, 1, status)
            status = np.where(np.abs(x) > np.pi * cosphi, -1, status)
            if is_scalar:
                return float(lat), float(lon), int(status)
            return lat, lon, status
        if is_scalar:
            return float(lat), float(lon)
        return lat, lon

    elif proj_code == MERCATOR:
        phicom = 2.0 * np.arctan(np.exp(y))
        sinphi = -np.cos(phicom)
        cosphi = np.sin(phicom)
        lat = np.arcsin(np.clip((sinphi * coslatc) + (cosphi * np.cos(x) * sinlatc), -1.0, 1.0))
        cos_lat = np.cos(lat)
        safe_cos = np.where(cos_lat == 0.0, 1.0, cos_lat)
        lon = np.arcsin(np.clip(cosphi * np.sin(x) / safe_cos, -1.0, 1.0)) + lonc
        if return_status:
            status = np.where(arc_distance(lat, lon, latc, lonc) > (np.pi / 2.0), 1, 0)
            status = np.where(np.abs(x) > (np.pi / 2.0), -1, status)
            if is_scalar:
                return float(lat), float(lon), int(status)
            return lat, lon, status
        if is_scalar:
            return float(lat), float(lon)
        return lat, lon

    else:
        # Azimuthal projections: POSTEL, GNOMONIC, STEREOGRAPHIC, ORTHOGRAPHIC, LAMBERT
        r = np.hypot(x, y)
        safe_r = np.where(r == 0.0, 1.0, r)
        cosp = np.where(r == 0.0, 1.0, x / safe_r)
        sinp = np.where(r == 0.0, 0.0, y / safe_r)
        status = np.zeros_like(r, dtype=int)

        if proj_code == POSTEL:
            rm = r
            status = np.where(rm > (np.pi / 2.0), 1, 0)
        elif proj_code == GNOMONIC:
            rm = np.arctan(r)
        elif proj_code == STEREOGRAPHIC:
            rm = 2.0 * np.arctan(0.5 * r)
            status = np.where(rm > (np.pi / 2.0), 1, 0)
        elif proj_code == ORTHOGRAPHIC:
            status = np.where(r > 1.0, -1, 0)
            rm = np.arcsin(np.clip(r, 0.0, 1.0))
        elif proj_code == LAMBERT:
            status = np.where(r > 2.0, -1, 0)
            rm = 2.0 * np.arcsin(0.5 * np.clip(r, 0.0, 2.0))
            status = np.where((rm > (np.pi / 2.0)) & (status == 0), 1, status)
        else:
            raise ValueError(f"Unknown projection code: {projection}")

        cosr = np.cos(rm)
        sinr = np.sin(rm)
        sinlat = sinlatc * cosr + coslatc * sinr * sinp
        lat = np.arcsin(np.clip(sinlat, -1.0, 1.0))
        coslat = np.cos(lat)
        safe_coslat = np.where(coslat == 0.0, 1.0, coslat)
        sinlon = np.where(coslat == 0.0, 0.0, sinr * cosp / safe_coslat)
        lon = np.arcsin(np.clip(sinlon, -1.0, 1.0))
        lon = np.where(cosr < (sinlat * sinlatc), np.pi - lon, lon)
        lon = lon + lonc

        if return_status:
            if is_scalar:
                return float(lat), float(lon), int(status)
            return lat, lon, status
        if is_scalar:
            return float(lat), float(lon)
        return lat, lon


def sphere2img(lat, lon, latc, lonc, xcenter, ycenter, rsun, peff,
               ecc=0.0, chi=0.0, xinvrt=0, yinvrt=0, return_hemisphere=False):
    """
    Perform a mapping from heliographic coordinates latitude and longitude
    (in radians) to plate location on an image of the sun.

    Compatible with both cartography.c and SDO/HMI sphere2img.pro.
    Copied/translated from SSWIDL/SDO/HMI/IDL code by Xudong Sun (xudongs@hawaii.edu).

    Arguments:
        lat, lon: Heliographic coordinates in radians (scalar or numpy array).
        latc, lonc: Heliographic coordinates of disc center (in radians).
        xcenter, ycenter: Plate location of image center in plate coordinates (pixels).
        rsun: Apparent semi-diameter of solar disc in plate coordinates (pixels).
        peff: Position angle of heliographic pole, measured eastward from north (radians).
        ecc: Eccentricity of fit ellipse due to distortion (default: 0.0).
        chi: Position angle of ellipse major axis (radians, default: 0.0).
        xinvrt, yinvrt: Coordinate inversion flags (default: 0).
        return_hemisphere: If True, returns (x, y, hemisphere). If False, returns (x, y).

    Returns:
        x, y: Plate coordinates (xi, eta).
        hemisphere (optional): 1 if point is on far side (>90 deg from disc center), 0 otherwise.
    """
    is_scalar = np.isscalar(lat) and np.isscalar(lon)
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)

    sin_asd = 0.004660
    cos_asd = 0.99998914

    sin_latc = np.sin(latc)
    cos_latc = np.cos(latc)

    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    cos_lat_lon = cos_lat * np.cos(lon - lonc)

    cos_cang = sin_lat * sin_latc + cos_latc * cos_lat_lon
    # 1 if on far side (>90 deg from disc center, cos_cang < 0), 0 otherwise
    hemisphere = np.where(cos_cang < 0.0, 1, 0)

    r = rsun * cos_asd / (1.0 - cos_cang * sin_asd)
    xr = r * cos_lat * np.sin(lon - lonc)
    yr = r * (sin_lat * cos_latc - sin_latc * cos_lat_lon)

    if xinvrt:
        xr = -xr
    if yinvrt:
        yr = -yr

    if 0.0 < ecc < 1.0:
        squash = np.sqrt(1.0 - ecc * ecc)
        cchi = np.cos(chi)
        schi = np.sin(chi)
        s2chi = schi * schi
        c2chi = 1.0 - s2chi
        xp = xr * (s2chi + squash * c2chi) - yr * (1.0 - squash) * schi * cchi
        yp = yr * (c2chi + squash * s2chi) - xr * (1.0 - squash) * schi * cchi
        xr = xp
        yr = yp

    cospa = np.cos(peff)
    sinpa = np.sin(peff)
    x = xr * cospa - yr * sinpa + xcenter
    y = xr * sinpa + yr * cospa + ycenter

    if is_scalar:
        x_out, y_out = float(x), float(y)
        hemi_out = int(hemisphere)
    else:
        x_out, y_out = x, y
        hemi_out = hemisphere

    if return_hemisphere:
        return x_out, y_out, hemi_out
    return x_out, y_out


def sphere2plane(lat, lon, latc, lonc, projection=CYLEQA):
    """
    Perform a mapping from heliographic coordinates latitude and longitude
    (in radians) to map location in the given projection.
    Copied/translated from SSWIDL/SDO/HMI/IDL code by Xudong Sun (xudongs@hawaii.edu).
    """
    if isinstance(projection, str):
        proj_map = {
            "RECTANGULAR": RECTANGULAR, "CASSINI": CASSINI, "MERCATOR": MERCATOR,
            "CYLEQA": CYLEQA, "SINEQA": SINEQA, "GNOMONIC": GNOMONIC,
            "POSTEL": POSTEL, "STEREOGRAPHIC": STEREOGRAPHIC,
            "ORTHOGRAPHIC": ORTHOGRAPHIC, "LAMBERT": LAMBERT
        }
        proj_code = proj_map.get(projection.upper(), -1)
    else:
        proj_code = int(projection)

    sin_latc = np.sin(latc)
    cos_latc = np.cos(latc)
    yc_merc = np.log(np.tan(np.pi / 4.0 + 0.5 * latc))

    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    cos_lat_lon = cos_lat * np.cos(lon - lonc)
    cos_cang = sin_lat * sin_latc + cos_latc * cos_lat_lon
    hemisphere = np.where(cos_cang < 0.0, 1, 0)

    if proj_code == RECTANGULAR:
        x = lon - lonc
        y = lat - latc
        return x, y, hemisphere
    elif proj_code == CASSINI:
        x = np.arcsin(np.clip(cos_lat * np.sin(lon - lonc), -1.0, 1.0))
        y = np.arctan2(np.tan(lat), np.cos(lon - lonc)) - latc
        return x, y, hemisphere
    elif proj_code == CYLEQA:
        x = lon - lonc
        y = sin_lat - sin_latc
        return x, y, hemisphere
    elif proj_code == SINEQA:
        x = cos_lat * (lon - lonc)
        y = lat - latc
        return x, y, hemisphere
    elif proj_code == MERCATOR:
        x = lon - lonc
        y = np.log(np.tan(np.pi / 4.0 + 0.5 * lat)) - yc_merc
        return x, y, hemisphere

    rm = np.arccos(np.clip(cos_cang, -1.0, 1.0))

    if proj_code == POSTEL:
        r = rm
    elif proj_code == GNOMONIC:
        r = np.tan(rm)
    elif proj_code == STEREOGRAPHIC:
        r = 2.0 * np.tan(0.5 * rm)
    elif proj_code == ORTHOGRAPHIC:
        r = np.sin(rm)
    elif proj_code == LAMBERT:
        r = 2.0 * np.sin(0.5 * rm)
    else:
        return -1

    safe_rm = np.where(rm == 0.0, 1.0, rm)
    x = np.where(rm != 0.0, r * cos_lat * np.sin(lon - lonc) / safe_rm, 0.0)
    y = np.where(rm != 0.0, r * (sin_lat * cos_latc - sin_latc * cos_lat_lon) / safe_rm, 0.0)

    return x, y, hemisphere
