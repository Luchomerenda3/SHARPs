"""
 *  cartography.py     <---    cartography.c                          
 *
 *  Functions for mapping between plate, heliographic, and various map
 *    coordinate systems, including scaling.
 *
 *  Contents:
 *      img2sphere		Map from plate location to heliographic coordinates
 *      plane2sphere		Map from map location to heliographic or geographical coordinates
 *      sphere2img		Map from heliographic coordinates to plate location
 *      sphere2plane		Map from heliographic/geographic coordinates to map location
 *
 *  Responsible:  Rick Bogart                   RBogart@solar.Stanford.EDU
 *  Translation to python made by Lucho. 
 *
 *
 *  Bugs:
 *    - It is assumed that the function atan2() returns the correct value
 *      for the quadrant; not all libraries may support this feature.
 *    - sphere2img uses a constant for the correction due to the finite
 *      distance to the sun.
 *    - sphere2plane and plane2sphere are not true inverses for the following
 *	    projections: (Lambert) cylindrical equal area, sinusoidal equal area
 *	    (Sanson-Flamsteed), and Mercator; for these projections, sphere2plane is
 *	    implemented as the normal projection, while plane2sphere is implemented
 *	    as the oblique projection tangent at the normal to the central meridian
 *      plane2sphere doesn't return 1 if the x coordinate would map to a point
 *	    off the surface.
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
    return np.arccos(cosa)


def img2sphere(x, y, ang_r, latc, lonc, pa):

    """  Map projected coordinates (x, y) to (lon, lat) and (rho | sig, chi)
     *
     *  Arguments:
     *    x }	    Plate locations, in units of the image radius, relative
     *    y }		to the image center
     *    ang_r	    Apparent semi-diameter of sun (angular radius of sun at
     *			the observer's tangent line)
     *    latc	    Latitude of disc center, uncorrected for light travel time
     *    lonc	    Longitude of disc center
     *    pa	    Position angle of solar north on image, measured eastward
     *			from north (sky coordinates)
     *  Return values:
     *    rho	    Angle point:sun_center:observer
     *    lon	    Heliographic longitude
     *    lat	    Heliographic latitude
     *    sinlat	    sine of heliographic latitude
     *    coslat	    cosine of heliographic latitude
     *    sig	    Angle point:observer:sun_center
     *    mu	    cosine of angle between the point:observer line and the
     *			local normal
     *    chi	    Position angle on image measured westward from solar
     *			north
     *
     *  All angles are in radians.
     *  Return value is 1 if point is outside solar radius (in which case the
     *    heliographic coordinates and mu are meaningless), 0 otherwise.
     *  It is assumed that the image is direct; the x or y coordinates require a
     *    sign change if the image is inverted.
     """

    ang_r0 = 0.0
    sinang_r = 0.0
    tanang_r = 0.0
    latc0 = 0.0
    coslatc = 0.0
    sinlatc = 0.0

    if ang_r != ang_r0:
        sinang_r = np.sin(ang_r)
        tanang_r = np.tan(ang_r)
        ang_r0 = ang_r

    if latc != latc0:
        sinlatc = np.sin(latc)
        coslatc = np.cos(latc)
        latc0 = latc

    chi = np.arctan2(x, y) + pa
    while chi > 2 * np.pi:
        chi -= 2 * np.pi
    while chi < 0:
        chi += 2 * np.pi

    sig = np.arctan(np.hypot(x, y) * tanang_r)
    sinsig = np.sin(sig)
    rho = np.arcsin(sinsig / sinang_r) - sig

    if sig > ang_r:
        return -1

    mu = np.cos(rho + sig)
    sinr = np.sin(rho)
    cosr = np.cos(rho)

    sinlat = sinlatc * np.cos(rho) + coslatc * sinr * np.cos(chi)
    coslat = np.sqrt(1.0 - sinlat * sinlat)
    lat = np.arcsin(sinlat)
    sinlon = sinr * np.sin(chi) / coslat
    lon = np.arcsin(sinlon)

    if cosr < (sinlat * sinlatc):
        lon = np.pi - lon

    lon += lonc
    while lon < 0:
        lon += 2 * np.pi
    while lon >= 2 * np.pi:
        lon -= 2 * np.pi

    return rho, lat, lon, sinlat, coslat, sig, mu, chi

def plane2sphere(x, y, latc, lonc, projection):

    """
     *  Perform the inverse mapping from rectangular coordinates x, y on a map
     *    in a particular projection to heliographic (or geographic) coordinates
     *    latitude and longitude (in radians).
     *  The map coordinates are first transformed into arc and azimuth coordinates
     *    relative to the center of the map according to the appropriate inverse
     *    transformation for the projection, and thence to latitude and longitude
     *    from the known heliographic coordinates of the map center (in radians).
     *  The scale of the map coordinates is assumed to be in units of radians at
     *    the map center (or other appropriate location of minimum distortion).
     *
     *  Arguments:
     *      x }         Map coordinates, in units of radians at the scale
     *      y }           appropriate to the map center
     *      latc        Latitude of the map center (in radians)
     *      lonc        Longitude of the map center (in radians)
     *      *lat        Returned latitude (in radians)
     *      *lon        Returned longitude (in radians)
     *      projection  A code specifying the map projection to be used: see below
     *
     *  The following projections are supported:
     *      RECTANGULAR     A "rectangular" mapping of x and y directly to
     *                      longitude and latitude, respectively; it is the
     *			normal cylindrical equidistant projection (plate
     *			carr�e) tangent at the equator and equidistant
     *			along meridians. Central latitudes off the equator
     *			merely result in displacement of the map in y
     *			Also known as CYLEQD
     *		CASSINI		The transverse cylindrical equidistant projection
     *			(Cassini-Soldner) equidistant along great circles
     *			perpendicular to the central meridian
     *      MERCATOR        Mercator's conformal projection, in which paths of
     *                      constant bearing are straight lines
     *      CYLEQA          Lambert's normal equal cylindrical (equal-area)
     *                      projection, in which evenly-spaced meridians are
     *                      evenly spaced in x and evenly-spaced parallels are
     *                      separated by the cosine of the latitude
     *      SINEQA          The Sanson-Flamsteed sinusoidal equal-area projection,
     *                      in which evenly-spaced parallels are evenly spaced in
     *                      y and meridians are sinusoidal curves
     *      GNOMONIC        The gnomonic, or central, projection, in which all
     *                      straight lines correspond to geodesics; projection
     *                      from the center of the sphere onto a tangent plane
     *      POSTEL          Postel's azimuthal equidistant projection, in which
     *                      straight lines through the center of the map are
     *                      geodesics with a uniform scale
     *      STEREOGRAPHIC   The stereographic projection, mapping from the
     *                      antipode of the map center onto a tangent plane
     *      ORTHOGRAPHIC    The orthographic projection, mapping from infinity
     *                      onto a tangent plane
     *      LAMBERT         Lambert's azimuthal equivalent projection
     *
     *  The function returns -1 if the requested point on the map does not project
     *    back to the sphere or is not a principal value, 1 if it projects to a
     *    point on a hidden hemisphere (if that makes sense), 0 otherwise
    """

    latc0 = 0.0
    sinlatc = 0.0
    coslatc = 1.0
    status = 0

    if latc != latc0:
        coslatc = np.cos(latc)
        sinlatc = np.sin(latc)

    latc0 = latc

    if projection == RECTANGULAR:
        lon = lonc + x
        lat = latc + y
        if arc_distance(lat, lon, latc, lonc) > np.pi / 2:
            status = 1
        if np.fabs(x) > np.pi or np.fabs(y) > np.pi / 2:
            status = -1
        return lat, lon, status

    elif projection == CASSINI:
        sinx = np.sin(x)
        cosy = np.cos(y + latc)
        siny = np.sin(y + latc)
        lat = np.arccos(np.sqrt(cosy * cosy + siny * siny * sinx * sinx))
        if y < -latc:
            lat *= -1
        lon = lonc + np.arcsin(sinx / np.cos(lat)) if np.fabs(lat) < np.pi / 2 else lonc
        if y > (np.pi / 2 - latc) or y < (-np.pi / 2 - latc):
            lon = 2 * lonc + np.pi - lon
        if lon < -np.pi:
            lon += 2 * np.pi
        if lon > np.pi:
            lon -= 2 * np.pi
        if np.arccos(lat, lon, latc, lonc) > np.pi / 2:
            status = 1
        if np.fabs(x) > np.pi or np.fabs(y) > np.pi / 2:
            status = -1

        return lat, lon, status

    if projection == CYLEQA:
        if np.abs(y) > 1.0:
            y = np.copysign(1.0, y)
            status = -1
        cosphi = np.sqrt(1.0 - y*y)
        lat = np.arcsin((y * coslatc) + (cosphi * np.cos(x) * sinlatc))
        test = (np.cos(lat) == 0.0) and 0.0 or cosphi * np.sin(x) / np.cos(lat)
        lon = np.arcsin(test) + lonc
        if np.fabs(x) > np.pi/2:
            status = 1
            while x > np.pi/2:
                lon = np.pi - lon
                x -= np.pi
            while x < -np.pi/2:
                lon = -np.pi - lon
                x += np.pi
        if arc_distance(lat, lon, latc, lonc) > np.pi/2:
            status = 1
        return lat, lon, status

    if projection == SINEQA:
        cosphi = np.cos(y)
        if cosphi <= 0.0:
            lat = y
            lon = lonc
            if cosphi < 0.0:
                status = -1
            return lat, lon, status

        lat = np.arcsin((np.sin(y) * coslatc) + (cosphi * np.cos(x/cosphi) * sinlatc))
        coslat = np.cos(lat)
        if coslat <= 0.0:
            lon = lonc
            if coslat < 0.0:
                status = 1
            return lat, lon, status

        test = cosphi * np.sin(x/cosphi) / coslat
        lon = np.arcsin(test) + lonc

        if np.fabs(x) > np.pi * cosphi:
            return lat, lon, -1

        if np.abs(x) > np.pi/2:
            status = 1
            while x > np.pi/2:
                lon = np.pi - lon
                x -= np.pi
            while x < -np.pi/2:
                lon = -np.pi - lon
                x += np.pi

        return lat, lon, status

    if projection == MERCATOR:
        phicom = 2.0 * np.arctan(np.exp(y))
        sinphi = -np.cos(phicom)
        cosphi = np.sin(phicom)
        lat = np.arcsin((sinphi * coslatc) + (cosphi * np.cos(x) * sinlatc))
        lon = np.arcsin(cosphi * np.sin(x) / np.cos(lat)) + lonc
        if arc_distance(lat, lon, latc, lonc) > np.pi / 2:
            status = 1
        if np.abs(x) > np.pi / 2:
            status = -1
        return lat, lon, status

    # Convert to polar coordinates
    r = np.hypot(x, y)
    cosp = (r == 0.0) * 1.0 + (r != 0.0) * x / r
    sinp = (r == 0.0) * 0.0 + (r != 0.0) * y / r

    if projection == POSTEL:
        rm = r
        if rm > np.pi / 2:
            status = 1

    elif projection == GNOMONIC:
        rm = np.arctan(r)

    elif projection == STEREOGRAPHIC:
        rm = 2 * np.arctan(0.5 * r)
        if rm > np.pi / 2:
            status = 1

    elif projection == ORTHOGRAPHIC:
        rm = np.arcsin(np.clip(r, 0, 1))
        if r > 1.0:
            status = -1

    elif projection == LAMBERT:
        rm = 2 * np.arcsin(0.5 * np.clip(r, 0, 2))
        if rm > np.pi / 2 and status == 0:
            status = 1


    cosr = np.cos(rm)
    sinr = np.sin(rm)
    sinlat = sinlatc * cosr + coslatc * sinr * sinp
    lat = np.arcsin(sinlat)
    coslat = np.cos(lat)
    sinlon = (coslat == 0.0) if 0.0 else sinr * cosp / coslat

    lon = np.arcsin(sinlon)
    if cosr < (sinlat * sinlatc):
        lon = np.pi - lon
    lon += lonc

    return lat, lon, status


def sphere2img(lat, lon, latc, lonc, xcenter, ycenter, rsun, peff, ecc, chi, xinvrt, yinvrt):

    """
     *  Perform a mapping from heliographic coordinates latitude and longitude
     *    (in radians) to plate location on an image of the sun.  The plate
     *    location is in units of the image radius and is given relative to
     *    the image center.  The function returns 1 if the requested point is
     *    on the far side (>90 deg from disc center), 0 otherwise.
     *
     *  Arguments:
     *      lat         Latitude (in radians)
     *      lon         Longitude (in radians)
     *      latc        Heliographic latitude of the disc center (in radians)
     *      lonc        Heliographic longitude of the disc center (in radians)
     *      *x }        Plate locations, in units of the image radius, relative
     *      *y }          to the image center
     *      xcenter }   Plate locations of the image center, in units of the
     *      ycenter }     image radius, and measured from an arbitrary origin
     *                    (presumably the plate center or a corner)
     *      rsun        Apparent semi-diameter of the solar disc, in plate
     *                    coordinates
     *      peff        Position angle of the heliographic pole, measured
     *                    eastward from north, relative to the north direction
     *                    on the plate, in radians
     *      ecc         Eccentricity of the fit ellipse presumed due to image
     *                    distortion (no distortion in direction of major axis)
     *      chi         Position angle of the major axis of the fit ellipse,
     *                    measure eastward from north,  relative to the north
     *                    direction on the plate, in radians (ignored if ecc = 0)
     *      xinvrt}     Flag parameters: if not equal to 0, the respective
     *      yinvrt}       coordinates on the image x and y are inverted
     *
     *  The heliographic coordinates are first mapped into the polar coordinates
     *    in an orthographic projection centered at the appropriate location and
     *    oriented with north in direction of increasing y and west in direction
     *    of increasing x.  The radial coordinate is corrected for foreshortening
     *    due to the finite distance to the Sun. If the eccentricity of the fit
     *    ellipse is non-zero the coordinate of the mapped point is proportionately
     *    reduced in the direction parallel to the minor axis.
     *
     *  Bugs:
     *    The finite distance correction uses a fixed apparent semi-diameter
     *    of 16'01'' appropriate to 1.0 AU.  In principle the plate radius could
     *    be used, but this would require the plate scale to be supplied and the
     *    correction would probably be erroneous and in any case negligible.
     *
     *    The ellipsoidal correction has not been tested very thoroughly.
     *
     *    The return value is based on a test which does not take foreshortening
     *    into account.
    """
    sin_asd = 0.004660
    cos_asd = 0.99998914
    last_latc = 0.0
    cos_latc = 1.0
    sin_latc = 0.0

    if latc != last_latc:
        sin_latc = np.sin(latc)
        cos_latc = np.cos(latc)
        last_latc = latc

    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    cos_lat_lon = cos_lat * np.cos(lon - lonc)

    cos_cang = sin_lat * sin_latc + cos_latc * cos_lat_lon
    hemisphere = 1 if cos_cang >= 0 else 0
    r = rsun * cos_asd / (1 - cos_cang * sin_asd)
    xr = r * cos_lat * np.sin(lon - lonc)
    yr = r * (sin_lat * cos_latc - sin_latc * cos_lat_lon)

    if xinvrt:
        xr *= -1
    if yinvrt:
        yr *= -1
    if ecc > 0 and ecc < 1:
        squash = np.sqrt(1 - ecc * ecc)
        cchi = np.cos(chi)
        schi = np.sin(chi)
        s2chi = schi * schi
        c2chi = 1 - s2chi
        xp = xr * (s2chi + squash * c2chi) - yr * (1 - squash) * schi * cchi
        yp = yr * (c2chi + squash * s2chi) - xr * (1 - squash) * schi * cchi
        xr = xp
        yr = yp

    cospa = np.cos(peff)
    sinpa = np.sin(peff)
    x = xr * cospa - yr * sinpa
    y = xr * sinpa + yr * cospa
    y += ycenter
    x += xcenter

    return x, y, hemisphere


def sphere2plane(lat, lon, latc, lonc, projection):

    """
     *  Perform a mapping from heliographic (or geographic or celestial)
     *    coordinates latitude and longitude (in radians) to map location in
     *    the given projection.  The function returns 1 if the requested point is
     *    on the far side (>90 deg from disc center), 0 otherwise.
     *
     *  Arguments:
     *      lat         Latitude (in radians)
     *      lon         Longitude (in radians)
     *      latc        Heliographic latitude of the disc center (in radians)
     *      lonc        Heliographic longitude of the disc center (in radians)
     *      *x }        Plate locations, in units of the image radius, relative
     *      *y }          to the image center
     *      projection  code specifying the map projection to be used:
     *		      see plane2sphere
    """

    last_latc = 0.0
    cos_latc = 1.0
    sin_latc = 0.0
    yc_merc = 0.0

    if latc != last_latc:
        sin_latc = np.sin(latc)
        cos_latc = np.cos(latc)
        last_latc = latc
        yc_merc = np.log(np.tan(np.pi / 4 + 0.5 * latc))

    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    cos_lat_lon = cos_lat * np.cos(lon - lonc)
    cos_cang = sin_lat * sin_latc + cos_latc * cos_lat_lon
    hemisphere = 1 if (cos_cang < 0.0) else 0

    if projection == "RECTANGULAR":
        x = lon - lonc
        y = lat - latc
        return x, y, hemisphere
    elif projection == "CASSINI":
        x = np.arcsin(cos_lat * np.sin(lon - lonc))
        y = np.arctan2(np.tan(lat), np.cos(lon - lonc)) - latc
        return x, y, hemisphere
    elif projection == "CYLEQA":
        x = lon - lonc
        y = sin_lat - sin_latc
        return x, y, hemisphere
    elif projection == "SINEQA":
        x = cos_lat * (lon - lonc)
        y = lat - latc
        return x, y, hemisphere
    elif projection == "MERCATOR":
        x = lon - lonc
        y = np.log(np.tan(np.pi / 4 + 0.5 * lat)) - yc_merc
        return x, y, hemisphere

    rm = np.arccos(cos_cang)

    if projection == "POSTEL":
        r = rm
    elif projection == "GNOMONIC":
        r = np.tan(rm)
    elif projection == "STEREOGRAPHIC":
        r = 2.0 * np.tan(0.5 * rm)
    elif projection == "ORTHOGRAPHIC":
        r = np.sin(rm)
    elif projection == "LAMBERT":
        r = 2.0 * np.sin(0.5 * rm)
    else:
        return -1

    if rm != 0:
        x = r * cos_lat * np.sin(lon - lonc) / np.sin(rm)
        y = r * (sin_lat * cos_latc - sin_latc * cos_lat_lon) / np.sin(rm)
    else:
        x = 0
        y = 0

    return x, y, hemisphere
