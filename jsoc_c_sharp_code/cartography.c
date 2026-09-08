/*
 *  cartography.c                               	~rick/src/util
 *
 *  Functions for mapping between plate, heliographic, and various map
 *    coordinate systems, including scaling.
 *
 *  Contents:
 *      img2sphere		Map from plate location to heliographic
 *				coordinates
 *      plane2sphere		Map from map location to heliographic or
 *				geographical coordinates
 *      sphere2img		Map from heliographic coordinates to plate
 *				location
 *      sphere2plane		Map from heliographic/geographic coordinates
 *				to map location
 *
 *  Responsible:  Rick Bogart                   RBogart@solar.Stanford.EDU
 *
 *  Usage:
 *    int img2sphere (double x, double y, double ang_r, double latc,
 *	double lonc, double pa, double *rho, double *lat, double *lon,
 *      double *sinlat, double *coslat, double *sig, double *mu, double *chi);
 *    int plane2sphere (double x, double y, double latc, double lonc,
 *      double *lat, double *lon, int projection);
 *    int sphere2img (double lat, double lon, double latc, double lonc,
 *      double *x, double *y, double xcenter, double ycenter,
 *      double rsun, double peff, double ecc, double chi,
 *      int xinvrt, int yinvrt);
 *    int sphere2plane (double lat, double lon, double latc, double lonc,
 *	double *x, double *y, int projection);
 *
 *  Bugs:
 *    It is assumed that the function atan2() returns the correct value
 *      for the quadrant; not all libraries may support this feature.
 *    sphere2img uses a constant for the correction due to the finite
 *      distance to the sun.
 *    sphere2plane and plane2sphere are not true inverses for the following
 *	projections: (Lambert) cylindrical equal area, sinusoidal equal area
 *	(Sanson-
 *	Flamsteed), and Mercator; for these projections, sphere2plane is
 *	implemented as the normal projection, while plane2sphere is implemented
 *	as the oblique projection tangent at the normal to the central meridian
 *    plane2sphere doesn't return 1 if the x coordinate would map to a point
 *	off the surface.
 *
 *  Planned updates:
 *    Provide appropriate oblique versions for cylindrical and
 *	pseudo-cylindrical projections in sphere2plane
 *
 *  Revision history is at end of file.
 */
#include <math.h>

#define RECTANGULAR	(0)
#define CASSINI		(1)
#define MERCATOR	(2)
#define CYLEQA		(3)
#define SINEQA		(4)
#define GNOMONIC	(5)
#define POSTEL		(6)
#define STEREOGRAPHIC	(7)
#define ORTHOGRAPHIC	(8)
#define LAMBERT		(9)

static double arc_distance (double lat, double lon, double latc, double lonc) {
  double cosa = sin (lat) * sin (latc) + cos (lat) * cos (latc) * cos (lon - lonc);
  return acos (cosa);
}

int img2sphere (double x, double y, double ang_r, double latc, double lonc,
    double pa, double *rho, double *lat, double *lon, double *sinlat,
    double *coslat, double *sig, double *mu, double *chi) {
/*
 *  Map projected coordinates (x, y) to (lon, lat) and (rho | sig, chi)
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
 *
 */
  static double ang_r0 = 0.0, sinang_r = 0.0, tanang_r = 0.0;
  static double latc0 = 0.0, coslatc = 1.0, sinlatc = 0.0;
  double cosr, sinr, sinlon, sinsig;

  if (ang_r != ang_r0) {
    sinang_r = sin (ang_r);
    tanang_r = tan (ang_r);
    ang_r0 = ang_r;
  }
  if (latc != latc0) {
    sinlatc = sin (latc);
    coslatc = cos (latc);
    latc0 = latc;
  }
  *chi = atan2 (x, y) + pa;
  while (*chi > 2 * M_PI) *chi -= 2 * M_PI;
  while (*chi < 0.0) *chi += 2 * M_PI;
             /*  Camera curvature correction, no small angle approximations  */
  *sig = atan (hypot (x, y) * tanang_r);
  sinsig = sin (*sig);
  *rho = asin (sinsig / sinang_r) - *sig;
  if (*sig > ang_r) return (-1);
  *mu = cos (*rho + *sig);
  sinr = sin (*rho);
  cosr = cos (*rho);

  *sinlat = sinlatc * cos (*rho) + coslatc * sinr * cos (*chi);
  *coslat = sqrt (1.0 - *sinlat * *sinlat);
  *lat = asin (*sinlat);
  sinlon = sinr * sin (*chi) / *coslat;
  *lon = asin (sinlon);
  if (cosr < (*sinlat * sinlatc)) *lon = M_PI - *lon;
  *lon += lonc;
  while (*lon < 0.0) *lon += 2 * M_PI;
  while (*lon >= 2 * M_PI) *lon -= 2 * M_PI;
  return (0);
}

int plane2sphere (double x, double y, double latc, double lonc,
        double *lat, double *lon, int projection) {
/*
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
 */
  static double latc0 = 0.0, sinlatc = 0.0, coslatc = 1.0 ;
  double r, rm, test;
  double cosr, sinr, cosp, sinp, coslat, sinlat, sinlon;
  double sinphi, cosphi, phicom;
  int status = 0;

  if (latc != latc0) {
    coslatc = cos (latc);
    sinlatc = sin (latc);
  }
  latc0 = latc;

  switch (projection) {
    case (RECTANGULAR):

// printf("  RECTANGULAR projection=%d\n", projection);

      *lon = lonc + x;
      *lat = latc + y;
      if (arc_distance (*lat, *lon, latc, lonc) > M_PI_2) status = 1;
      if (fabs (x) > M_PI || fabs (y) > M_PI_2) status = -1;
      return status;
    case (CASSINI): {

// printf("  CASSINI projection=%d\n", projection);

      double sinx = sin (x);
      double cosy = cos (y + latc);
      double siny = sin (y + latc);
      *lat = acos (sqrt (cosy * cosy + siny * siny * sinx * sinx));
      if (y < -latc) *lat *= -1;
      *lon = (fabs (*lat) < M_PI_2) ? lonc + asin (sinx / cos (*lat)) : lonc;
      if (y > (M_PI_2 - latc) || y < (-M_PI_2 - latc)) 
        *lon = 2 * lonc + M_PI - *lon;
      if (*lon < -M_PI) *lon += 2* M_PI;
      if (*lon > M_PI) *lon -= 2 * M_PI;
      if (arc_distance (*lat, *lon, latc, lonc) > M_PI_2) status = 1;
      if (fabs (x) > M_PI || fabs (y) > M_PI_2) status = -1;
      return status;
      }
    case (CYLEQA):

// printf("  CYLEQA projection=%d\n", projection);

      if (fabs (y) > 1.0) {
        y = copysign (1.0, y);
	status = -1;
      }
      cosphi = sqrt (1.0 - y*y);
      *lat = asin ((y * coslatc) + (cosphi * cos (x) * sinlatc));
      test = (cos (*lat) == 0.0) ? 0.0 : cosphi * sin (x) / cos (*lat);
      *lon = asin (test) + lonc;
      if (fabs (x) > M_PI_2) {
        status = 1;
        while (x > M_PI_2) {
          *lon = M_PI - *lon;
	  x -= M_PI;
        }
        while (x < -M_PI_2) {
          *lon = -M_PI - *lon;
	  x += M_PI;
        }
      }
      if (arc_distance (*lat, *lon, latc, lonc) > M_PI_2) status = 1;
      return status;
    case (SINEQA):

// printf("  SINEQA projection=%d\n", projection);

      cosphi = cos (y);
      if (cosphi <= 0.0) {
        *lat = y;
        *lon = lonc;
	if (cosphi < 0.0) status = -1;
        return status;
      }
      *lat = asin ((sin (y) * coslatc) + (cosphi * cos (x/cosphi) * sinlatc));
      coslat = cos (*lat);
      if (coslat <= 0.0) {
        *lon = lonc;
	if (coslat < 0.0) status = 1;
        return status;
      }
      test = cosphi * sin (x/cosphi) / coslat;
      *lon = asin (test) + lonc;
      if (fabs (x) > M_PI * cosphi) return (-1);
      if (fabs (x) > M_PI_2) {
        status = 1;
        while (x > M_PI_2) {
          *lon = M_PI - *lon;
	  x -= M_PI;
        }
        while (x < -M_PI_2) {
          *lon = -M_PI - *lon;
	  x += M_PI;
        }
      }
/*
      if (arc_distance (*lat, *lon, latc, lonc) > M_PI_2) status = 1;
*/
      return status;
    case (MERCATOR):

// printf("  MERCATOR projection=%d\n", projection);

      phicom = 2.0 * atan (exp (y));
      sinphi = -cos (phicom);
      cosphi = sin (phicom);
      *lat = asin ((sinphi  * coslatc) + (cosphi * cos (x) * sinlatc));
      *lon = asin (cosphi * sin (x) / cos (*lat)) + lonc;
      if (arc_distance (*lat, *lon, latc, lonc) > M_PI_2) status = 1;
      if (fabs (x) > M_PI_2) status = -1;

      return status;
  }
                                          /*  Convert to polar coordinates  */
  r = hypot (x, y);
  cosp = (r == 0.0) ? 1.0 : x / r;
  sinp = (r == 0.0) ? 0.0 : y / r;
                                                        /*  Convert to arc  */
  switch (projection) {
    case (POSTEL):

// printf("  POSTEL projection=%d\n", projection);

      rm = r;
      if (rm > M_PI_2) status = 1;
      break;
    case (GNOMONIC):

// printf("  GNOMONIC projection=%d\n", projection);

      rm = atan (r);
      break;
    case (STEREOGRAPHIC):

// printf("  STEREOGRAPHIC projection=%d\n", projection);

      rm = 2 * atan (0.5 * r);
      if (rm > M_PI_2) status = 1;
      break;
    case (ORTHOGRAPHIC):

// printf("  ORTHOGRAPHIC projection=%d\n", projection);

      if ( r > 1.0 ) {
	r = 1.0;
        status = -1;
      }
      rm = asin (r);
      break;
    case (LAMBERT):
      if ( r > 2.0 ) {
        r = 2.0;
        status = -1;
      }
      rm = 2 * asin (0.5 * r);
      if (rm > M_PI_2 && status == 0) status = 1;

// printf("  LAMBERT projection=%d\n", projection);

      break;
  }
  cosr = cos (rm);
  sinr = sin (rm);
                                          /*  Convert to latitude-longitude  */
  sinlat = sinlatc * cosr + coslatc * sinr * sinp;
  *lat = asin (sinlat);
  coslat = cos (*lat);
  sinlon = (coslat == 0.0) ? 0.0 : sinr * cosp / coslat;
/*  This should never happen except for roundoff errors, but just in case:   */
/*
  if (sinlon + 1.0 <= 0.0) *lon = -M_PI_2;
  else if (sinlon - 1.0 >= 0.0) *lon = M_PI_2;
  else
*/
  *lon = asin (sinlon);
                                     /*  Correction suggested by Dick Shine  */
  if (cosr < (sinlat * sinlatc)) *lon = M_PI - *lon;
  *lon += lonc;
  return status;
}

int sphere2img (double lat, double lon, double latc, double lonc,
        double *x, double *y, double xcenter, double ycenter,
        double rsun, double peff, double ecc, double chi,
        int xinvrt, int yinvrt) {
/*
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
 */
  static double sin_asd = 0.004660, cos_asd = 0.99998914;
                                                   /*  appropriate to 1 AU  */
  static double last_latc = 0.0, cos_latc = 1.0, sin_latc = 0.0;
  double r, cos_cang, xr, yr;
  double sin_lat, cos_lat, cos_lat_lon, cospa, sinpa;
  double squash, cchi, schi, c2chi, s2chi, xp, yp;
  int hemisphere;

  if (latc != last_latc) {
      sin_latc = sin (latc);
      cos_latc = cos (latc);
      last_latc = latc;
  }
  sin_lat     = sin (lat);
  cos_lat     = cos (lat);
  cos_lat_lon = cos_lat * cos (lon - lonc);

  cos_cang   = sin_lat * sin_latc + cos_latc * cos_lat_lon;
  hemisphere = (cos_cang < 0.0) ? 1 : 0;
  r = rsun * cos_asd / (1.0 - cos_cang * sin_asd);
  xr = r * cos_lat * sin (lon - lonc);
  yr = r * (sin_lat * cos_latc - sin_latc * cos_lat_lon);
                                        /*  Change sign for inverted images  */
  if (xinvrt) xr *= -1.0;
  if (yinvrt) yr *= -1.0;
                          /*  Correction for ellipsoidal squashing of image  */
  if (ecc > 0.0 && ecc < 1.0) {
    squash = sqrt (1.0 - ecc * ecc);
    cchi = cos (chi);
    schi = sin (chi);
    s2chi = schi * schi;
    c2chi = 1.0 - s2chi;
    xp = xr * (s2chi + squash * c2chi) - yr * (1.0 - squash) * schi * cchi;
    yp = yr * (c2chi + squash * s2chi) - xr * (1.0 - squash) * schi * cchi;
    xr = xp;
    yr = yp;
  }

  cospa = cos (peff);
  sinpa = sin (peff);
  *x = xr * cospa - yr * sinpa;
  *y = xr * sinpa + yr * cospa;

  *y += ycenter;
  *x += xcenter;

  return (hemisphere);
}

int sphere2plane (double lat, double lon, double latc, double lonc,
        double *x, double *y, int projection) {
/*
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
 */
  static double last_latc = 0.0, cos_latc = 1.0, sin_latc = 0.0, yc_merc;
  double r, rm, cos_cang;
  double sin_lat, cos_lat, cos_lat_lon;
  double cos_phi, sin_phi;
  int hemisphere;

  if (latc != last_latc) {
    sin_latc = sin (latc);
    cos_latc = cos (latc);
    last_latc = latc;
    yc_merc = log (tan (M_PI_4 + 0.5 * latc));
  }
  sin_lat = sin (lat);
  cos_lat = cos (lat);
  cos_lat_lon = cos_lat * cos (lon - lonc);
  cos_cang = sin_lat * sin_latc + cos_latc * cos_lat_lon;
  hemisphere = (cos_cang < 0.0) ? 1 : 0;
						/*  Cylindrical projections  */
  switch (projection) {
					 /*  Normal cylindrical equidistant  */
    case (RECTANGULAR):
      *x = lon - lonc;
      *y = lat - latc;
      return hemisphere;
				     /*  Transverse cylindrical equidistant  */
    case (CASSINI):
      *x = asin (cos_lat * sin (lon - lonc));
      *y = atan2 (tan (lat), cos (lon - lonc)) - latc;
      return hemisphere;
	     /*  Normal cylindrical equivalent - differs from sphere2plane  */
    case (CYLEQA):
      *x = lon - lonc;
      *y = sin_lat - sin_latc;
      return hemisphere;
	      /*  Normal sinusoidal equivalent - differs from sphere2plane  */
    case (SINEQA):
      *x = cos_lat * (lon - lonc);
      *y = lat - latc;
      return hemisphere;
			   /*  Normal Mercator - differs from sphere2plane  */
    case (MERCATOR):
      *x = lon - lonc;
      *y = log (tan (M_PI_4 + 0.5 * lat)) - yc_merc;
      return hemisphere;
  }
						 /*  Azimuthal projections  */
  rm = acos (cos_cang);

  switch (projection) {
    case (POSTEL):
      r = rm;
      break;
    case (GNOMONIC):
      r = tan (rm);
      break;
    case (STEREOGRAPHIC):
      r = 2.0 * tan (0.5 * rm);
      break;
    case (ORTHOGRAPHIC):
      r = sin (rm);
      break;
    case (LAMBERT):
      r = 2.0 * sin (0.5 * rm);
      break;
    default:
      return -1;
  }
  if (rm != 0.) {
    *x = r * cos_lat * sin (lon - lonc) / sin (rm);
    *y = r * (sin_lat * cos_latc - sin_latc * cos_lat_lon) / sin(rm);
  } else {
    *x = 0.;
    *y = 0.;
  }
  return hemisphere;
}

/*
 *  Revision History
 *  v 0.9  94.08.03     Rick Bogart     created this file
 *         94.10.27     R Bogart        reorganized for inclusion in libM
 *         94.11.24     R Bogart        fixed bug in sinusoidal equal-area
 *              mapping and minor bug in cylindrical equal-area mapoing
 *              (latc != 0); added Mercator projection; more comments
 *         94.12.12     R Bogart & Luiz Sa      fixed non-azimuthal projections
 *              in plane2sphere to support center of map at other than L0L0;
 *              removed unnecessary (and incorrect) scaling in azimuthal
 *              mappings
 *  v 1.0  95.08.18     R Bogart & L Sa implemented optimum_scale();
 *	added arguments and code to sphere2img() to deal with elliptic
 *	images and image inversion; additional comments;
 *	minor fix in SINEQA option of plane2sphere(); changed sign on
 *	on position angle used in sphere2img() to conform with standard
 *	usage (positive eastward from north in sky)
 *  96.05.17     R Bogart        fixed bug in longitude calculation of
 *	points "over the pole" for azimuthal projections in plane2sphere()
 *	and corrected bug in sphere2img().
 *  96.06.07	R Bogart	return status of sphere2img indicates
 *	whether mapped point is on far side of globe.
 *  96.12.18	R Bogart	removed debugging prints from function
 *	optimum_scale()
 *  97.01.21	R Bogart	plane2sphere now returns status
 *  98.09.02	R Bogart	added img2sphere
 *  99.02.25	R Bogart	fixed calculated chi in img2sphere to
 *	conform to man page description; plane2sphere now returns 1 if
 *	requested point is not principal value in rectangular, cylindrical
 *	equal-area, sinusoidal equal-area, and Mercator projections
 *  99.12.13	R Bogart	added sphere2plane
 *  00.04.24	R Bogart	modifications to plane2sphere to avoid
 *	occasional roundoff errors leading to undefined results at the
 *	limb
 *  05.01.07	R Bogart	fixed bug in longitude calculation in
 *	plane2sphere() for points more than 90deg from target longitude
 *	of map center
 *  06.11.10	R Bogart	fixed return value of sphere2plane for
 *	cylindrical projection(s); added support for Cassini's projection
 *	(transverse cylindrical equidistant); added support for normal
 *	cylindrical and pseudocylindrical projections to sphere2plane
 *  09.07.02	R Bogart	copied to private location for inclusion in
 *	DRMS modules and standalone code; removed optimum_scale(), pow2scale(),
 *	name2proj(), proj2name()
 */
