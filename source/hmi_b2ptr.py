import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

def hmi_b2ptr(index, bvec, lonlat=None):
    """
    ;	Convert HMI vector field in native components
    ;	(field, inclination, azimuth w.r.t. plane of sky)
    ;	into spherical coordinate components
    ;	(zonal B_p, meridional B_t, radial B_r)
    ;	For details, see
    ;	Sun, 2013, ArXiv, 1309.2392 (http://arxiv.org/abs/1309.2392)
    ; SAMPLE CALLS:
    ;	IDL> files = ['hmi.sharp_720s.377.20110215_000000_TAI.field.fits', $
    ;	IDL> 		  'hmi.sharp_720s.377.20110215_000000_TAI.inclination.fits', $
    ;	IDL> 		  'hmi.sharp_720s.377.20110215_000000_TAI.azimuth.fits']
    ;	IDL> read_sdo, files, index, data
    ;	IDL> hmi_b2rtp, index[0], data, bptr, lonlat=lonlat
    ; INPUT:
    ;	index:	Index structure
    ;	bvec:	Three dimensional array [nx,ny,3], for three images: field (G),
    ;			inclination (deg) and azimuth (deg) arrays. Inclination is defined 0
    ;			perpendicular out of the plane-of-sky (POS) and !pi into the POS.
    ;			Azimuth is 0 in +y CCD direction and increase CCW
    ; OUTPUT:
    ;	bptr:	Three dimensional array [nx,ny,3], for three images: Bp, Bt, Br (G)
    ;			Bp is positive when pointing west; Bt is positive when pointing south
    ; OPTIONAL OUTPUT:
    ;	lonlat:	Three dimensional array [nx,ny,2], for two images:
    ;			Stonyhurst longitude and latitude
    ; HISTORY:
    ;   2014.02.01 - Xudong Sun (xudongs@sun.stanford.edu)
    ; NOTE:
    ;	Written for HMI full disk and SHARP data, header needs to conform
    ;	WCS standard. Minimal check implemented so far
    ;	For full disk images, large memory is needed
    ;	Note the output retains the p-angle of bvec
    ;	The sign of the field vector is independent of the image orientation
    ;	i.e. if Bt is positive (southward) and the image is upside-down (p=180),
    ;	it remains positive (southward) when the image is rotated by 180 deg
    ;-
    """
    # Check dimensions
    if bvec.shape != (3, index['NAXIS1'], index['NAXIS2']):
        print('Dimension of bvec incorrect')
        return

    # Convert bvec to B_xi, B_eta, B_zeta as defined in Eq (1) in Sun (2013)
    dtor = np.radians(1)
    field = bvec[0, :, :]
    gamma = bvec[1, :, :] * dtor
    psi = bvec[2, :, :] * dtor

    b_xi = -field * np.sin(gamma) * np.sin(psi)
    b_eta = field * np.sin(gamma) * np.cos(psi)
    b_zeta = field * np.cos(gamma)

    # WCS conversion
    wcs = WCS(index)
    # Assuming coord is obtained or computed earlier
    # Use wcs.world_to_pixel or a similar method to get phi and lambda from coord

    # Get Stonyhurst longitude/latitude
    phi, lambda_ = wcs.all_world2pix(lonlat[0], lonlat[1], 0)  # 0 for origin

    # Get matrix to convert, according to Eq (1) in Gary & Hagyard (1990)
    b = index['CRLT_OBS'] * dtor
    p = -index['CROTA2'] * dtor

    sinb, cosb = np.sin(b), np.cos(b)
    sinp, cosp = np.sin(p), np.cos(p)
    sinphi, cosphi = np.sin(phi * dtor), np.cos(phi * dtor)
    sinlam, coslam = np.sin(lambda_ * dtor), np.cos(lambda_ * dtor)

    k11 = coslam * (sinb * sinp * cosphi + cosp * sinphi) - sinlam * cosb * sinp
    k12 = -coslam * (sinb * cosp * cosphi - sinp * sinphi) + sinlam * cosb * cosp
    k13 = coslam * cosb * cosphi + sinlam * sinb
    k21 = sinlam * (sinb * sinp * cosphi + cosp * sinphi) + coslam * cosb * sinp
    k22 = -sinlam * (sinb * cosp * cosphi - sinp * sinphi) - coslam * cosb * cosp
    k23 = sinlam * cosb * cosphi - coslam * sinb
    k31 = -sinb * sinp * sinphi + cosp * cosphi
    k32 = sinb * cosp * sinphi + sinp * cosphi
    k33 = -cosb * sinphi

    # Calculate the output (Bp, Bt, Br)
    bptr = np.zeros_like(bvec)  # Assuming bvec has the shape (3, nx, ny)

    bptr[0, :, :] = k31 * b_xi + k32 * b_eta + k33 * b_zeta
    bptr[1, :, :] = k21 * b_xi + k22 * b_eta + k23 * b_zeta
    bptr[2, :, :] = k11 * b_xi + k12 * b_eta + k13 * b_zeta

    return bptr
