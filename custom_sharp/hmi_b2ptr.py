import numpy as np

try:
    from .cartography import img2sphere
except ImportError:
    from cartography import img2sphere


def hmi_b2ptr(index, bvec, return_lonlat=False):
    """
    Convert HMI vector field in native components
    (field, inclination, azimuth w.r.t. plane of sky)
    into spherical coordinate components
    (zonal B_p, meridional B_t, radial B_r).

    For details, see Sun, 2013, ArXiv: 1309.2392 (http://arxiv.org/abs/1309.2392)
    Copied/translated from SSWIDL/SDO/HMI/IDL code by Xudong Sun (xudongs@hawaii.edu).

    Arguments:
        index: FITS header (dict or astropy.io.fits.Header)
        bvec: 3D array of shape (3, ny, nx) or (ny, nx, 3) containing [field, inc, azi]
              Field in Gauss, inclination in deg, azimuth in deg.
        return_lonlat: If True, returns tuple (bptr, lonlat).

    Returns:
        bptr: 3D array matching bvec shape containing [Bp, Bt, Br] in Gauss.
              Bp is positive pointing west; Bt is positive pointing south.
        lonlat (optional): 3D array containing [lon, lat] in degrees.
    """
    dtor = np.radians(1.0)
    nx = int(index['NAXIS1'])
    ny = int(index['NAXIS2'])

    # Support both (3, ny, nx) and (ny, nx, 3)
    channel_first = (bvec.shape[0] == 3)
    if channel_first:
        if bvec.shape != (3, ny, nx):
            raise ValueError(f"Dimension of bvec incorrect: expected (3, {ny}, {nx}), got {bvec.shape}")
        field = bvec[0, :, :].astype(np.float64)
        gamma = bvec[1, :, :].astype(np.float64) * dtor
        psi = bvec[2, :, :].astype(np.float64) * dtor
    else:
        if bvec.shape != (ny, nx, 3):
            raise ValueError(f"Dimension of bvec incorrect: expected ({ny}, {nx}, 3), got {bvec.shape}")
        field = bvec[:, :, 0].astype(np.float64)
        gamma = bvec[:, :, 1].astype(np.float64) * dtor
        psi = bvec[:, :, 2].astype(np.float64) * dtor

    # Convert bvec to B_xi, B_eta, B_zeta as defined in Eq (1) in Sun (2013)
    b_xi = -field * np.sin(gamma) * np.sin(psi)
    b_eta = field * np.sin(gamma) * np.cos(psi)
    b_zeta = field * np.cos(gamma)

    # Compute Stonyhurst heliographic coordinates (phi, lambda) using img2sphere
    crpix1, crpix2 = index['CRPIX1'], index['CRPIX2']
    cdelt1, cdelt2 = index['CDELT1'], index['CDELT2']
    crval1, crval2 = index.get('CRVAL1', 0.0), index.get('CRVAL2', 0.0)
    rsun_obs = index['RSUN_OBS']
    crota2 = index['CROTA2']
    crlt_obs = index['CRLT_OBS']

    xi_1d = ((np.arange(nx, dtype=np.float64) + 1.0 - crpix1) * cdelt1 + crval1) / rsun_obs
    eta_1d = ((np.arange(ny, dtype=np.float64) + 1.0 - crpix2) * cdelt2 + crval2) / rsun_obs
    xi, eta = np.meshgrid(xi_1d, eta_1d)

    latc = crlt_obs * dtor
    lonc = 0.0
    ang_r = (rsun_obs / 3600.0) * dtor
    pa = -crota2 * dtor

    # img2sphere returns: rho, lat, lon, sinlat, coslat, sig, mu, chi
    # In img2sphere, lat is lambda, lon is phi (both in radians)
    _, lam, phi, _, _, _, _, _ = img2sphere(xi, eta, ang_r, latc, lonc, pa)

    # Transformation matrix according to Eq (1) in Gary & Hagyard (1990)
    # and Eqs (7)(8) in Sun (2013)
    b = latc
    p = pa

    sinb, cosb = np.sin(b), np.cos(b)
    sinp, cosp = np.sin(p), np.cos(p)
    sinphi, cosphi = np.sin(phi), np.cos(phi)
    sinlam, coslam = np.sin(lam), np.cos(lam)

    k11 = coslam * (sinb * sinp * cosphi + cosp * sinphi) - sinlam * cosb * sinp
    k12 = -coslam * (sinb * cosp * cosphi - sinp * sinphi) + sinlam * cosb * cosp
    k13 = coslam * cosb * cosphi + sinlam * sinb
    k21 = sinlam * (sinb * sinp * cosphi + cosp * sinphi) + coslam * cosb * sinp
    k22 = -sinlam * (sinb * cosp * cosphi - sinp * sinphi) - coslam * cosb * cosp
    k23 = sinlam * cosb * cosphi - coslam * sinb
    k31 = -sinb * sinp * sinphi + cosp * cosphi
    k32 = sinb * cosp * sinphi + sinp * cosphi
    k33 = -cosb * sinphi

    # Output: (Bp, Bt, Br) identical to (Bxh, -Byh, Bzh)
    bp = k31 * b_xi + k32 * b_eta + k33 * b_zeta
    bt = k21 * b_xi + k22 * b_eta + k23 * b_zeta
    br = k11 * b_xi + k12 * b_eta + k13 * b_zeta

    if channel_first:
        bptr = np.stack([bp, bt, br], axis=0)
    else:
        bptr = np.stack([bp, bt, br], axis=-1)

    if return_lonlat:
        if channel_first:
            lonlat = np.stack([np.degrees(phi), np.degrees(lam)], axis=0)
        else:
            lonlat = np.stack([np.degrees(phi), np.degrees(lam)], axis=-1)
        return bptr, lonlat

    return bptr
