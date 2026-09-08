import numpy as np

try:
    from .cartography import img2sphere
except ImportError:
    from cartography import img2sphere


def bvec_errorprop(hd, fld, inc, azi, err_fld, err_inc, err_azi, cc_fi, cc_fa, cc_ia):
    """
    Converting vector field and covariance matrix components in field/inclination/azimuth
    into variances of Bp, Bt, Br.
    Based on errorprop.c module in HMI pipeline, originally written by Y. Liu.
    Reference: Eqs (10) (11) of https://arxiv.org/abs/1309.2392
    Copied/translated from SSWIDL/SDO/HMI/IDL code by Xudong Sun (xudongs@hawaii.edu).

    Returns:
        var_bp, var_bt, var_br: 2D arrays of variance matching fld.shape
    """
    dtor = np.radians(1.0)

    # Extract relevant metadata from the header
    crpix1, crpix2 = hd['CRPIX1'], hd['CRPIX2']
    cdelt1, cdelt2 = hd['CDELT1'], hd['CDELT2']
    crval1, crval2 = hd['CRVAL1'], hd['CRVAL2']
    rsun_obs = hd['RSUN_OBS']  # Solar disk radius in arcsec
    crota2 = hd['CROTA2']      # Negative p-angle
    crlt_obs = hd['CRLT_OBS']  # Disk center latitude (degrees)

    # In Python/NumPy, 2D array shape is (nyo, nxo)
    nyo, nxo = fld.shape
    xi_1d = ((np.arange(nxo, dtype=np.float64) + 1.0 - crpix1) * cdelt1 + crval1) / rsun_obs
    eta_1d = ((np.arange(nyo, dtype=np.float64) + 1.0 - crpix2) * cdelt2 + crval2) / rsun_obs
    xi, eta = np.meshgrid(xi_1d, eta_1d)

    latc = crlt_obs * dtor
    lonc = 0.0
    ang_r = (rsun_obs / 3600.0) * dtor
    pa = -crota2 * dtor

    # img2sphere returns: rho, lat, lon, sinlat, coslat, sig, mu, chi
    _, lat, lon, _, _, _, _, _ = img2sphere(xi, eta, ang_r, latc, lonc, pa)

    # Transformation matrix elements (Eq 14)
    pAng = pa
    a11 = -np.sin(latc) * np.sin(pAng) * np.sin(lon - lonc) + np.cos(pAng) * np.cos(lon - lonc)
    a12 = np.sin(latc) * np.cos(pAng) * np.sin(lon - lonc) + np.sin(pAng) * np.cos(lon - lonc)
    a13 = -np.cos(latc) * np.sin(lon - lonc)

    a21 = -np.sin(lat) * (np.sin(latc) * np.sin(pAng) * np.cos(lon - lonc) + np.cos(pAng) * np.sin(lon - lonc)) - np.cos(lat) * np.cos(latc) * np.sin(pAng)
    a22 = np.sin(lat) * (np.sin(latc) * np.cos(pAng) * np.cos(lon - lonc) - np.sin(pAng) * np.sin(lon - lonc)) + np.cos(lat) * np.cos(latc) * np.cos(pAng)
    a23 = -np.cos(latc) * np.sin(lat) * np.cos(lon - lonc) + np.sin(latc) * np.cos(lat)

    a31 = np.cos(lat) * (np.sin(latc) * np.sin(pAng) * np.cos(lon - lonc) + np.cos(pAng) * np.sin(lon - lonc)) - np.sin(lat) * np.cos(latc) * np.sin(pAng)
    a32 = -np.cos(lat) * (np.sin(latc) * np.cos(pAng) * np.cos(lon - lonc) - np.sin(pAng) * np.sin(lon - lonc)) + np.sin(lat) * np.cos(latc) * np.cos(pAng)
    a33 = np.cos(lat) * np.cos(latc) * np.cos(lon - lonc) + np.sin(lat) * np.sin(latc)

    # Sine and cosine calculations
    sin_inc = np.sin(inc)
    cos_inc = np.cos(inc)
    sin_azi = np.sin(azi)
    cos_azi = np.cos(azi)

    # Covariance calculations
    var_fld = err_fld ** 2
    var_inc = err_inc ** 2
    var_azi = err_azi ** 2
    cov_fi = err_fld * err_inc * cc_fi
    cov_fa = err_fld * err_azi * cc_fa
    cov_ia = err_inc * err_azi * cc_ia

    # Partial derivatives
    dBp_dfld = (-a11 * sin_inc * sin_azi + a12 * sin_inc * cos_azi + a13 * cos_inc)
    dBp_dinc = (-a11 * cos_inc * sin_azi + a12 * cos_inc * cos_azi - a13 * sin_inc) * fld
    dBp_dazi = (-a11 * sin_inc * cos_azi - a12 * sin_inc * sin_azi) * fld

    dBt_dfld = (-a21 * sin_inc * sin_azi + a22 * sin_inc * cos_azi + a23 * cos_inc) * (-1.0)
    dBt_dinc = (-a21 * cos_inc * sin_azi + a22 * cos_inc * cos_azi - a23 * sin_inc) * fld * (-1.0)
    dBt_dazi = (-a21 * sin_inc * cos_azi - a22 * sin_inc * sin_azi) * fld * (-1.0)

    dBr_dfld = (-a31 * sin_inc * sin_azi + a32 * sin_inc * cos_azi + a33 * cos_inc)
    dBr_dinc = (-a31 * cos_inc * sin_azi + a32 * cos_inc * cos_azi - a33 * sin_inc) * fld
    dBr_dazi = (-a31 * sin_inc * cos_azi - a32 * sin_inc * sin_azi) * fld

    # Variance calculations
    var_bp = (dBp_dfld**2 * var_fld +
              dBp_dinc**2 * var_inc +
              dBp_dazi**2 * var_azi +
              2.0 * dBp_dfld * dBp_dinc * cov_fi +
              2.0 * dBp_dfld * dBp_dazi * cov_fa +
              2.0 * dBp_dinc * dBp_dazi * cov_ia)

    var_bt = (dBt_dfld**2 * var_fld +
              dBt_dinc**2 * var_inc +
              dBt_dazi**2 * var_azi +
              2.0 * dBt_dfld * dBt_dinc * cov_fi +
              2.0 * dBt_dfld * dBt_dazi * cov_fa +
              2.0 * dBt_dinc * dBt_dazi * cov_ia)

    var_br = (dBr_dfld**2 * var_fld +
              dBr_dinc**2 * var_inc +
              dBr_dazi**2 * var_azi +
              2.0 * dBr_dfld * dBr_dinc * cov_fi +
              2.0 * dBr_dfld * dBr_dazi * cov_fa +
              2.0 * dBr_dinc * dBr_dazi * cov_ia)

    return var_bp, var_bt, var_br
