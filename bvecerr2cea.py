import numpy as np
from astropy.io import fits
from .bvec_errorprop import bvec_errorprop
from .find_cea_coord import find_cea_coord
from .prep_hd import prep_hd

def bvecerr2cea(infile_fld, infile_inc, infile_azi,
                infile_err_fld, infile_err_inc, infile_err_azi,
                infile_cc_fld_inc, infile_cc_fld_azi, infile_cc_inc_azi,
                amb, phi_c=None, lambda_c=None, nx=None, ny=None, dx=None, dy=None, infile_disamb=None):

    """
    ; This module converts FD or cutout vector field UNCERTAINTIES to CEA maps
    ; Input1: File names of field, inclination, azimuth
    ; Input2: File names of uncertainties of field, inclination, azimuth (sqrt of variance)
    ; Input3: File names of correlation coefficient of field/inclination, field/azimuth, inclination/azimuth (covariance/variance)
    ; Output: Maps of Bp, Bt, Br uncertainty and FITS header
    ; Optional input: center coordinate, image size, pixel size
    ;		(if not provided, compute from cutout header;
    ;		 if there is no info of patches, as in FD, the module fails)
    ; Example: creating a 600x600 CEA maps centered at Carrington lon 170, lat 13
    ; bvec2cea, 'hmi.B_720s.20160524_140000_TAI.field.fits', $
    ; 			'hmi.B_720s.20160524_140000_TAI.inclination.fits', $
    ; 			'hmi.B_720s.20160524_140000_TAI.azimuth.fits', $
    ;			'hmi.B_720s.20160524_140000_TAI.field_err.fits', $
    ; 			'hmi.B_720s.20160524_140000_TAI.inclination_err.fits', $
    ; 			'hmi.B_720s.20160524_140000_TAI.azimuth_err.fits', $
    ;			'hmi.B_720s.20160524_140000_TAI.field_inclination_err.fits', $
    ; 			'hmi.B_720s.20160524_140000_TAI.field_az_err.fits', $
    ; 			'hmi.B_720s.20160524_140000_TAI.inclin_azimuth_err.fits', $
    ; 			err_bp, err_bt, err_br, hd_out, $
    ; 			infile_disamb='hmi.B_720s.20160524_140000_TAI.disambig.fits', $
    ; 			phi_c=170, lambda_c=13, nx=600, ny=600
    ; writefits, 'br_err.fits', br_err, hd_out
    ; Xudong Sun (xudongs@hawaii.edu): Feb 01 2019
    """

    # Conversion factor from degrees to radians
    dtor = np.radians(1)

    # Read data
    with fits.open(infile_fld) as hdul:
        fld, hd = hdul[0].data, hdul[0].header
    inc = fits.getdata(infile_inc) * dtor
    azi = fits.getdata(infile_azi) * dtor

    err_fld = fits.getdata(infile_err_fld)
    err_inc = fits.getdata(infile_err_inc) * dtor
    err_azi = fits.getdata(infile_err_azi) * dtor

    cc_fi = fits.getdata(infile_cc_fld_inc)
    cc_fa = fits.getdata(infile_cc_fld_azi)
    cc_ia = fits.getdata(infile_cc_inc_azi)

    # Disambiguation and check file consistency
    if infile_disamb:
        disamb = fits.getdata(infile_disamb)
        if disamb.shape != azi.shape:
            print('Disambiguation resolution does not match azimuth')
            return
        amb = 2 if amb is None or not 0 <= amb <= 2 else amb
        disamb = disamb // (2 ** amb)
        idx = np.where(disamb != 0)
        azi[idx] += np.pi

    # Ensure input images have matching sizes
    if not all(x.shape == fld.shape for x in [inc, azi, err_fld, err_inc, err_azi, cc_fi, cc_fa, cc_ia]):
        print('Input image sizes do not match')
        return
    # Set default parameters based on FITS header or predefined values
    phi_c = (phi_c if phi_c is not None else
             (hd['LONDTMAX'] + hd['LONDTMIN']) / 2 + hd['CRLN_OBS'])
    lambda_c = lambda_c if lambda_c is not None else (hd['LATDTMAX'] + hd['LATDTMIN']) / 2
    dx = dx if dx is not None else 0.03
    dy = dy if dy is not None else 0.03
    nx = nx if nx is not None else int(np.round((hd['LONDTMAX'] - hd['LONDTMIN']) / dx))
    ny = ny if ny is not None else int(np.round((hd['LATDTMAX'] - hd['LATDTMIN']) / dy))

    # Error propagation to obtain variance of Bp, Bt, Br
    var_bp, var_bt, var_br = np.zeros(fld.shape), np.zeros(fld.shape), np.zeros(fld.shape)
    bvec_errorprop(hd, fld, inc, azi, err_fld, err_inc, err_azi, cc_fi, cc_fa, cc_ia, var_bp, var_bt, var_br)

    # Convert to CEA coordinates
    xi, eta, lat, lon = find_cea_coord(hd, phi_c, lambda_c, nx, ny, dx, dy)

    # Sampling to CEA grid
    var_bp_map = interpolate(var_bp, xi, eta, method='linear', bounds_error=False, fill_value=np.nan)
    var_bt_map = interpolate(var_bt, xi, eta, method='linear', bounds_error=False, fill_value=np.nan)
    var_br_map = interpolate(var_br, xi, eta, method='linear', bounds_error=False, fill_value=np.nan)

    # Compute standard deviation as final error
    err_bp = np.sqrt(var_bp_map)
    err_bt = np.sqrt(var_bt_map)
    err_br = np.sqrt(var_br_map)

    # Prepare output header
    hd_out = prep_hd(hd, err_br, phi_c, lambda_c, nx, ny, dx, dy)

    return err_bp, err_bt, err_br, hd_out
