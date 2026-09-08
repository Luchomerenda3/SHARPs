import numpy as np
from astropy.io import fits
from scipy.ndimage import map_coordinates

try:
    from .bvec_errorprop import bvec_errorprop
    from .find_cea_coord import find_cea_coord
    from .prep_hd import prep_hd
except ImportError:
    from bvec_errorprop import bvec_errorprop
    from find_cea_coord import find_cea_coord
    from prep_hd import prep_hd


def bvecerr2cea(infile_fld: str, infile_inc: str, infile_azi: str,
                infile_err_fld: str, infile_err_inc: str, infile_err_azi: str,
                infile_cc_fld_inc: str, infile_cc_fld_azi: str, infile_cc_inc_azi: str,
                amb: int = 2, phi_c: float = None, lambda_c: float = None,
                nx: int = None, ny: int = None, dx: float = None, dy: float = None,
                infile_disamb: str = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, fits.Header]:
    """
    Convert FD or cutout vector field UNCERTAINTIES to CEA maps.
    Copied/translated from SSWIDL/SDO/HMI/IDL code by Xudong Sun (xudongs@hawaii.edu).
    """
    dtor = np.radians(1.0)

    # Read data and header
    fld, hd = fits.getdata(infile_fld, header=True)
    fld = fld.astype(np.float64)

    inc = fits.getdata(infile_inc).astype(np.float64) * dtor
    azi = fits.getdata(infile_azi).astype(np.float64) * dtor

    err_fld = fits.getdata(infile_err_fld).astype(np.float64)
    err_inc = fits.getdata(infile_err_inc).astype(np.float64) * dtor
    err_azi = fits.getdata(infile_err_azi).astype(np.float64) * dtor

    cc_fi = fits.getdata(infile_cc_fld_inc).astype(np.float64)
    cc_fa = fits.getdata(infile_cc_fld_azi).astype(np.float64)
    cc_ia = fits.getdata(infile_cc_inc_azi).astype(np.float64)

    # Disambiguation
    if infile_disamb:
        disamb = fits.getdata(infile_disamb)
        disamb = np.nan_to_num(disamb, nan=0).astype(np.int32)
        if disamb.shape != azi.shape:
            raise ValueError("Disambiguation resolution does not match azimuth")
        amb = 2 if (amb is None or not (0 <= amb <= 2)) else int(amb)
        disamb = disamb // (2 ** amb)
        idx = np.where(disamb % 2 != 0)
        azi[idx] += np.pi

    # Ensure input images have matching sizes
    images = [inc, azi, err_fld, err_inc, err_azi, cc_fi, cc_fa, cc_ia]
    if not all(x.shape == fld.shape for x in images):
        raise ValueError("Input image sizes do not match")

    # Check required header keywords
    req_keys = ['CRLT_OBS', 'CRLN_OBS', 'CROTA2', 'RSUN_OBS', 'CDELT1', 'CRPIX1', 'CRPIX2']
    for k in req_keys:
        if k not in hd:
            raise KeyError(f"Header keyword {k} missing")

    # Determine default patch parameters
    has_lon_bounds = ('LONDTMAX' in hd) and ('LONDTMIN' in hd)
    has_lat_bounds = ('LATDTMAX' in hd) and ('LATDTMIN' in hd)

    if phi_c is None:
        if not has_lon_bounds:
            raise ValueError("No x center (LONDTMAX/LONDTMIN missing)")
        phi_c = (hd['LONDTMAX'] + hd['LONDTMIN']) / 2.0 + hd['CRLN_OBS']

    if lambda_c is None:
        if not has_lat_bounds:
            raise ValueError("No y center (LATDTMAX/LATDTMIN missing)")
        lambda_c = (hd['LATDTMAX'] + hd['LATDTMIN']) / 2.0

    dx = 0.03 if dx is None else float(abs(dx))
    dy = 0.03 if dy is None else float(abs(dy))

    if nx is None:
        if not has_lon_bounds:
            raise ValueError("No x dimension (LONDTMAX/LONDTMIN missing)")
        nx = int(np.round(np.round((hd['LONDTMAX'] - hd['LONDTMIN']) * 1e3) / 1e3 / dx))

    if ny is None:
        if not has_lat_bounds:
            raise ValueError("No y dimension (LATDTMAX/LATDTMIN missing)")
        ny = int(np.round(np.round((hd['LATDTMAX'] - hd['LATDTMIN']) * 1e3) / 1e3 / dy))

    # Error propagation to obtain variance of Bp, Bt, Br
    var_bp, var_bt, var_br = bvec_errorprop(
        hd, fld, inc, azi, err_fld, err_inc, err_azi, cc_fi, cc_fa, cc_ia
    )

    # Convert to CEA coordinates
    xi, eta, lat, lon = find_cea_coord(hd, phi_c, lambda_c, nx, ny, dx, dy)

    # Sampling to CEA grid
    valid_coords = np.isfinite(xi) & np.isfinite(eta)
    coords = [np.where(valid_coords, eta, -1.0), np.where(valid_coords, xi, -1.0)]

    var_bp_map = map_coordinates(var_bp, coords, order=1, mode='constant', cval=np.nan)
    var_bt_map = map_coordinates(var_bt, coords, order=1, mode='constant', cval=np.nan)
    var_br_map = map_coordinates(var_br, coords, order=1, mode='constant', cval=np.nan)

    var_bp_map[~valid_coords] = np.nan
    var_bt_map[~valid_coords] = np.nan
    var_br_map[~valid_coords] = np.nan

    # Compute standard deviation as final error
    err_bp = np.sqrt(np.maximum(0.0, var_bp_map))
    err_bt = np.sqrt(np.maximum(0.0, var_bt_map))
    err_br = np.sqrt(np.maximum(0.0, var_br_map))

    # Prepare output header
    hd_out = prep_hd(hd, err_br, phi_c, lambda_c, nx, ny, dx, dy)

    return err_bp, err_bt, err_br, hd_out
