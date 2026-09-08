import numpy as np
from astropy.io import fits
from scipy.ndimage import map_coordinates

try:
    from .find_cea_coord import find_cea_coord
    from .img2heliovec import img2heliovec
    from .prep_hd import prep_hd
except ImportError:
    from find_cea_coord import find_cea_coord
    from img2heliovec import img2heliovec
    from prep_hd import prep_hd


def bvec2cea(infile_fld: str, infile_inc: str, infile_azi: str, amb: int = 2,
             infile_disamb: str = None, phi_c: float = None, lambda_c: float = None,
             nx: int = None, ny: int = None, dx: float = None, dy: float = None,
             xyz: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray, fits.Header]:
    """
    Convert FD or cutout vector field to CEA maps.
    Copied/translated from SSWIDL/SDO/HMI/IDL code by Xudong Sun (xudongs@hawaii.edu).
    """
    dtor = np.radians(1.0)

    # Read FITS images and header
    fld, hd = fits.getdata(infile_fld, header=True)
    fld = fld.astype(np.float64)

    inc = fits.getdata(infile_inc).astype(np.float64) * dtor
    azi = fits.getdata(infile_azi).astype(np.float64) * dtor

    amb = int(amb) if amb is not None else 2
    if amb > 2 or amb < 0:
        amb = 2  # default radial acute

    # Disambiguation
    if infile_disamb:
        disamb = fits.getdata(infile_disamb)
        disamb = np.nan_to_num(disamb, nan=0).astype(np.int32)
        if disamb.shape != azi.shape:
            raise ValueError("Disambiguation resolution does not match azimuth")
        disamb = disamb // (2 ** amb)
        idx = np.where(disamb % 2 != 0)
        azi[idx] += np.pi

    # Check input sizes
    if not (fld.shape == inc.shape == azi.shape):
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

    # Compute CEA coordinates (xi, eta in CCD pixels, lat/lon in radians)
    xi, eta, lat, lon = find_cea_coord(hd, phi_c, lambda_c, nx, ny, dx, dy)

    # Magnetic field components in image coordinates
    bx_img = -fld * np.sin(inc) * np.sin(azi)
    by_img = fld * np.sin(inc) * np.cos(azi)
    bz_img = fld * np.cos(inc)

    # Perform bilinear interpolation (map_coordinates takes [row, col] -> [eta, xi])
    valid_coords = np.isfinite(xi) & np.isfinite(eta)
    coords = [np.where(valid_coords, eta, -1.0), np.where(valid_coords, xi, -1.0)]

    bx_map = map_coordinates(bx_img, coords, order=1, mode='constant', cval=np.nan)
    by_map = map_coordinates(by_img, coords, order=1, mode='constant', cval=np.nan)
    bz_map = map_coordinates(bz_img, coords, order=1, mode='constant', cval=np.nan)

    bx_map[~valid_coords] = np.nan
    by_map[~valid_coords] = np.nan
    bz_map[~valid_coords] = np.nan

    # Vector transform to heliographic coordinates
    disk_lonc = 0.0
    disk_latc = hd['CRLT_OBS'] * dtor
    pa = hd['CROTA2'] * (-1.0) * dtor

    bp, bt, br = img2heliovec(bx_map, by_map, bz_map, lon, lat, disk_lonc, disk_latc, pa)

    if not xyz:
        bt *= -1.0

    # Prepare output header
    hd_out = prep_hd(hd, br, phi_c, lambda_c, nx, ny, dx, dy)

    return bp, bt, br, hd_out
