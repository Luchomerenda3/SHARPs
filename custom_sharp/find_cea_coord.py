import numpy as np

try:
    from .cartography import plane2sphere, sphere2img
except ImportError:
    from cartography import plane2sphere, sphere2img


def find_cea_coord(header, phi_c, lambda_c, nx, ny, dx, dy):
    """
    Convert the cutout index to CCD coordinate (xi, eta).
    Copied/translated from SSWIDL/SDO/HMI/IDL code by Xudong Sun (xudongs@hawaii.edu).

    Arguments:
        header: FITS header (astropy.io.fits.Header or dict)
        phi_c: CEA patch center Carrington longitude (degrees)
        lambda_c: CEA patch center Carrington latitude (degrees)
        nx, ny: CEA patch size in pixels
        dx, dy: Pixel scale in degrees

    Returns:
        xi, eta: CCD coordinates in pixels with respect to patch lower-left (0, 0), shape (ny, nx)
        lat, lon: Heliographic coordinates in radians (Stonyhurst), shape (ny, nx)
    """
    dtor = np.radians(1.0)
    nx, ny = int(nx), int(ny)

    # 1D coordinates in radians
    x_1d = (np.arange(nx, dtype=np.float64) - (nx - 1.0) / 2.0) * dx * dtor
    y_1d = (np.arange(ny, dtype=np.float64) - (ny - 1.0) / 2.0) * dy * dtor
    x, y = np.meshgrid(x_1d, y_1d)  # shape (ny, nx)

    # Relevant ephemeris
    rSun = header['RSUN_OBS'] / header['CDELT1']
    disk_latc = header['CRLT_OBS'] * dtor
    disk_lonc = header['CRLN_OBS'] * dtor
    disk_xc = header['CRPIX1'] - 1.0
    disk_yc = header['CRPIX2'] - 1.0
    pa = -header['CROTA2'] * dtor

    latc = lambda_c * dtor
    lonc = phi_c * dtor - disk_lonc  # Stonyhurst

    # Vectorized coordinate conversions
    lat, lon = plane2sphere(x, y, latc, lonc)
    xi, eta = sphere2img(lat, lon, disk_latc, 0.0, disk_xc, disk_yc, rSun, pa)

    return xi, eta, lat, lon
