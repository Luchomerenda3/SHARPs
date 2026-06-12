from img2heliovec import img2heliovec
import numpy as np
from astropy.io import fits
from scipy.interpolate import griddata
from astropy.wcs import WCS
from .prep_hd import prep_hd

def bvec2cea(infile_fld, infile_inc, infile_azi, amb, infile_disamb=None, phi_c=None, lambda_c=None, nx=None, ny=None, dx=None, dy=None, xyz=False):

    # Read FITS files
    fld, fld_hdr = fits.getdata(infile_fld, header=True)
    dtor = np.radians(1)  # degrees to radians conversion factor
    inc = fits.getdata(infile_inc) * dtor
    azi = fits.getdata(infile_azi) * dtor

    # Apply disambiguation if necessary
    if infile_disamb:
        disamb = fits.getdata(infile_disamb).astype(np.int32)

        if disamb.shape != azi.shape:
            raise ValueError('Disambiguation resolution does not match azimuth')

        disamb = disamb // (2 ** amb)
        idx = np.where(disamb % 2 != 0)
        azi[idx] += np.pi  # Adjust azimuth based on disambiguation

    # Set default values for parameters if not provided
    if phi_c is None or lambda_c is None:
        phi_c, lambda_c = fld_hdr['CRVAL1'], fld_hdr['CRVAL2']
    if nx is None or ny is None:
        nx, ny = fld_hdr['NAXIS1'], fld_hdr['NAXIS2']
    if dx is None or dy is None:
        dx, dy = np.abs(fld_hdr['CDELT1']), np.abs(fld_hdr['CDELT2'])

    # WCS for coordinate transformation
    wcs = WCS(fld_hdr)
    x, y = np.meshgrid(np.arange(nx), np.arange(ny))
    lon, lat = wcs.all_pix2world(x, y, 0)

    # Calculate magnetic field components in image coordinates
    bx_img = -fld * np.sin(inc) * np.sin(azi)
    by_img = fld * np.sin(inc) * np.cos(azi)
    bz_img = fld * np.cos(inc)

    # Interpolate magnetic field components onto the heliographic grid
    grid_lon, grid_lat = np.meshgrid(np.linspace(lon.min(), lon.max(), nx),
                                     np.linspace(lat.min(), lat.max(), ny))
    bx_map = griddata((lon.ravel(), lat.ravel()), bx_img.ravel(), (grid_lon, grid_lat), method='cubic')
    by_map = griddata((lon.ravel(), lat.ravel()), by_img.ravel(), (grid_lon, grid_lat), method='cubic')
    bz_map = griddata((lon.ravel(), lat.ravel()), bz_img.ravel(), (grid_lon, grid_lat), method='cubic')

    # Transformation to heliographic coordinates
    disk_lonc = 0
    disk_latc = np.radians(fld_hdr.get('CRLT_OBS', 0))
    pa = np.radians(fld_hdr.get('CROTA2', 0) * -1)
    bp, bt, br = img2heliovec(bx_map, by_map, bz_map, grid_lon, grid_lat, disk_lonc, disk_latc, pa)

    if not xyz:
        bt *= -1

    # Prepare the output FITS header
    # Assuming prep_hd function will populate header based on requirements
    hd_out = prep_hd(fld_hdr, br, phi_c, lambda_c, nx, ny, dx, dy)

    return bp, bt, br, hd_out
