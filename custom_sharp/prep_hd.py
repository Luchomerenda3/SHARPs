from astropy.io import fits


def prep_hd(hd_in, data_out, phi_c, lambda_c, nx, ny, dx, dy):
    """
    Prepare header for CEA field maps.
    Copied/translated from SSWIDL/SDO/HMI/IDL code by Xudong Sun (xudongs@hawaii.edu).

    Arguments:
        hd_in: Input FITS header
        data_out: 2D numpy array of output data (e.g., br)
        phi_c, lambda_c: CEA center longitude and latitude (degrees)
        nx, ny: Dimensions in pixels
        dx, dy: Pixel scale in degrees

    Returns:
        hd_out: Clean astropy.io.fits.Header configured for the CEA projection
    """
    # Create clean header initialized with basic FITS structure
    hd_out = fits.Header()
    if data_out is not None:
        hd_out['SIMPLE'] = True
        hd_out['BITPIX'] = -64
        hd_out['NAXIS'] = 2
        hd_out['NAXIS1'] = int(nx)
        hd_out['NAXIS2'] = int(ny)
    else:
        hd_out['NAXIS1'] = int(nx)
        hd_out['NAXIS2'] = int(ny)

    # Basic telescope and observation parameters
    keys_def = ['TELESCOP', 'INSTRUME', 'WAVELNTH', 'CAMERA']
    for key in keys_def:
        if key in hd_in:
            hd_out[key] = hd_in[key]

    # Time and solar ephemeris metadata
    keys0 = ['DATE', 'DATE_S', 'DATE-OBS', 'T_OBS', 'T_REC', 'TRECEPOC', 'TRECSTEP', 'TRECUNIT', 'HARPNUM']
    keys1 = ['DSUN_OBS', 'DSUN_REF', 'RSUN_REF', 'CRLN_OBS', 'CRLT_OBS', 'CAR_ROT',
             'OBS_VR', 'OBS_VW', 'OBS_VN', 'RSUN_OBS']
    keys2 = ['QUALITY', 'QUAL_S', 'QUALLEV1']

    for key in (keys0 + keys1 + keys2):
        if key in hd_in:
            hd_out[key] = hd_in[key]

    # Set CEA-specific WCS parameters
    hd_out['CUNIT1'] = 'degree'
    hd_out['CUNIT2'] = 'degree'
    hd_out['CRPIX1'] = (nx - 1.0) / 2.0 + 1.0
    hd_out['CRPIX2'] = (ny - 1.0) / 2.0 + 1.0
    hd_out['CRVAL1'] = float(phi_c)
    hd_out['CRVAL2'] = float(lambda_c)
    hd_out['CDELT1'] = float(dx)
    hd_out['CDELT2'] = float(dy)
    hd_out['CTYPE1'] = 'CRLN-CEA'
    hd_out['CTYPE2'] = 'CRLT-CEA'
    hd_out['CROTA2'] = 0.0

    hd_out['WCSNAME'] = 'Carrington Heliographic'
    hd_out['BUNIT'] = 'Mx/cm^2'

    return hd_out
