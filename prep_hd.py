

def prep_hd(hd_in, data_out, phi_c, lambda_c, nx, ny, dx, dy):

	"""
	; prepare header for CEA field maps
	; modified Jan 31 2019; no long dependent on save file
	"""

	# Create a new FITS header from the input
	hd_out = hd_in.deepcopy()

	# Set basic telescope and observation parameters
	keys_def = ['TELESCOP', 'INSTRUME', 'WAVELNTH', 'CAMERA']
	for key in keys_def:
		hd_out[key] = hd_in[key]

	# Copy time and other metadata
	keys0 = ['DATE', 'DATE_S', 'DATE-OBS', 'T_OBS', 'T_REC', 'TRECEPOC', 'TRECSTEP', 'TRECUNIT', 'HARPNUM']
	keys1 = ['DSUN_OBS', 'DSUN_REF', 'RSUN_REF', 'CRLN_OBS', 'CRLT_OBS', 'CAR_ROT', 'OBS_VR', 'OBS_VW', 'OBS_VN', 'RSUN_OBS']
	keys2 = ['QUALITY', 'QUAL_S', 'QUALLEV1']

	for key in keys0 + keys1 + keys2:
		hd_out[key] = hd_in[key]

	# Set CEA-specific parameters
	hd_out['CUNIT1'] = 'degree'
	hd_out['CUNIT2'] = 'degree'
	hd_out['CRPIX1'] = (nx - 1) / 2. + 1
	hd_out['CRPIX2'] = (ny - 1) / 2. + 1
	hd_out['CRVAL1'] = phi_c
	hd_out['CRVAL2'] = lambda_c
	hd_out['CDELT1'] = dx
	hd_out['CDELT2'] = dy
	hd_out['CTYPE1'] = 'CRLN-CEA'
	hd_out['CTYPE2'] = 'CRLT-CEA'
	hd_out['CROTA2'] = 0.0

	hd_out['WCSNAME'] = 'Carrington Heliographic'
	hd_out['BUNIT'] = 'Mx/cm^2'

	return hd_out
