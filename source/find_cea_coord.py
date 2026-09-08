import numpy as np

from cartography import plane2sphere, sphere2img


def find_cea_coord(header, phi_c, lambda_c, nx, ny, dx, dy):

	"""
	; Convert the cutout index to CCD coordinate (xi,eta)
	; Input: index, CEA patch center Carrington coordinate [phi_c,lambda_c], patch size [nx,ny], [dx, dy]
	; Output: images xi and eta for CCD coordinate, lat and lon (Stonyhurst)
	"""

	# Conversion factor from degrees to radians
	dtor = np.radians(1)

	# Ensure nx and ny are integers
	nx, ny = int(nx), int(ny)

	# Create arrays of CEA coordinates
	x = (np.arange(nx) - (nx - 1) / 2) * dx * dtor
	y = (np.arange(ny) - (ny - 1) / 2) * dy * dtor
	x, y = np.meshgrid(x, y)

	# Relevant ephemeris
	rSun = header['RSUN_OBS'] / header['CDELT1']
	disk_latc = header['CRLT_OBS'] * dtor
	disk_lonc = header['CRLN_OBS'] * dtor
	disk_xc = header['CRPIX1'] - 1
	disk_yc = header['CRPIX2'] - 1
	pa = -header['CROTA2'] * dtor

	latc = lambda_c * dtor
	lonc = phi_c * dtor - disk_lonc

	# Convert coordinate (assuming plane2sphere and sphere2img functions are defined)
	lat = np.zeros((nx, ny))
	lon = np.zeros((nx, ny))
	xi = np.zeros((nx, ny))
	eta = np.zeros((nx, ny))

	# TODO: check plane2sphere & sphere2img
	for i in range(nx):
		for j in range(ny):
			lat0, lon0 = plane2sphere(x[i, j], y[i, j], latc, lonc)
			lat[i, j] = lat0
			lon[i, j] = lon0

			xi0, eta0 = sphere2img(lat[i, j], lon[i, j], disk_latc, 0, disk_xc, disk_yc, rSun, pa)
			xi[i, j] = xi0
			eta[i, j] = eta0

	return xi, eta, lat, lon

