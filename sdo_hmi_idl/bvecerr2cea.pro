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

pro bvecerr2cea, infile_fld, infile_inc, infile_azi, $
			infile_err_fld, infile_err_inc, infile_err_azi, $
			infile_cc_fld_inc, infile_cc_fld_azi, infile_cc_inc_azi, $
			err_bp, err_bt, err_br, hd_out, $
			infile_disamb=infile_disamb, amb=amb, $
			phi_c=phi_c, lambda_c=lambda_c, nx=nx, ny=ny, dx=dx, dy=dy

; Read data

dtor = asin(1.d0)/9.d1
fld = double(fitsio_read_image(infile_fld, hd))
inc = double(fitsio_read_image(infile_inc)) * dtor
azi = double(fitsio_read_image(infile_azi)) * dtor

err_fld = double(fitsio_read_image(infile_err_fld))			; sqrt of variance
err_inc = double(fitsio_read_image(infile_err_inc)) * dtor
err_azi = double(fitsio_read_image(infile_err_azi)) * dtor

cc_fi = double(fitsio_read_image(infile_cc_fld_inc))		; correlation coefficient
cc_fa = double(fitsio_read_image(infile_cc_fld_azi))
cc_ia = double(fitsio_read_image(infile_cc_inc_azi))

; Check file consitancy

if (keyword_set(infile_disamb)) then begin			; disambiguation
	disamb = fitsio_read_image(infile_disamb)
	sz = size(disamb) & sz_a = size(azi)
	if (sz[1] ne sz_a[1] or sz[2] ne sz_a[2]) then begin
		printf, 'disambiguation resolution does not match azimuth'
		return
	endif
	if (not keyword_set(amb)) then begin
		amb = 2
	endif else begin
		amb = fix(amb)
		if (amb gt 2 or amb lt 0) then amb = 2		; default radial acute
	endelse
	disamb = disamb / (2 ^ amb)
	idx = where(disamb ne 0, cc)
	if (cc ne 0) then azi[idx] += !pi
endif


sz_f = size(fld) & sz_i = size(inc) & sz_a = size(azi)
sz_e_f = size(err_fld) & sz_e_i = size(err_inc) & sz_e_a = size(err_azi)
sz_c_fi = size(cc_fi) & sz_c_fa = size(cc_fa) & sz_c_ia = size(cc_ia)
nxo = sz_f[1] & nyo = sz_f[2]
if (nxo ne sz_i[1] or nyo ne sz_i[2] or $
	nxo ne sz_a[1] or nyo ne sz_a[2] or $
	nxo ne sz_e_f[1] or nyo ne sz_e_f[2] or $
	nxo ne sz_e_i[1] or nyo ne sz_e_i[2] or $
	nxo ne sz_e_a[1] or nyo ne sz_e_a[2] or $
	nxo ne sz_c_fi[1] or nyo ne sz_c_fi[2] or $
	nxo ne sz_c_fa[1] or nyo ne sz_c_fa[2] or $
	nxo ne sz_c_ia[1] or nyo ne sz_c_ia[2]) then begin
	print, 'input image size do not match'
	return
endif

; Check requested output parameters

keys = ['crlt_obs','crln_obs','crota2','rsun_obs','cdelt1','crpix1','crpix2']
nkeys = n_elements(keys)

have_keys = 1
for i = 0, nkeys - 1 do begin
	key_val = sxpar(hd, keys[i], count=ct)
	if (ct eq 0) then $
		print, 'Keyword '+keys[i]+' missing'
	have_keys *= ct
endfor

if (have_keys eq 0) then return

maxlon = sxpar(hd, 'LONDTMAX', count=cln1)
minlon = sxpar(hd, 'LONDTMIN', count=cln0)
maxlat = sxpar(hd, 'LATDTMAX', count=clt1)
minlat = sxpar(hd, 'LATDTMIN', count=clt0)
	
if (not keyword_set(phi_c)) then begin
	if (cln1 eq 0 or cln0 eq 0) then begin
		print, 'No x center' & return
	endif
	phi_c = (maxlon + minlon) / 2.d0 + sxpar(hd, 'CRLN_OBS')
endif

if (not keyword_set(lambda_c)) then begin
	if (clt1 eq 0 or clt0 eq 0) then begin
		print, 'No y center' & return
	endif
	lambda_c = (maxlat + minlat) / 2.d0
endif

if (not keyword_set(dx)) then dx = 3.d-2 else dx = double(abs(dx))
if (not keyword_set(dy)) then dy = 3.d-2 else dy = double(abs(dy))

if (not keyword_set(nx)) then begin
	if (cln1 eq 0 or cln0 eq 0) then begin
		print, 'No x dimension' & return
	endif
	nx = round(round((maxlon - minlon) * 1.d3) / 1.d3 / dx)
endif

if (not keyword_set(ny)) then begin
	if (clt1 eq 0 or clt0 eq 0) then begin
		print, 'No y dimension' & return
	endif
	ny = round(round((maxlat - minlat) * 1.d3) / 1.d3 / dy)
endif

; Get variance of Bp, Bt, Br of input

var_bp = dblarr(nxo, nyo)
var_bt = dblarr(nxo, nyo)
var_br = dblarr(nxo, nyo)
bvec_errorprop, hd, fld, inc, azi, err_fld, err_inc, err_azi, cc_fi, cc_fa, cc_ia, $
		var_bp, var_bt, var_br

; Find coordinate of CEA pixel in image

xi = dblarr(nx, ny)		; in pixel wrt lower left of patch (0,0)
eta = dblarr(nx, ny)
lat = dblarr(nx, ny)
lon = dblarr(nx, ny)

find_cea_coord, hd, xi, eta, lat, lon, $
			phi_c, lambda_c, nx, ny, dx, dy		; get xi, eta

; Perform sampling

var_bp_map = interpolate(var_bp, xi, eta, /double, missing=!values.d_nan)
var_bt_map = interpolate(var_bt, xi, eta, /double, missing=!values.d_nan)
;var_br_map = interpolate(var_br, round(xi), round(eta), /double, missing=!values.d_nan)		; near neighbor in HMI pipeline
var_br_map = interpolate(var_br, xi, eta, /double, missing=!values.d_nan)

; Final

err_bp = sqrt(var_bp_map)
err_bt = sqrt(var_bt_map)
err_br = sqrt(var_br_map)

; Prepare header

prep_hd, hd, err_br, hd_out, phi_c, lambda_c, nx, ny, dx, dy

end