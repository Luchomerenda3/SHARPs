; Converting vector field and covariance matrix components in field/inclination/azimuth
; Into variances of Bp, Bt, Br
; Based on errorprop.c module in HMI pipeline, originally written by Y. Liu
; No checking of header and array sizes!
; Reference: Eqs (10) (11) of https://arxiv.org/abs/1309.2392
; Written by: Xudong Sun (xudongs@hawaii.edu)

pro bvec_errorprop, hd, fld, inc, azi, err_fld, err_inc, err_azi, cc_fi, cc_fa, cc_ia, $
		var_bp, var_bt, var_br

; Parameters, no error checking

dtor = asin(1.d0)/9.d1
crpix1 = sxpar(hd, 'CRPIX1') & crpix2 = sxpar(hd, 'CRPIX2')
cdelt1 = sxpar(hd, 'CDELT1') & cdelt2 = sxpar(hd, 'CDELT2')
crval1 = sxpar(hd, 'CRVAL1') & crval2 = sxpar(hd, 'CRVAL2')
rsun_obs = sxpar(hd, 'RSUN_OBS')	; solar disk radius in arcsec
crota2 = sxpar(hd, 'CROTA2')		; neg p-angle
crlt_obs =  sxpar(hd, 'CRLT_OBS')	; disk center lat

; Get lon/lat

nxo = (size(fld))[1] & nyo = (size(fld))[2]
xi = dblarr(nxo, nyo) & for i = 0, nxo - 1 do xi[i,*] = ((i + 1 - crpix1) * cdelt1 + crval1) / rsun_obs		; normalized
eta = dblarr(nxo, nyo) & for j = 0, nyo - 1 do eta[*,j] = ((j + 1 - crpix2) * cdelt2 + crval2) / rsun_obs

; lon = dblarr(nxo, nyo) & lat = dblarr(nxo, nyo)
img2sph, xi, eta, lon, lat, $
	lonc=0.d0, latc=crlt_obs*dtor, asd=rsun_obs/3.6d3*dtor, pa=(-1.)*crota2*dtor

; Coefficients. Note per Eq (14) there is
; k11=a31, k12=a32, k13=a33, k21=-a21, k22=-a22, k23=-a23, k31=a11, k32=a12, k33=a13

latc = crlt_obs * dtor
lonc = 0.d0
pAng = (-1.) * crota2 * dtor

a11 = - sin(latc) * sin(pAng) * sin(lon - lonc) + cos(pAng) * cos(lon - lonc)
a12 =  sin(latc) * cos(pAng) * sin(lon - lonc) + sin(pAng) * cos(lon - lonc)
a13 = - cos(latc) * sin(lon - lonc)
a21 = - sin(lat) * (sin(latc) * sin(pAng) * cos(lon - lonc) + cos(pAng) * sin(lon - lonc)) - cos(lat) * cos(latc) * sin(pAng)
a22 =  sin(lat) * (sin(latc) * cos(pAng) * cos(lon - lonc) - sin(pAng) * sin(lon - lonc)) + cos(lat) * cos(latc) * cos(pAng)
a23 = - cos(latc) * sin(lat) * cos(lon - lonc) + sin(latc) * cos(lat)
a31 =  cos(lat) * (sin(latc) * sin(pAng) * cos(lon - lonc) + cos(pAng) * sin(lon - lonc)) - sin(lat) * cos(latc) * sin(pAng)
a32 = - cos(lat) * (sin(latc) * cos(pAng) * cos(lon - lonc) - sin(pAng) * sin(lon - lonc)) + sin(lat) * cos(latc) * cos(pAng)
a33 =  cos(lat) * cos(latc) * cos(lon - lonc) + sin(lat) * sin(latc)

; sine/cosine
sin_inc = sin(inc) & cos_inc = cos(inc)
sin_azi = sin(azi) & cos_azi = cos(azi)

; covariance
var_fld = err_fld * err_fld
var_inc = err_inc * err_inc
var_azi = err_azi * err_azi
cov_fi = err_fld * err_inc * cc_fi
cov_fa = err_fld * err_azi * cc_fa
cov_ia = err_inc * err_azi * cc_ia

; Partial derivatives

dBp_dfld = (- a11 * sin_inc * sin_azi + a12 * sin_inc * cos_azi + a13 * cos_inc)
dBp_dinc = (- a11 * cos_inc * sin_azi + a12 * cos_inc * cos_azi - a13 * sin_inc) * fld
dBp_dazi = (- a11 * sin_inc * cos_azi - a12 * sin_inc * sin_azi) * fld

dBt_dfld = (- a21 * sin_inc * sin_azi + a22 * sin_inc * cos_azi + a23 * cos_inc) * (-1)
dBt_dinc = (- a21 * cos_inc * sin_azi + a22 * cos_inc * cos_azi - a23 * sin_inc) * fld * (-1)
dBt_dazi = (- a21 * sin_inc * cos_azi - a22 * sin_inc * sin_azi) * fld * (-1)

dBr_dfld = (- a31 * sin_inc * sin_azi + a32 * sin_inc * cos_azi + a33 * cos_inc)
dBr_dinc = (- a31 * cos_inc * sin_azi + a32 * cos_inc * cos_azi - a33 * sin_inc) * fld
dBr_dazi = (- a31 * sin_inc * cos_azi - a32 * sin_inc * sin_azi) * fld

; Final

var_bp = dBp_dfld * dBp_dfld * var_fld + dBp_dinc * dBp_dinc * var_inc + dBp_dazi * dBp_dazi * var_azi + $
		 2 * dBp_dfld * dBp_dinc * cov_fi + 2 * dBp_dfld * dBp_dazi * cov_fa + 2 * dBp_dinc * dBp_dazi * cov_ia
		 
var_bt = dBt_dfld * dBt_dfld * var_fld + dBt_dinc * dBt_dinc * var_inc + dBt_dazi * dBt_dazi * var_azi + $
		 2 * dBt_dfld * dBt_dinc * cov_fi + 2 * dBt_dfld * dBt_dazi * cov_fa + 2 * dBt_dinc * dBt_dazi * cov_ia
		 
var_br = dBr_dfld * dBr_dfld * var_fld + dBr_dinc * dBr_dinc * var_inc + dBr_dazi * dBr_dazi * var_azi + $
		 2 * dBr_dfld * dBr_dinc * cov_fi + 2 * dBr_dfld * dBr_dazi * cov_fa + 2 * dBr_dinc * dBr_dazi * cov_ia


end