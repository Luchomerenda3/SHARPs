;+
;
; Convert image coordinate (xi, eta) into Stonyhurst (lon,lat)
; Adapted from R. Bogart's cartography.c by X. Sun
; Can process whole array now
;
; Input: image coord xi, eta in unit of apparent solar radius
; Output: Stonyhurst lon, lat in radian, mu (cosine between point:obs and local normal)
; Optional input: disk center lonc, latc, apparent solar radius asd, p-angle pa
; Optional output: rho (angle point:sun center:obs), sig (angle point:obs:sun center),
;					mu (cosine between point:obs and local normal)
;					chi (position angle on image measured westward from north)

pro img2sph, xi, eta, lon, lat, $
		lonc=lonc, latc=latc, asd=asd, pa=pa, $
		rho=rho, sig=sig, mu=mu, chi=chi

if not keyword_set(lonc) then lonc = 0.d0
if not keyword_set(latc) then latc = 0.d0
if not keyword_set(asd) then asd = 4.7026928d-3			; 970 arcsec
if not keyword_set(pa) then pa = 0.d0

;lon = !values.f_nan & lat = !values.f_nan

r = double(sqrt(xi^2+eta^2))
idx = where(r le 0. or r ge 1., count)
if (count ne 0) then r[idx] =  !values.d_nan
;if (r le 0. or r ge 1.) then return
;help, r
chi = atan(xi, eta) + pa
;
idx = where(chi gt (2.*!pi), count)
if (count ne 0) then begin
	for i = 0, count - 1 do $
		while (chi[idx[i]] gt 2*!pi) do chi[idx[i]] -= (2*!pi)
endif
;
idx = where(chi lt 0., count)
if (count ne 0) then begin
	for i = 0, count - 1 do $
		while (chi[idx[i]] lt 0.) do chi[idx[i]] += (2*!pi)
endif
;

sig = atan(r * tan(asd))
rho = asin(sin(sig) / sin(asd)) - sig
idx = where(sig gt asd, count)
if (count ne 0) then sig[idx] = !values.d_nan
;if (sig gt asd) then return
mu = cos(rho + sig)
;help, mu

sinr = sin(rho) & cosr = cos(rho)
sinlat = sin(latc) * cosr + cos(latc) * sinr * cos(chi)
coslat = sqrt(1. - sinlat * sinlat)

lat = asin(sinlat)
sinlon = sinr * sin(chi) / cos(lat)
lon = asin(sinlon)
idx = where(cosr lt (sin(lat) * sin(latc)), count)
if (count ne 0) then lon[idx] = !pi - lon[idx]
;if (cosr lt (sin(lat) * sin(latc))) then lon = !pi - lon
lon += lonc
;
idx = where(lon lt 0., count)
if (count ne 0) then begin
	for i = 0, count - 1 do $
		while (lon[idx[i]] lt 0.) do lon[idx[i]] += (2 * !pi)
endif
;
idx = where(lon ge (2 * !pi), count)
if (count ne 0) then begin
	for i = 0, count - 1 do $
		while (lon[idx[i]] ge (2 * !pi)) do lon[idx[i]] -= (2 * !pi)
endif


end