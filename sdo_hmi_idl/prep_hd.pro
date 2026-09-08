; prepare header for CEA field maps
; modified Jan 31 2019; no long dependent on save file

pro prep_hd, hd_in, data_out, hd_out, phi_c, lambda_c, nx, ny, dx, dy

; Get template, some basics

mkhdr, hd_out, data_out;, /extend
keys_def = ['TELESCOP', 'INSTRUME', 'WAVELNTH', 'CAMERA']
n_def = n_elements(keys_def)
for i = 0, n_def - 1 do sxaddpar, hd_out, keys_def[i], sxpar(hd_in, keys_def[i])

; Copy times, etc

keys0 = ['DATE','DATE_S','DATE-OBS','T_OBS','T_REC','TRECEPOC','TRECSTEP','TRECUNIT','HARPNUM']
keys1 = ['DSUN_OBS','DSUN_REF','RSUN_REF','CRLN_OBS','CRLT_OBS','CAR_ROT',$
		'OBS_VR','OBS_VW','OBS_VN','RSUN_OBS']
keys2 = ['QUALITY','QUAL_S','QUALLEV1']
n0 = n_elements(keys0)
n1 = n_elements(keys1)
n2 = n_elements(keys2)
for i = 0, n0 - 1 do sxaddpar, hd_out, keys0[i], sxpar(hd_in, keys0[i]), before='TRECSTEP'
for i = 0, n1 - 1 do sxaddpar, hd_out, keys1[i], sxpar(hd_in, keys1[i]), before='TELESCOP'
for i = 0, n2 - 1 do sxaddpar, hd_out, keys2[i], sxpar(hd_in, keys2[i]), before='BUNIT'

; Adding parameters

;sxaddpar, hd_out, 'NAXIS', 2, format='(i)', before='EXTEND'
;sxaddpar, hd_out, 'NAXIS1', nx, format='(i)', before='EXTEND'
;sxaddpar, hd_out, 'NAXIS2', ny, format='(i)', before='EXTEND'

sxaddpar, hd_out, 'CUNIT1', 'degree'
sxaddpar, hd_out, 'CUNIT2', 'degree'

sxaddpar, hd_out, 'CRPIX1', (nx - 1.) / 2. + 1., format='(f)', before='CUNIT1'
sxaddpar, hd_out, 'CRPIX2', (ny - 1.) / 2. + 1., format='(f)', before='CUNIT1'
sxaddpar, hd_out, 'CRVAL1', phi_c, format='(f)', before='CUNIT1'
sxaddpar, hd_out, 'CRVAL2', lambda_c, format='(f)', before='CUNIT1'
sxaddpar, hd_out, 'CDELT1', dx, format='(f)', before='CUNIT1'
sxaddpar, hd_out, 'CDELT2', dy, format='(f)', before='CUNIT1'
sxaddpar, hd_out, 'CTYPE1', 'CRLN-CEA', before='CUNIT1'
sxaddpar, hd_out, 'CTYPE2', 'CRLT-CEA', before='CUNIT1'
sxaddpar, hd_out, 'CROTA2', 0.0, format='(f)', before='CUNIT1'

sxaddpar, hd_out, 'WCSNAME', 'Carrington Heliographic'
sxaddpar, hd_out, 'BUNIT', 'Mx/cm^2'

end
