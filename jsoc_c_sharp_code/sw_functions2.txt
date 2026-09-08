
/*===========================================
 
 The following functions calculate these spaceweather indices from the vector magnetic field data:
 
 USFLUX Total unsigned flux in Maxwells
 MEANGAM Mean inclination angle, gamma, in degrees
 MEANGBT Mean value of the total field gradient, in Gauss/Mm
 MEANGBZ Mean value of the vertical field gradient, in Gauss/Mm
 MEANGBH Mean value of the horizontal field gradient, in Gauss/Mm
 MEANJZD Mean vertical current density, in mA/m2
 TOTUSJZ Total unsigned vertical current, in Amperes
 MEANALP Mean twist parameter, alpha, in 1/Mm
 MEANJZH Mean current helicity in G2/m
 TOTUSJH Total unsigned current helicity in G2/m
 ABSNJZH Absolute value of the net current helicity in G2/m
 SAVNCPP Sum of the Absolute Value of the Net Currents Per Polarity in Amperes
 MEANPOT Mean photospheric excess magnetic energy density in ergs per cubic centimeter
 TOTPOT Total photospheric magnetic energy density in ergs per cubic centimeter
 MEANSHR Mean shear angle (measured using Btotal) in degrees
 CMASK The total number of pixels that contributed to the calculation of all the indices listed above

 And these spaceweather indices from the line-of-sight magnetic field data:
 USFLUXL Total unsigned flux in Maxwells
 MEANGBL Mean value of the line-of-sight field gradient, in Gauss/Mm
 CMASKL The total number of pixels that contributed to the calculation of USFLUXL and MEANGBL
 R_VALUE Karel Schrijver's R parameter
 
 The indices are calculated on the pixels in which the conf_disambig segment is greater than 70 and
 pixels in which the bitmap segment is greater than 30. These ranges are selected because the CCD
 coordinate bitmaps are interpolated for certain data (at the time of this CVS submit, all data
 prior to 2013.08.21_17:24:00_TAI contain interpolated bitmaps; data post-2013.08.21_17:24:00_TAI
 contain nearest-neighbor bitmaps).
 
 In the CCD coordinates, this means that we are selecting the pixels that equal 90 in conf_disambig
 and the pixels that equal 33 or 34 in bitmap. Here are the definitions of the pixel values:
 
 For conf_disambig:
 50 : not all solutions agree (weak field method applied)
 60 : not all solutions agree (weak field + annealed)
 90 : all solutions agree (strong field + annealed)
 0 : not disambiguated
 
 For bitmap:
 1  : weak field outside smooth bounding curve
 2  : strong field outside smooth bounding curve
 33 : weak field inside smooth bounding curve
 34 : strong field inside smooth bounding curve
 
 Written by Monica Bobra 15 August 2012
 Potential Field code (appended) written by Keiji Hayashi
 Error analysis modification 21 October 2013
 
 ===========================================*/
#include <math.h>
#include <mkl.h>

#define PI       (M_PI)
#define MUNAUGHT (0.0000012566370614) /* magnetic constant */

/*===========================================*/

/* Example function 1: Compute total unsigned flux in units of G/cm^2 */

//  To compute the unsigned flux, we simply calculate
//  flux = surface integral [(vector Bz) dot (normal vector)],
//       = surface integral [(magnitude Bz)*(magnitude normal)*(cos theta)].
//  However, since the field is radial, we will assume cos theta = 1.
//  Therefore the pixels only need to be corrected for the projection.

//  To convert G to G*cm^2, simply multiply by the number of square centimeters per pixel.
//  As an order of magnitude estimate, we can assign 0.5 to CDELT1 and 722500m/arcsec to (RSUN_REF/RSUN_OBS).
//  (Gauss/pix^2)(CDELT1)^2(RSUN_REF/RSUN_OBS)^2(100.cm/m)^2
//  =Gauss*cm^2

int computeAbsFlux(float *bz_err, float *bz, int *dims, float *absFlux,
                   float *mean_vf_ptr, float *mean_vf_err_ptr, float *count_mask_ptr, int *mask,
                   int *bitmask, float cdelt1, double rsun_ref, double rsun_obs)

{
    
    int nx = dims[0];
    int ny = dims[1];
    int i = 0;
    int j = 0;
    int count_mask = 0;
    double sum = 0.0;
    double err = 0.0;
    *absFlux = 0.0;
    *mean_vf_ptr = 0.0;
    
    
    if (nx <= 0 || ny <= 0) return 1;
    
	for (i = 0; i < nx; i++)
	{
	   for (j = 0; j < ny; j++)
	   {
            if ( mask[j * nx + i] < 70 || bitmask[j * nx + i] < 30 ) continue;
            if isnan(bz[j * nx + i]) continue;
            sum += (fabs(bz[j * nx + i]));
            err += bz_err[j * nx + i]*bz_err[j * nx + i];
            count_mask++;
	   }
	}
    
    *mean_vf_ptr     = sum*cdelt1*cdelt1*(rsun_ref/rsun_obs)*(rsun_ref/rsun_obs)*100.0*100.0;
    *mean_vf_err_ptr = (sqrt(err))*fabs(cdelt1*cdelt1*(rsun_ref/rsun_obs)*(rsun_ref/rsun_obs)*100.0*100.0); // error in the unsigned flux
    *count_mask_ptr  = count_mask;
    return 0;
}

/*===========================================*/
/* Example function 2: Calculate Bh, the horizontal field, in units of Gauss */
// Native units of Bh are Gauss

int computeBh(float *bx_err, float *by_err, float *bh_err, float *bx, float *by, float *bz, float *bh, int *dims,
			  float *mean_hf_ptr, int *mask, int *bitmask)

{
    
    int nx = dims[0];
    int ny = dims[1];
    int i = 0;
    int j = 0;
    int count_mask = 0;
    double sum = 0.0;
    *mean_hf_ptr = 0.0;
    
    if (nx <= 0 || ny <= 0) return 1;
    
	for (i = 0; i < nx; i++)
    {
	    for (j = 0; j < ny; j++)
        {
            if isnan(bx[j * nx + i])
            {
                bh[j * nx + i] = NAN;
                bh_err[j * nx + i] = NAN;
                continue;
            }
            if isnan(by[j * nx + i])
            {
                bh[j * nx + i] = NAN;
                bh_err[j * nx + i] = NAN;
                continue;
            }
            bh[j * nx + i] = sqrt( bx[j * nx + i]*bx[j * nx + i] + by[j * nx + i]*by[j * nx + i] );
            sum += bh[j * nx + i];
            bh_err[j * nx + i]=sqrt( bx[j * nx + i]*bx[j * nx + i]*bx_err[j * nx + i]*bx_err[j * nx + i] + by[j * nx + i]*by[j * nx + i]*by_err[j * nx + i]*by_err[j * nx + i])/ bh[j * nx + i];
            count_mask++;
        }
    }
    
    *mean_hf_ptr = sum/(count_mask); // would be divided by nx*ny if shape of count_mask = shape of magnetogram
    
    return 0;
}

/*===========================================*/
/* Example function 3: Calculate Gamma in units of degrees */
// Native units of atan(x) are in radians; to convert from radians to degrees, multiply by (180./PI)
//
// Error analysis calculations are done in radians (since derivatives are only true in units of radians),
// and multiplied by (180./PI) at the end for consistency in units.

int computeGamma(float *bz_err, float *bh_err, float *bx, float *by, float *bz, float *bh, int *dims,
                 float *mean_gamma_ptr, float *mean_gamma_err_ptr, int *mask, int *bitmask)
{
    int nx = dims[0];
    int ny = dims[1];
    int i = 0;
    int j = 0;
    int count_mask = 0;
    double sum = 0.0;
    double err = 0.0;
    *mean_gamma_ptr = 0.0;
    
    if (nx <= 0 || ny <= 0) return 1;
    
	for (i = 0; i < nx; i++)
    {
	    for (j = 0; j < ny; j++)
        {
            if (bh[j * nx + i] > 100)
            {
                if ( mask[j * nx + i] < 70 || bitmask[j * nx + i] < 30 ) continue;
                if isnan(bz[j * nx + i]) continue;
                if isnan(bz_err[j * nx + i]) continue;
                if isnan(bh_err[j * nx + i]) continue;
                if isnan(bh[j * nx + i]) continue;
                if (bz[j * nx + i] == 0) continue;
                sum += fabs(atan(bh[j * nx + i]/fabs(bz[j * nx + i])))*(180./PI);
                err += (1/(1+((bh[j * nx + i]*bh[j * nx + i])/(bz[j * nx + i]*bz[j * nx + i]))))*(1/(1+((bh[j * nx + i]*bh[j * nx + i])/(bz[j * nx + i]*bz[j * nx + i])))) *
                ( ((bh_err[j * nx + i]*bh_err[j * nx + i])/(bz[j * nx + i]*bz[j * nx + i])) +
                 ((bh[j * nx + i]*bh[j * nx + i]*bz_err[j * nx + i]*bz_err[j * nx + i])/(bz[j * nx + i]*bz[j * nx + i]*bz[j * nx + i]*bz[j * nx + i])) );
                count_mask++;
            }
        }
    }
    
    *mean_gamma_ptr = sum/count_mask;
    *mean_gamma_err_ptr = (sqrt(err)/(count_mask))*(180./PI);
    //printf("MEANGAM=%f\n",*mean_gamma_ptr);
    //printf("MEANGAM_err=%f\n",*mean_gamma_err_ptr);
    return 0;
}

/*===========================================*/
/* Example function 4: Calculate B_Total*/
// Native units of B_Total are in gauss

int computeB_total(float *bx_err, float *by_err, float *bz_err, float *bt_err, float *bx, float *by, float *bz, float *bt, int *dims, int *mask, int *bitmask)
{
    
    int nx = dims[0];
    int ny = dims[1];
    int i = 0;
    int j = 0;
    int count_mask = 0;
	
    if (nx <= 0 || ny <= 0) return 1;
    
	for (i = 0; i < nx; i++)
    {
	    for (j = 0; j < ny; j++)
        {
            if isnan(bx[j * nx + i])
            {
                bt[j * nx + i] = NAN;
                bt_err[j * nx + i] = NAN;
                continue;
            }
            if isnan(by[j * nx + i])
            {
                bt[j * nx + i] = NAN;
                bt_err[j * nx + i] = NAN;
                continue;
            }
            if isnan(bz[j * nx + i])
            {
                bt[j * nx + i] = NAN;
                bt_err[j * nx + i] = NAN;
                continue;
            }
            bt[j * nx + i] = sqrt( bx[j * nx + i]*bx[j * nx + i] + by[j * nx + i]*by[j * nx + i] + bz[j * nx + i]*bz[j * nx + i]);
            bt_err[j * nx + i]=sqrt(bx[j * nx + i]*bx[j * nx + i]*bx_err[j * nx + i]*bx_err[j * nx + i] + by[j * nx + i]*by[j * nx + i]*by_err[j * nx + i]*by_err[j * nx + i] +  bz[j * nx + i]*bz[j * nx + i]*bz_err[j * nx + i]*bz_err[j * nx + i] ) / bt[j * nx + i];
        }
    }
    return 0;
}

/*===========================================*/
/* Example function 5:  Derivative of B_Total SQRT( (dBt/dx)^2 + (dBt/dy)^2 ) */

int computeBtotalderivative(float *bt, int *dims, float *mean_derivative_btotal_ptr, int *mask, int *bitmask, float *derx_bt, float *dery_bt, float *bt_err, float *mean_derivative_btotal_err_ptr, float *err_termAt, float *err_termBt)
{
    
    int nx = dims[0];
    int ny = dims[1];
    int i = 0;
    int j = 0;
    int count_mask = 0;
    double sum = 0.0;
    double err = 0.0;
    *mean_derivative_btotal_ptr = 0.0;
    
    if (nx <= 0 || ny <= 0) return 1;
    
    /* brute force method of calculating the derivative (no consideration for edges) */
    for (i = 1; i <= nx-2; i++)
    {
	for (j = 0; j <= ny-1; j++)
        {
           derx_bt[j * nx + i] = (bt[j * nx + i+1] - bt[j * nx + i-1])*0.5;
           err_termAt[j * nx + i] = (((bt[j * nx + (i+1)]-bt[j * nx + (i-1)])*(bt[j * nx + (i+1)]-bt[j * nx + (i-1)])) * (bt_err[j * nx + (i+1)]*bt_err[j * nx + (i+1)] + bt_err[j * nx + (i-1)]*bt_err[j * nx + (i-1)])) ;
        }
    }
    
    /* brute force method of calculating the derivative (no consideration for edges) */
    for (i = 0; i <= nx-1; i++)
    {
	for (j = 1; j <= ny-2; j++)
        {
           dery_bt[j * nx + i] = (bt[(j+1) * nx + i] - bt[(j-1) * nx + i])*0.5;
           err_termBt[j * nx + i] = (((bt[(j+1) * nx + i]-bt[(j-1) * nx + i])*(bt[(j+1) * nx + i]-bt[(j-1) * nx + i])) * (bt_err[(j+1) * nx + i]*bt_err[(j+1) * nx + i] + bt_err[(j-1) * nx + i]*bt_err[(j-1) * nx + i])) ;
        }
    }
    
    /* consider the edges for the arrays that contribute to the variable "sum" in the computation below.
    ignore the edges for the error terms as those arrays have been initialized to zero. 
    this is okay because the error term will ultimately not include the edge pixels as they are selected out by the mask and bitmask arrays.*/
    i=0;
    for (j = 0; j <= ny-1; j++)
    {
        derx_bt[j * nx + i] = ( (-3*bt[j * nx + i]) + (4*bt[j * nx + (i+1)]) - (bt[j * nx + (i+2)]) )*0.5;
    }
    
    i=nx-1;
    for (j = 0; j <= ny-1; j++)
    {
        derx_bt[j * nx + i] = ( (3*bt[j * nx + i]) + (-4*bt[j * nx + (i-1)]) - (-bt[j * nx + (i-2)]) )*0.5;
    }
    
    j=0;
    for (i = 0; i <= nx-1; i++)
    {
        dery_bt[j * nx + i] = ( (-3*bt[j*nx + i]) + (4*bt[(j+1) * nx + i]) - (bt[(j+2) * nx + i]) )*0.5;
    }
    
    j=ny-1;
    for (i = 0; i <= nx-1; i++)
    {
        dery_bt[j * nx + i] = ( (3*bt[j * nx + i]) + (-4*bt[(j-1) * nx + i]) - (-bt[(j-2) * nx + i]) )*0.5;
    }

    // Calculate the sum only
    for (i = 1; i <= nx-2; i++)
    {
        for (j = 1; j <= ny-2; j++)
        {
            if ( mask[j * nx + i] < 70 || bitmask[j * nx + i] < 30 ) continue;
            if ( (derx_bt[j * nx + i] + dery_bt[j * nx + i]) == 0) continue;
            if isnan(bt[j * nx + i])      continue;
            if isnan(bt[(j+1) * nx + i])  continue;
            if isnan(bt[(j-1) * nx + i])  continue;
            if isnan(bt[j * nx + i-1])    continue;
            if isnan(bt[j * nx + i+1])    continue;
            if isnan(bt_err[j * nx + i])  continue;
            if isnan(derx_bt[j * nx + i]) continue;
            if isnan(dery_bt[j * nx + i]) continue;
            sum += sqrt( derx_bt[j * nx + i]*derx_bt[j * nx + i]  + dery_bt[j * nx + i]*dery_bt[j * nx + i]  ); /* Units of Gauss */
            err += err_termBt[j * nx + i] / (16.0*( derx_bt[j * nx + i]*derx_bt[j * nx + i]  + dery_bt[j * nx + i]*dery_bt[j * nx + i]  ))+
                   err_termAt[j * nx + i] / (16.0*( derx_bt[j * nx + i]*derx_bt[j * nx + i]  + dery_bt[j * nx + i]*dery_bt[j * nx + i]  )) ;
            count_mask++;
        }
    }
    
    *mean_derivative_btotal_ptr     = (sum)/(count_mask);
    *mean_derivative_btotal_err_ptr = (sqrt(err))/(count_mask);
    //printf("MEANGBT=%f\n",*mean_derivative_btotal_ptr);
    //printf("MEANGBT_err=%f\n",*mean_derivative_btotal_err_ptr);
    
    return 0;
}


/*===========================================*/
/* Example function 6:  Derivative of Bh SQRT( (dBh/dx)^2 + (dBh/dy)^2 ) */

int computeBhderivative(float *bh, float *bh_err, int *dims, float *mean_derivative_bh_ptr, float *mean_derivative_bh_err_ptr, int *mask, int *bitmask, float *derx_bh, float *dery_bh, float *err_termAh, float *err_termBh)
{
    
    int nx = dims[0];
    int ny = dims[1];
    int i = 0;
    int j = 0;
    int count_mask = 0;
    double sum= 0.0;
    double err =0.0;
    *mean_derivative_bh_ptr = 0.0;
    
    if (nx <= 0 || ny <= 0) return 1;
    
    /* brute force method of calculating the derivative (no consideration for edges) */
    for (i = 1; i <= nx-2; i++)
    {
	for (j = 0; j <= ny-1; j++)
        {
           derx_bh[j * nx + i] = (bh[j * nx + i+1] - bh[j * nx + i-1])*0.5;
           err_termAh[j * nx + i] = (((bh[j * nx + (i+1)]-bh[j * nx + (i-1)])*(bh[j * nx + (i+1)]-bh[j * nx + (i-1)])) * (bh_err[j * nx + (i+1)]*bh_err[j * nx + (i+1)] + bh_err[j * nx + (i-1)]*bh_err[j * nx + (i-1)]));
        }
    }
    
    /* brute force method of calculating the derivative (no consideration for edges) */
    for (i = 0; i <= nx-1; i++)
    {
       for (j = 1; j <= ny-2; j++)
       {
          dery_bh[j * nx + i] = (bh[(j+1) * nx + i] - bh[(j-1) * nx + i])*0.5;
          err_termBh[j * nx + i] = (((bh[ (j+1) * nx + i]-bh[(j-1) * nx + i])*(bh[(j+1) * nx + i]-bh[(j-1) * nx + i])) * (bh_err[(j+1) * nx + i]*bh_err[(j+1) * nx + i] + bh_err[(j-1) * nx + i]*bh_err[(j-1) * nx + i]));
       }
    }
    
    /* consider the edges for the arrays that contribute to the variable "sum" in the computation below.
    ignore the edges for the error terms as those arrays have been initialized to zero. 
    this is okay because the error term will ultimately not include the edge pixels as they are selected out by the mask and bitmask arrays.*/
    i=0;
    for (j = 0; j <= ny-1; j++)
    {
        derx_bh[j * nx + i] = ( (-3*bh[j * nx + i]) + (4*bh[j * nx + (i+1)]) - (bh[j * nx + (i+2)]) )*0.5;
    }
    
    i=nx-1;
    for (j = 0; j <= ny-1; j++)
    {
        derx_bh[j * nx + i] = ( (3*bh[j * nx + i]) + (-4*bh[j * nx + (i-1)]) - (-bh[j * nx + (i-2)]) )*0.5;
    }
    
    j=0;
    for (i = 0; i <= nx-1; i++)
    {
        dery_bh[j * nx + i] = ( (-3*bh[j*nx + i]) + (4*bh[(j+1) * nx + i]) - (bh[(j+2) * nx + i]) )*0.5;
    }
    
    j=ny-1;
    for (i = 0; i <= nx-1; i++)
    {
        dery_bh[j * nx + i] = ( (3*bh[j * nx + i]) + (-4*bh[(j-1) * nx + i]) - (-bh[(j-2) * nx + i]) )*0.5;
    }
    
    
    for (i = 0; i <= nx-1; i++)
    {
        for (j = 0; j <= ny-1; j++)
        {
            if ( mask[j * nx + i] < 70 || bitmask[j * nx + i] < 30 ) continue;
            if ( (derx_bh[j * nx + i] + dery_bh[j * nx + i]) == 0) continue;
            if isnan(bh[j * nx + i])      continue;
            if isnan(bh[(j+1) * nx + i])  continue;
            if isnan(bh[(j-1) * nx + i])  continue;
            if isnan(bh[j * nx + i-1])    continue;
            if isnan(bh[j * nx + i+1])    continue;
            if isnan(bh_err[j * nx + i])  continue;
            if isnan(derx_bh[j * nx + i]) continue;
            if isnan(dery_bh[j * nx + i]) continue;
            sum += sqrt( derx_bh[j * nx + i]*derx_bh[j * nx + i]  + dery_bh[j * nx + i]*dery_bh[j * nx + i]  ); /* Units of Gauss */
            err += err_termBh[j * nx + i] / (16.0*( derx_bh[j * nx + i]*derx_bh[j * nx + i]  + dery_bh[j * nx + i]*dery_bh[j * nx + i]  ))+
                   err_termAh[j * nx + i] / (16.0*( derx_bh[j * nx + i]*derx_bh[j * nx + i]  + dery_bh[j * nx + i]*dery_bh[j * nx + i]  )) ;
            count_mask++;
        }
    }
    
    *mean_derivative_bh_ptr     = (sum)/(count_mask); // would be divided by ((nx-2)*(ny-2)) if shape of count_mask = shape of magnetogram
    *mean_derivative_bh_err_ptr = (sqrt(err))/(count_mask); // error in the quantity (sum)/(count_mask)
    //printf("MEANGBH=%f\n",*mean_derivative_bh_ptr);
    //printf("MEANGBH_err=%f\n",*mean_derivative_bh_err_ptr);
    
    return 0;
}

/*===========================================*/
/* Example function 7:  Derivative of B_vertical SQRT( (dBz/dx)^2 + (dBz/dy)^2 ) */

int computeBzderivative(float *bz, float *bz_err, int *dims, float *mean_derivative_bz_ptr, float *mean_derivative_bz_err_ptr, int *mask, int *bitmask, float *derx_bz, float *dery_bz, float *err_termA, float *err_termB)
{
    
    int nx = dims[0];
    int ny = dims[1];
    int i = 0;
    int j = 0;
    int count_mask = 0;
    double sum = 0.0;
    double err = 0.0;
    *mean_derivative_bz_ptr = 0.0;
    
    if (nx <= 0 || ny <= 0) return 1;
    
    /* brute force method of calculating the derivative (no consideration for edges) */
    for (i = 1; i <= nx-2; i++)
    {
	for (j = 0; j <= ny-1; j++)
        {
           derx_bz[j * nx + i] = (bz[j * nx + i+1] - bz[j * nx + i-1])*0.5;
           err_termA[j * nx + i] = (((bz[j * nx + (i+1)]-bz[j * nx + (i-1)])*(bz[j * nx + (i+1)]-bz[j * nx + (i-1)])) * (bz_err[j * nx + (i+1)]*bz_err[j * nx + (i+1)] + bz_err[j * nx + (i-1)]*bz_err[j * nx + (i-1)]));
        }
    }
    
    /* brute force method of calculating the derivative (no consideration for edges) */
    for (i = 0; i <= nx-1; i++)
    {
        for (j = 1; j <= ny-2; j++)
        {
           dery_bz[j * nx + i] = (bz[(j+1) * nx + i] - bz[(j-1) * nx + i])*0.5;
           err_termB[j * nx + i] = (((bz[(j+1) * nx + i]-bz[(j-1) * nx + i])*(bz[(j+1) * nx + i]-bz[(j-1) * nx + i])) * (bz_err[(j+1) * nx + i]*bz_err[(j+1) * nx + i] + bz_err[(j-1) * nx + i]*bz_err[(j-1) * nx + i]));
        }
    }
    
    /* consider the edges for the arrays that contribute to the variable "sum" in the computation below.
    ignore the edges for the error terms as those arrays have been initialized to zero. 
    this is okay because the error term will ultimately not include the edge pixels as they are selected out by the mask and bitmask arrays.*/
    i=0;
    for (j = 0; j <= ny-1; j++)
    {
        derx_bz[j * nx + i] = ( (-3*bz[j * nx + i]) + (4*bz[j * nx + (i+1)]) - (bz[j * nx + (i+2)]) )*0.5;
    }
    
    i=nx-1;
    for (j = 0; j <= ny-1; j++)
    {
        derx_bz[j * nx + i] = ( (3*bz[j * nx + i]) + (-4*bz[j * nx + (i-1)]) - (-bz[j * nx + (i-2)]) )*0.5;
    }
    
    j=0;
    for (i = 0; i <= nx-1; i++)
    {
        dery_bz[j * nx + i] = ( (-3*bz[j*nx + i]) + (4*bz[(j+1) * nx + i]) - (bz[(j+2) * nx + i]) )*0.5;
    }
    
    j=ny-1;
    for (i = 0; i <= nx-1; i++)
    {
        dery_bz[j * nx + i] = ( (3*bz[j * nx + i]) + (-4*bz[(j-1) * nx + i]) - (-bz[(j-2) * nx + i]) )*0.5;
    }
    
    
    for (i = 0; i <= nx-1; i++)
    {
        for (j = 0; j <= ny-1; j++)
        {
            if ( mask[j * nx + i] < 70 || bitmask[j * nx + i] < 30 ) continue;
            if ( (derx_bz[j * nx + i] + dery_bz[j * nx + i]) == 0) continue;
            if isnan(bz[j * nx + i])      continue;
            if isnan(bz[(j+1) * nx + i])  continue;
            if isnan(bz[(j-1) * nx + i])  continue;
            if isnan(bz[j * nx + i-1])    continue;
            if isnan(bz[j * nx + i+1])    continue;
            if isnan(bz_err[j * nx + i])  continue;
            if isnan(derx_bz[j * nx + i]) continue;
            if isnan(dery_bz[j * nx + i]) continue;
            sum += sqrt( derx_bz[j * nx + i]*derx_bz[j * nx + i]  + dery_bz[j * nx + i]*dery_bz[j * nx + i] ); /* Units of Gauss */
            err += err_termB[j * nx + i] / (16.0*( derx_bz[j * nx + i]*derx_bz[j * nx + i]  + dery_bz[j * nx + i]*dery_bz[j * nx + i]  )) +
                   err_termA[j * nx + i] / (16.0*( derx_bz[j * nx + i]*derx_bz[j * nx + i]  + dery_bz[j * nx + i]*dery_bz[j * nx + i]  )) ;
            count_mask++;
        }
    }
    
    *mean_derivative_bz_ptr = (sum)/(count_mask); // would be divided by ((nx-2)*(ny-2)) if shape of count_mask = shape of magnetogram
    *mean_derivative_bz_err_ptr = (sqrt(err))/(count_mask); // error in the quantity (sum)/(count_mask)
    //printf("MEANGBZ=%f\n",*mean_derivative_bz_ptr);
    //printf("MEANGBZ_err=%f\n",*mean_derivative_bz_err_ptr);
    
	return 0;
}

/*===========================================*/
/* Example function 8:  Current Jz = (dBy/dx) - (dBx/dy) */

//  In discretized space like data pixels,
//  the current (or curl of B) is calculated as
//  the integration of the field Bx and By along
//  the circumference of the data pixel divided by the area of the pixel.
//  One form of differencing (a word for the differential operator
//  in the discretized space) of the curl is expressed as
//  (dx * (Bx(i,j-1)+Bx(i,j)) / 2
//  +dy * (By(i+1,j)+By(i,j)) / 2
//  -dx * (Bx(i,j+1)+Bx(i,j)) / 2
//  -dy * (By(i-1,j)+By(i,j)) / 2) / (dx * dy)
//
//
//  To change units from Gauss/pixel to mA/m^2 (the units for Jz in Leka and Barnes, 2003),
//  one must perform the following unit conversions:
//  (Gauss)(1/arcsec)(arcsec/meter)(Newton/Gauss*Ampere*meter)(Ampere^2/Newton)(milliAmpere/Ampere), or
//  (Gauss)(1/CDELT1)(RSUN_OBS/RSUN_REF)(1 T / 10^4 Gauss)(1 / 4*PI*10^-7)( 10^3 milliAmpere/Ampere), or
//  (Gauss)(1/CDELT1)(RSUN_OBS/RSUN_REF)(0.00010)(1/MUNAUGHT)(1000.),
//  where a Tesla is represented as a Newton/Ampere*meter.
//
//  As an order of magnitude estimate, we can assign 0.5 to CDELT1 and 722500m/arcsec to (RSUN_REF/RSUN_OBS).
//  In that case, we would have the following:
//  (Gauss/pix)(1/0.5)(1/722500)(10^-4)(4*PI*10^7)(10^3), or
//  jz * (35.0)
//
//  The units of total unsigned vertical current (us_i) are simply in A. In this case, we would have the following:
//  (Gauss/pix)(1/CDELT1)(RSUN_OBS/RSUN_REF)(0.00010)(1/MUNAUGHT)(CDELT1)(CDELT1)(RSUN_REF/RSUN_OBS)(RSUN_REF/RSUN_OBS)
//  = (Gauss/pix)(0.00010)(1/MUNAUGHT)(CDELT1)(RSUN_REF/RSUN_OBS)


// Comment out random number generator, which can only run on solar3
// int computeJz(float *bx_err, float *by_err, float *bx, float *by, int *dims, float *jz, float *jz_err, float *jz_err_squared,
//	      int *mask, int *bitmask, float cdelt1, double rsun_ref, double rsun_obs,float *derx, float *dery, float *noisebx,
//              float *noiseby, float *noisebz)

int computeJz(float *bx_err, float *by_err, float *bx, float *by, int *dims, float *jz, float *jz_err, float *jz_err_squared,
              int *mask, int *bitmask, float cdelt1, double rsun_ref, double rsun_obs,float *derx, float *dery, float *err_term1, float *err_term2)


{
    int nx = dims[0];
    int ny = dims[1];
    int i = 0;
    int j = 0;
    int count_mask = 0;
    
	if (nx <= 0 || ny <= 0) return 1;
    
    /* Calculate the derivative*/
    /* brute force method of calculating the derivative (no consideration for edges) */
    
    for (i = 1; i <= nx-2; i++)
    {
	for (j = 0; j <= ny-1; j++)
        {
           derx[j * nx + i]      = (by[j * nx + i+1] - by[j * nx + i-1])*0.5;
           err_term1[j * nx + i] = (by_err[j * nx + i+1])*(by_err[j * nx + i+1]) + (by_err[j * nx + i-1])*(by_err[j * nx + i-1]);
        }
    }
    
    for (i = 0; i <= nx-1; i++)
    {
	for (j = 1; j <= ny-2; j++)
        {
           dery[j * nx + i]      = (bx[(j+1) * nx + i] - bx[(j-1) * nx + i])*0.5;
           err_term2[j * nx + i] = (bx_err[(j+1) * nx + i])*(bx_err[(j+1) * nx + i]) + (bx_err[(j-1) * nx + i])*(bx_err[(j-1) * nx + i]);
        }
    }

    /* consider the edges for the arrays that contribute to the variable "sum" in the computation below.
    ignore the edges for the error terms as those arrays have been initialized to zero. 
    this is okay because the error term will ultimately not include the edge pixels as they are selected out by the mask and bitmask arrays.*/

    i=0;
    for (j = 0; j <= ny-1; j++)
    {
        derx[j * nx + i]      = ( (-3*by[j * nx + i]) + (4*by[j * nx + (i+1)]) - (by[j * nx + (i+2)]) )*0.5;
    }
    
    i=nx-1;
    for (j = 0; j <= ny-1; j++)
    {
        derx[j * nx + i]      = ( (3*by[j * nx + i]) + (-4*by[j * nx + (i-1)]) - (-by[j * nx + (i-2)]) )*0.5;
    }
    
    j=0;
    for (i = 0; i <= nx-1; i++)
    {
        dery[j * nx + i]      = ( (-3*bx[j*nx + i]) + (4*bx[(j+1) * nx + i]) - (bx[(j+2) * nx + i]) )*0.5;
    }
    
    j=ny-1;
    for (i = 0; i <= nx-1; i++)
    {
        dery[j * nx + i]      = ( (3*bx[j * nx + i]) + (-4*bx[(j-1) * nx + i]) - (-bx[(j-2) * nx + i]) )*0.5;
    }

    
    for (i = 0; i <= nx-1; i++)
    {
        for (j = 0; j <= ny-1; j++)
        {
            // calculate jz at all points         
            jz[j * nx + i]            = (derx[j * nx + i]-dery[j * nx + i]);       // jz is in units of Gauss/pix
            jz_err[j * nx + i]        = 0.5*sqrt( err_term1[j * nx + i] + err_term2[j * nx + i] ) ;
            jz_err_squared[j * nx + i]= (jz_err[j * nx + i]*jz_err[j * nx + i]);
            count_mask++;            
        }
    }
	return 0;
}
 
/*===========================================*/

/* Example function 9:  Compute quantities on Jz array */
// Compute mean and total current on Jz array.

int computeJzsmooth(float *bx, float *by, int *dims, float *jz, float *jz_smooth, float *jz_err, float *jz_rms_err, float *jz_err_squared_smooth,
                    float *mean_jz_ptr, float *mean_jz_err_ptr, float *us_i_ptr, float *us_i_err_ptr, int *mask, int *bitmask,
                    float cdelt1, double rsun_ref, double rsun_obs,float *derx, float *dery)

{
    
    int nx = dims[0];
    int ny = dims[1];
    int i = 0;
    int j = 0;
    int count_mask = 0;
	double curl = 0.0;
    double us_i = 0.0;
    double err = 0.0;
    
	if (nx <= 0 || ny <= 0) return 1;
    
    /* At this point, use the smoothed Jz array with a Gaussian (FWHM of 4 pix and truncation width of 12 pixels) but keep the original array dimensions*/
    for (i = 0; i <= nx-1; i++)
    {
        for (j = 0; j <= ny-1; j++)
        {
            if ( mask[j * nx + i] < 70 || bitmask[j * nx + i] < 30 ) continue;
            if isnan(derx[j * nx + i]) continue;
            if isnan(dery[j * nx + i]) continue;
            if isnan(jz[j * nx + i]) continue;
            curl +=     (jz[j * nx + i])*(1/cdelt1)*(rsun_obs/rsun_ref)*(0.00010)*(1/MUNAUGHT)*(1000.); /* curl is in units of mA / m^2 */
            us_i += fabs(jz[j * nx + i])*(cdelt1/1)*(rsun_ref/rsun_obs)*(0.00010)*(1/MUNAUGHT);         /* us_i is in units of A */
            err  += (jz_err[j * nx + i]*jz_err[j * nx + i]);
            count_mask++;
        }
    }
    
    /* Calculate mean vertical current density (mean_jz) and total unsigned vertical current (us_i) using smoothed Jz array and continue conditions above */
    *mean_jz_ptr     = curl/(count_mask);        /* mean_jz gets populated as MEANJZD */
    *mean_jz_err_ptr = (sqrt(err)/count_mask)*((1/cdelt1)*(rsun_obs/rsun_ref)*(0.00010)*(1/MUNAUGHT)*(1000.)); // error in the quantity MEANJZD
    
    *us_i_ptr        = (us_i);                   /* us_i gets populated as TOTUSJZ */
    *us_i_err_ptr    = (sqrt(err))*((cdelt1/1)*(rsun_ref/rsun_obs)*(0.00010)*(1/MUNAUGHT)); // error in the quantity TOTUSJZ
    
    //printf("MEANJZD=%f\n",*mean_jz_ptr);
    //printf("MEANJZD_err=%f\n",*mean_jz_err_ptr);
    
    //printf("TOTUSJZ=%g\n",*us_i_ptr);
    //printf("TOTUSJZ_err=%g\n",*us_i_err_ptr);
    
	return 0;
    
}

/*===========================================*/

/* Example function 10:  Twist Parameter, alpha */

// The twist parameter, alpha, is defined as alpha = Jz/Bz. In this case, the calculation
// for alpha is weighted by Bz (following Hagino et al., http://adsabs.harvard.edu/abs/2004PASJ...56..831H):

// numerator   = sum of all Jz*Bz
// denominator = sum of Bz*Bz
// alpha       = numerator/denominator

// The units of alpha are in 1/Mm
// The units of Jz are in Gauss/pix; the units of Bz are in Gauss.
//
// Therefore, the units of Jz/Bz = (Gauss/pix)(1/Gauss)(pix/arcsec)(arsec/meter)(meter/Mm), or
//                               = (Gauss/pix)(1/Gauss)(1/CDELT1)(RSUN_OBS/RSUN_REF)(10^6)
//                               = 1/Mm

int computeAlpha(float *jz_err, float *bz_err, float *bz, int *dims, float *jz, float *jz_smooth, float *mean_alpha_ptr, float *mean_alpha_err_ptr, int *mask, int *bitmask, float cdelt1, double rsun_ref, double rsun_obs)

{
    int nx                     = dims[0];
    int ny                     = dims[1];
    int i                      = 0;
    int j                      = 0;
	double alpha_total         = 0.0;
    double C                   = ((1/cdelt1)*(rsun_obs/rsun_ref)*(1000000.));
    double total               = 0.0;
    double A                   = 0.0;
    double B                   = 0.0;
    
	if (nx <= 0 || ny <= 0) return 1;
	for (i = 1; i < nx-1; i++)
    {
	    for (j = 1; j < ny-1; j++)
        {
            if ( mask[j * nx + i] < 70 || bitmask[j * nx + i] < 30 ) continue;
            if isnan(jz[j * nx + i])   continue;
            if isnan(bz[j * nx + i])   continue;
            if (jz[j * nx + i] == 0.0) continue;
            if (bz[j * nx + i] == 0.0) continue;
            A += jz[j*nx+i]*bz[j*nx+i];
            B += bz[j*nx+i]*bz[j*nx+i];
        }
    }
    
	for (i = 1; i < nx-1; i++)
    {
	    for (j = 1; j < ny-1; j++)
        {
            if ( mask[j * nx + i] < 70 || bitmask[j * nx + i] < 30 ) continue;
            if isnan(jz[j * nx + i])   continue;
            if isnan(bz[j * nx + i])   continue;
            if (jz[j * nx + i] == 0.0) continue;
            if (bz[j * nx + i] == 0.0) continue;
            total += bz[j*nx+i]*bz[j*nx+i]*jz_err[j*nx+i]*jz_err[j*nx+i] + (jz[j*nx+i]-2*bz[j*nx+i]*A/B)*(jz[j*nx+i]-2*bz[j*nx+i]*A/B)*bz_err[j*nx+i]*bz_err[j*nx+i];
        }
    }
    
    /* Determine the absolute value of alpha. The units for alpha are 1/Mm */
    alpha_total              = ((A/B)*C);
    *mean_alpha_ptr          = alpha_total;
    *mean_alpha_err_ptr      = (C/B)*(sqrt(total));
    
	return 0;
}

/*===========================================*/
/* Example function 11:  Helicity (mean current helicty, total unsigned current helicity, absolute value of net current helicity) */

//  The current helicity is defined as Bz*Jz and the units are G^2 / m
//  The units of Jz are in G/pix; the units of Bz are in G.
//  Therefore, the units of Bz*Jz = (Gauss)*(Gauss/pix) = (Gauss^2/pix)(pix/arcsec)(arcsec/meter)
//                                                      = (Gauss^2/pix)(1/CDELT1)(RSUN_OBS/RSUN_REF)
//                                                      =  G^2 / m.

int computeHelicity(float *jz_err, float *jz_rms_err, float *bz_err, float *bz, int *dims, float *jz, float *mean_ih_ptr,
                    float *mean_ih_err_ptr, float *total_us_ih_ptr, float *total_abs_ih_ptr,
                    float *total_us_ih_err_ptr, float *total_abs_ih_err_ptr, int *mask, int *bitmask, float cdelt1, double rsun_ref, double rsun_obs)

{
    
    int nx         = dims[0];
    int ny         = dims[1];
    int i          = 0;
    int j          = 0;
    int count_mask = 0;
	double sum     = 0.0;
	double sum2    = 0.0;
	double err     = 0.0;
	
	if (nx <= 0 || ny <= 0) return 1;
    
	for (i = 0; i < nx; i++)
	{
		for (j = 0; j < ny; j++)
		{
            if ( mask[j * nx + i] < 70 || bitmask[j * nx + i] < 30 ) continue;
            if isnan(jz[j * nx + i])     continue;
            if isnan(bz[j * nx + i])     continue;
            if isnan(jz_err[j * nx + i]) continue;
            if isnan(bz_err[j * nx + i]) continue;
            if (bz[j * nx + i] == 0.0)   continue;
            if (jz[j * nx + i] == 0.0)   continue;
            sum     +=     (jz[j * nx + i]*bz[j * nx + i])*(1/cdelt1)*(rsun_obs/rsun_ref); // contributes to MEANJZH and ABSNJZH
            sum2    += fabs(jz[j * nx + i]*bz[j * nx + i])*(1/cdelt1)*(rsun_obs/rsun_ref); // contributes to TOTUSJH
            err     += (jz_err[j * nx + i]*jz_err[j * nx + i]*bz[j * nx + i]*bz[j * nx + i]) + (bz_err[j * nx + i]*bz_err[j * nx + i]*jz[j * nx + i]*jz[j * nx + i]);
            count_mask++;
        }
    }
    
	*mean_ih_ptr          = sum/count_mask ; /* Units are G^2 / m ; keyword is MEANJZH */
	*total_us_ih_ptr      = sum2           ; /* Units are G^2 / m ; keyword is TOTUSJH */
	*total_abs_ih_ptr     = fabs(sum)      ; /* Units are G^2 / m ; keyword is ABSNJZH */
    
    *mean_ih_err_ptr      = (sqrt(err)/count_mask)*(1/cdelt1)*(rsun_obs/rsun_ref) ; // error in the quantity MEANJZH
    *total_us_ih_err_ptr  = (sqrt(err))*(1/cdelt1)*(rsun_obs/rsun_ref) ;            // error in the quantity TOTUSJH
    *total_abs_ih_err_ptr = (sqrt(err))*(1/cdelt1)*(rsun_obs/rsun_ref) ;            // error in the quantity ABSNJZH
    
    //printf("MEANJZH=%f\n",*mean_ih_ptr);
    //printf("MEANJZH_err=%f\n",*mean_ih_err_ptr);
    
    //printf("TOTUSJH=%f\n",*total_us_ih_ptr);
    //printf("TOTUSJH_err=%f\n",*total_us_ih_err_ptr);
    
    //printf("ABSNJZH=%f\n",*total_abs_ih_ptr);
    //printf("ABSNJZH_err=%f\n",*total_abs_ih_err_ptr);
    
	return 0;
}

/*===========================================*/
/* Example function 12:  Sum of Absolute Value per polarity  */

//  The Sum of the Absolute Value per polarity is defined as the following:
//  fabs(sum(jz gt 0)) + fabs(sum(jz lt 0)) and the units are in Amperes per square arcsecond.
//  The units of jz are in G/pix. In this case, we would have the following:
//  Jz = (Gauss/pix)(1/CDELT1)(0.00010)(1/MUNAUGHT)(RSUN_REF/RSUN_OBS)(RSUN_REF/RSUN_OBS)(RSUN_OBS/RSUN_REF),
//     = (Gauss/pix)(1/CDELT1)(0.00010)(1/MUNAUGHT)(RSUN_REF/RSUN_OBS)
//
//  The error in this quantity is the same as the error in the mean vertical current (mean_jz_err).

int computeSumAbsPerPolarity(float *jz_err, float *bz_err, float *bz, float *jz, int *dims, float *totaljzptr, float *totaljz_err_ptr,
							 int *mask, int *bitmask, float cdelt1, double rsun_ref, double rsun_obs)

{
    int nx = dims[0];
    int ny = dims[1];
    int i=0;
    int j=0;
    int count_mask=0;
    double sum1=0.0;
    double sum2=0.0;
    double err=0.0;
    *totaljzptr=0.0;
    
    if (nx <= 0 || ny <= 0) return 1;
    
    for (i = 0; i < nx; i++)
    {
	 for (j = 0; j < ny; j++)
        {
            if ( mask[j * nx + i] < 70 || bitmask[j * nx + i] < 30 ) continue;
            if isnan(bz[j * nx + i]) continue;
            if isnan(jz[j * nx + i]) continue;
            if (bz[j * nx + i] >  0) sum1 += ( jz[j * nx + i])*(1/cdelt1)*(0.00010)*(1/MUNAUGHT)*(rsun_ref/rsun_obs);
            if (bz[j * nx + i] <= 0) sum2 += ( jz[j * nx + i])*(1/cdelt1)*(0.00010)*(1/MUNAUGHT)*(rsun_ref/rsun_obs);
            err += (jz_err[j * nx + i]*jz_err[j * nx + i]);
            count_mask++;
        }
    }
	
    *totaljzptr    = fabs(sum1) + fabs(sum2);  /* Units are Amperes per arcsecond */
    *totaljz_err_ptr = sqrt(err)*(1/cdelt1)*fabs((0.00010)*(1/MUNAUGHT)*(rsun_ref/rsun_obs));
    //printf("SAVNCPP=%g\n",*totaljzptr);
    //printf("SAVNCPP_err=%g\n",*totaljz_err_ptr);
    
    return 0;
}

/*===========================================*/
/* Example function 13:  Mean photospheric excess magnetic energy and total photospheric excess magnetic energy density */
// The units for magnetic energy density in cgs are ergs per cubic centimeter. The formula B^2/8*PI integrated over all space, dV
// automatically yields erg per cubic centimeter for an input B in Gauss. Note that the 8*PI can come out of the integral; thus,
// the integral is over B^2 dV and the 8*PI is divided at the end.
//
// Total magnetic energy is the magnetic energy density times dA, or the area, and the units are thus ergs/cm. To convert
// ergs per centimeter cubed to ergs per centimeter, simply multiply by the area per pixel in cm:
//   erg/cm^3*(CDELT1^2)*(RSUN_REF/RSUN_OBS ^2)*(100.^2)
// = erg/cm^3*(0.5 arcsec/pix)^2(722500m/arcsec)^2(100cm/m)^2
// = erg/cm^3*(1.30501e15)
// = erg/cm(1/pix^2)

int computeFreeEnergy(float *bx_err, float *by_err, float *bx, float *by, float *bpx, float *bpy, int *dims,
                      float *meanpotptr, float *meanpot_err_ptr, float *totpotptr, float *totpot_err_ptr, int *mask, int *bitmask,
					  float cdelt1, double rsun_ref, double rsun_obs)

{
    int nx = dims[0];
    int ny = dims[1];
    int i = 0;
    int j = 0;
    int count_mask = 0;
    double sum = 0.0;
    double sum1 = 0.0;
    double err = 0.0;
    *totpotptr = 0.0;
    *meanpotptr = 0.0;
    
    if (nx <= 0 || ny <= 0) return 1;
    
    for (i = 0; i < nx; i++)
    {
	for (j = 0; j < ny; j++)
        {
            if ( mask[j * nx + i] < 70 || bitmask[j * nx + i] < 30 ) continue;
            if isnan(bx[j * nx + i]) continue;
            if isnan(by[j * nx + i]) continue;
            sum  += ( ((bx[j * nx + i] - bpx[j * nx + i])*(bx[j * nx + i] - bpx[j * nx + i])) + ((by[j * nx + i] - bpy[j * nx + i])*(by[j * nx + i] - bpy[j * nx + i])) )*(cdelt1*cdelt1*(rsun_ref/rsun_obs)*(rsun_ref/rsun_obs)*100.0*100.0);
            sum1 += (  ((bx[j * nx + i] - bpx[j * nx + i])*(bx[j * nx + i] - bpx[j * nx + i])) + ((by[j * nx + i] - bpy[j * nx + i])*(by[j * nx + i] - bpy[j * nx + i])) );
            err  += 4.0*(bx[j * nx + i] - bpx[j * nx + i])*(bx[j * nx + i] - bpx[j * nx + i])*(bx_err[j * nx + i]*bx_err[j * nx + i]) +
            4.0*(by[j * nx + i] - bpy[j * nx + i])*(by[j * nx + i] - bpy[j * nx + i])*(by_err[j * nx + i]*by_err[j * nx + i]);
            count_mask++;
        }
    }
    
    /* Units of meanpotptr are ergs per centimeter */
	*meanpotptr      = (sum1) / (count_mask*8.*PI) ;     /* Units are ergs per cubic centimeter */
    *meanpot_err_ptr = (sqrt(err)) / (count_mask*8.*PI); // error in the quantity (sum)/(count_mask)
    
    /* Units of sum are ergs/cm^3, units of factor are cm^2/pix^2; therefore, units of totpotptr are ergs per centimeter */
    *totpotptr       = (sum)/(8.*PI);
    *totpot_err_ptr  = (sqrt(err))*fabs(cdelt1*cdelt1*(rsun_ref/rsun_obs)*(rsun_ref/rsun_obs)*100.0*100.0*(1/(8.*PI)));
    
    //printf("MEANPOT=%g\n",*meanpotptr);
    //printf("MEANPOT_err=%g\n",*meanpot_err_ptr);
    
    //printf("TOTPOT=%g\n",*totpotptr);
    //printf("TOTPOT_err=%g\n",*totpot_err_ptr);
    
    return 0;
}

/*===========================================*/
/* Example function 14:  Mean 3D shear angle, area with shear greater than 45, mean horizontal shear angle, area with horizontal shear angle greater than 45 */

int computeShearAngle(float *bx_err, float *by_err, float *bz_err, float *bx, float *by, float *bz, float *bpx, float *bpy, float *bpz, int *dims,
                      float *meanshear_angleptr, float *meanshear_angle_err_ptr, float *area_w_shear_gt_45ptr, int *mask, int *bitmask)


{
    int nx = dims[0];
    int ny = dims[1];
    int i = 0;
    int j = 0;
    float count_mask = 0;
    float count = 0;
    double dotproduct = 0.0;
    double magnitude_potential = 0.0;
    double magnitude_vector = 0.0;
    double shear_angle = 0.0;
    double denominator = 0.0;
    double term1 = 0.0;
    double term2 = 0.0;
    double term3 = 0.0;
    double sumsum = 0.0;
    double err = 0.0;
    double part1 = 0.0;
    double part2 = 0.0;
    double part3 = 0.0;
    *area_w_shear_gt_45ptr = 0.0;
    *meanshear_angleptr = 0.0;
	
    if (nx <= 0 || ny <= 0) return 1;
    
    for (i = 0; i < nx; i++)
    {
	for (j = 0; j < ny; j++)
        {
            if ( mask[j * nx + i] < 70 || bitmask[j * nx + i] < 30 ) continue;
            if isnan(bpx[j * nx + i]) continue;
            if isnan(bpy[j * nx + i]) continue;
            if isnan(bpz[j * nx + i]) continue;
            if isnan(bz[j * nx + i]) continue;
            if isnan(bx[j * nx + i]) continue;
            if isnan(by[j * nx + i]) continue;
            if isnan(bx_err[j * nx + i]) continue;
            if isnan(by_err[j * nx + i]) continue;
            if isnan(bz_err[j * nx + i]) continue;
            
            /* For mean 3D shear angle, percentage with shear greater than 45*/
            dotproduct            = (bpx[j * nx + i])*(bx[j * nx + i]) + (bpy[j * nx + i])*(by[j * nx + i]) + (bpz[j * nx + i])*(bz[j * nx + i]);
            magnitude_potential   = sqrt( (bpx[j * nx + i]*bpx[j * nx + i]) + (bpy[j * nx + i]*bpy[j * nx + i]) + (bpz[j * nx + i]*bpz[j * nx + i]));
            magnitude_vector      = sqrt( (bx[j * nx + i]*bx[j * nx + i])   + (by[j * nx + i]*by[j * nx + i])   + (bz[j * nx + i]*bz[j * nx + i]) );
            //printf("dotproduct=%f\n",dotproduct);
            //printf("magnitude_potential=%f\n",magnitude_potential);
            //printf("magnitude_vector=%f\n",magnitude_vector);
            
            shear_angle            = acos(dotproduct/(magnitude_potential*magnitude_vector))*(180./PI);
            sumsum                  += shear_angle;
            //printf("shear_angle=%f\n",shear_angle);
            count ++;
            
            if (shear_angle > 45) count_mask ++;
            
            // For the error analysis
            
            term1 = bx[j * nx + i]*by[j * nx + i]*bpy[j * nx + i] - by[j * nx + i]*by[j * nx + i]*bpx[j * nx + i] + bz[j * nx + i]*bx[j * nx + i]*bpz[j * nx + i] - bz[j * nx + i]*bz[j * nx + i]*bpx[j * nx + i];
            term2 = bx[j * nx + i]*bx[j * nx + i]*bpy[j * nx + i] - bx[j * nx + i]*by[j * nx + i]*bpx[j * nx + i] + bx[j * nx + i]*bz[j * nx + i]*bpy[j * nx + i] - bz[j * nx + i]*by[j * nx + i]*bpz[j * nx + i];
            term3 = bx[j * nx + i]*bx[j * nx + i]*bpz[j * nx + i] - bx[j * nx + i]*bz[j * nx + i]*bpx[j * nx + i] + by[j * nx + i]*by[j * nx + i]*bpz[j * nx + i] - by[j * nx + i]*bz[j * nx + i]*bpy[j * nx + i];
            
            part1 = bx[j * nx + i]*bx[j * nx + i] + by[j * nx + i]*by[j * nx + i] + bz[j * nx + i]*bz[j * nx + i];
            part2 = bpx[j * nx + i]*bpx[j * nx + i] + bpy[j * nx + i]*bpy[j * nx + i] + bpz[j * nx + i]*bpz[j * nx + i];
            part3 = bx[j * nx + i]*bpx[j * nx + i] + by[j * nx + i]*bpy[j * nx + i] + bz[j * nx + i]*bpz[j * nx + i];
            
            // denominator is squared
            denominator = part1*part1*part1*part2*(1.0-((part3*part3)/(part1*part2)));
            
            err = (term1*term1*bx_err[j * nx + i]*bx_err[j * nx + i])/(denominator) +
            (term1*term1*bx_err[j * nx + i]*bx_err[j * nx + i])/(denominator) +
            (term1*term1*bx_err[j * nx + i]*bx_err[j * nx + i])/(denominator) ;
            
        }
    }
    /* For mean 3D shear angle, area with shear greater than 45*/
    *meanshear_angleptr = (sumsum)/(count);                 /* Units are degrees */
    
    // For the error in the mean 3D shear angle:
    // If count_mask is 0, then we run into a divide by zero error. In this case, set *meanshear_angle_err_ptr to NAN
    // If count_mask is greater than zero, then compute the error.
    if (count_mask == 0)
        *meanshear_angle_err_ptr = NAN;
    else
        *meanshear_angle_err_ptr = (sqrt(err)/count_mask)*(180./PI);
    
    /* The area here is a fractional area -- the % of the total area. This has no error associated with it. */
    *area_w_shear_gt_45ptr   = (count_mask/(count))*(100.0);
    
    //printf("MEANSHR=%f\n",*meanshear_angleptr);
    //printf("ERRMSHA=%f\n",*meanshear_angle_err_ptr);
    //printf("SHRGT45=%f\n",*area_w_shear_gt_45ptr);
    return 0;
}

/*===========================================*/
/* Example function 15: R parameter as defined in Schrijver, 2007 */
//
// Note that there is a restriction on the function fsample()
// If the following occurs:
//      nx_out > floor((ny_in-1)/scale + 1) 
//      ny_out > floor((ny_in-1)/scale + 1),
// where n*_out are the dimensions of the output array and n*_in 
// are the dimensions of the input array, fsample() will usually result 
// in a segfault (though not always, depending on how the segfault was accessed.) 

int computeR(float *bz_err, float *los, int *dims, float *Rparam, float cdelt1,
             float *rim, float *p1p0, float *p1n0, float *p1p, float *p1n, float *p1,
             float *pmap, int nx1, int ny1, 
             int scale, float *p1pad, int nxp, int nyp, float *pmapn)

{ 
    int nx = dims[0];
    int ny = dims[1];
    int i = 0;
    int j = 0;
    int index, index1;
    double sum = 0.0;
    double err = 0.0;
    *Rparam = 0.0;
    struct fresize_struct fresboxcar, fresgauss;
    struct fint_struct fints;
    float sigma = 10.0/2.3548;
    
    // set up convolution kernels
    init_fresize_boxcar(&fresboxcar,1,1);
    init_fresize_gaussian(&fresgauss,sigma,20,1);

    // =============== [STEP 1] =============== 
    // bin the line-of-sight magnetogram down by a factor of scale 
    fsample(los, rim, nx, ny, nx, nx1, ny1, nx1, scale, 0, 0, 0.0);

    // =============== [STEP 2] =============== 
    // identify positive and negative pixels greater than +/- 150 gauss
    // and label those pixels with a 1.0 in arrays p1p0 and p1n0
    for (i = 0; i < nx1; i++)
    {
        for (j = 0; j < ny1; j++)
        {
            index = j * nx1 + i;
            if (rim[index] > 150)
                p1p0[index]=1.0;
            else
                p1p0[index]=0.0;
            if (rim[index] < -150)
                p1n0[index]=1.0;
            else
                p1n0[index]=0.0;
        }
    }

    // =============== [STEP 3] =============== 
    // smooth each of the negative and positive pixel bitmaps      
    fresize(&fresboxcar, p1p0, p1p, nx1, ny1, nx1, nx1, ny1, nx1, 0, 0, 0.0);
    fresize(&fresboxcar, p1n0, p1n, nx1, ny1, nx1, nx1, ny1, nx1, 0, 0, 0.0);

    // =============== [STEP 4] =============== 
    // find the pixels for which p1p and p1n are both equal to 1. 
    // this defines the polarity inversion line
    for (i = 0; i < nx1; i++)
    {
        for (j = 0; j < ny1; j++)
        {
            index = j * nx1 + i;
            if ((p1p[index] > 0.0) && (p1n[index] > 0.0))
                p1[index]=1.0;
            else
                p1[index]=0.0;
        }
    }

    // pad p1 with zeroes so that the gaussian colvolution in step 5
    // does not cut off data within hwidth of the edge
   
    // step i: zero p1pad
    for (i = 0; i < nxp; i++)
    {
        for (j = 0; j < nyp; j++)
        {
            index = j * nxp + i;
            p1pad[index]=0.0;
        }
    }

    // step ii: place p1 at the center of p1pad
    for (i = 0; i < nx1; i++)
    {
       for (j = 0; j < ny1; j++)
       {
            index  = j * nx1 + i; 
            index1 = (j+20) * nxp + (i+20);
            p1pad[index1]=p1[index];
        }
    }

    // =============== [STEP 5] =============== 
    // convolve the polarity inversion line map with a gaussian
    // to identify the region near the plarity inversion line
    // the resultant array is called pmap
    fresize(&fresgauss, p1pad, pmap, nxp, nyp, nxp, nxp, nyp, nxp, 0, 0, 0.0);


   // select out the nx1 x ny1 non-padded array  within pmap
    for (i = 0; i < nx1; i++)
    {
       for (j = 0; j < ny1; j++)
       {
            index  = j * nx1 + i; 
            index1 = (j+20) * nxp + (i+20);
            pmapn[index]=pmap[index1];
        }
    }

    // =============== [STEP 6] =============== 
    // the R parameter is calculated
    for (i = 0; i < nx1; i++)
    {
        for (j = 0; j < ny1; j++)
        {
            index = j * nx1 + i;
            if isnan(pmapn[index]) continue;
            if isnan(rim[index]) continue;
            sum += pmapn[index]*abs(rim[index]);
        }
    }
    
    if (sum < 1.0)
        *Rparam = 0.0;
    else
        *Rparam = log10(sum);

    //printf("R_VALUE=%f\n",*Rparam);

    free_fresize(&fresboxcar);
    free_fresize(&fresgauss);

    return 0;

}

/*===========================================*/
/* Example function 16: Lorentz force as defined in Fisher, 2012 */
// 
// This calculation is adapted from Xudong's code
// at /proj/cgem/lorentz/apps/lorentz.c 

int computeLorentz(float *bx,  float *by, float *bz, float *fx, float *fy, float *fz, int *dims, 
                   float *totfx_ptr, float *totfy_ptr, float *totfz_ptr, float *totbsq_ptr,
                   float *epsx_ptr, float *epsy_ptr, float *epsz_ptr, int *mask, int *bitmask,
                   float cdelt1, double rsun_ref, double rsun_obs)

{ 

    int nx = dims[0];
    int ny = dims[1];
    int nxny = nx*ny;
    int j = 0;
    int index;
    double totfx = 0, totfy = 0, totfz = 0;
    double bsq = 0, totbsq = 0;
    double epsx = 0, epsy = 0, epsz = 0;
    double area = cdelt1*cdelt1*(rsun_ref/rsun_obs)*(rsun_ref/rsun_obs)*100.0*100.0;
    double k_h = -1.0 * area / (4. * PI) / 1.0e20;
    double k_z = area / (8. * PI) / 1.0e20;

    if (nx <= 0 || ny <= 0) return 1;        

    for (int i = 0; i < nxny; i++) 
    {  
       if ( mask[i] < 70 || bitmask[i] < 30 ) continue;
       if isnan(bx[i]) continue;
       if isnan(by[i]) continue;
       if isnan(bz[i]) continue;
       fx[i]  = bx[i] * bz[i] * k_h;
       fy[i]  = by[i] * bz[i] * k_h;
       fz[i]  = (bx[i] * bx[i] + by[i] * by[i] - bz[i] * bz[i]) * k_z;
       bsq    = bx[i] * bx[i] + by[i] * by[i] + bz[i] * bz[i];
       totfx  += fx[i]; totfy += fy[i]; totfz += fz[i];
       totbsq += bsq;
    }
   
    *totfx_ptr  = totfx;
    *totfy_ptr  = totfy;
    *totfz_ptr  = totfz;    
    *totbsq_ptr = totbsq;  
    *epsx_ptr   = (totfx / k_h) / totbsq;
    *epsy_ptr   = (totfy / k_h) / totbsq;
    *epsz_ptr   = (totfz / k_z) / totbsq;

    //printf("TOTBSQ=%f\n",*totbsq_ptr);

    return 0;
 
}

/*===========================================*/

/* Example function 17: Compute total unsigned flux in units of G/cm^2 on the LOS field */

//  To compute the unsigned flux, we simply calculate
//  flux = surface integral [(vector LOS) dot (normal vector)],
//       = surface integral [(magnitude LOS)*(magnitude normal)*(cos theta)].
//  However, since the field is radial, we will assume cos theta = 1.
//  Therefore the pixels only need to be corrected for the projection.

//  To convert G to G*cm^2, simply multiply by the number of square centimeters per pixel.
//  As an order of magnitude estimate, we can assign 0.5 to CDELT1 and 722500m/arcsec to (RSUN_REF/RSUN_OBS).
//  (Gauss/pix^2)(CDELT1)^2(RSUN_REF/RSUN_OBS)^2(100.cm/m)^2
//  =Gauss*cm^2

int computeAbsFlux_los(float *los, int *dims, float *absFlux_los,
                       float *mean_vf_los_ptr, float *count_mask_los_ptr,
                       int *bitmask, float cdelt1, double rsun_ref, double rsun_obs)

{
    
    int nx = dims[0];
    int ny = dims[1];
    int i = 0;
    int j = 0;
    int count_mask_los = 0;
    double sum = 0.0;
    *absFlux_los = 0.0;
    *mean_vf_los_ptr = 0.0;
    
     
    if (nx <= 0 || ny <= 0) return 1;
    
	for (i = 0; i < nx; i++)
	{
	   for (j = 0; j < ny; j++)
	   {
	    if ( bitmask[j * nx + i] < 30 ) continue;
            if isnan(los[j * nx + i]) continue;
            sum += (fabs(los[j * nx + i]));
            count_mask_los++;
	   }
	}
    
    *mean_vf_los_ptr     = sum*cdelt1*cdelt1*(rsun_ref/rsun_obs)*(rsun_ref/rsun_obs)*100.0*100.0;
    *count_mask_los_ptr  = count_mask_los;

    //printf("USFLUXL=%f\n",*mean_vf_los_ptr);
    //printf("CMASKL=%f\n",*count_mask_los_ptr);

    return 0;
}

/*===========================================*/
/* Example function 18:  Derivative of B_LOS (approximately B_vertical) = SQRT( ( dLOS/dx )^2 + ( dLOS/dy )^2 ) */

int computeLOSderivative(float *los, int *dims, float *mean_derivative_los_ptr, int *bitmask, float *derx_los, float *dery_los)
{
    
    int nx = dims[0];
    int ny = dims[1];
    int i = 0;
    int j = 0;
    int count_mask = 0;
    double sum = 0.0;
    *mean_derivative_los_ptr = 0.0;
    
    if (nx <= 0 || ny <= 0) return 1;
    
    /* brute force method of calculating the derivative (no consideration for edges) */
    for (i = 1; i <= nx-2; i++)
    {
	for (j = 0; j <= ny-1; j++)
        {
           derx_los[j * nx + i] = (los[j * nx + i+1] - los[j * nx + i-1])*0.5;
        }
    }
    
    /* brute force method of calculating the derivative (no consideration for edges) */
    for (i = 0; i <= nx-1; i++)
    {
        for (j = 1; j <= ny-2; j++)
        {
           dery_los[j * nx + i] = (los[(j+1) * nx + i] - los[(j-1) * nx + i])*0.5;
        }
    }
    
    /* consider the edges for the arrays that contribute to the variable "sum" in the computation below.
    ignore the edges for the error terms as those arrays have been initialized to zero. 
    this is okay because the error term will ultimately not include the edge pixels as they are selected out by the mask and bitmask arrays.*/
    i=0;
    for (j = 0; j <= ny-1; j++)
    {
        derx_los[j * nx + i] = ( (-3*los[j * nx + i]) + (4*los[j * nx + (i+1)]) - (los[j * nx + (i+2)]) )*0.5;
    }
    
    i=nx-1;
    for (j = 0; j <= ny-1; j++)
    {
        derx_los[j * nx + i] = ( (3*los[j * nx + i]) + (-4*los[j * nx + (i-1)]) - (-los[j * nx + (i-2)]) )*0.5;
    }
    
    j=0;
    for (i = 0; i <= nx-1; i++)
    {
        dery_los[j * nx + i] = ( (-3*los[j*nx + i]) + (4*los[(j+1) * nx + i]) - (los[(j+2) * nx + i]) )*0.5;
    }
    
    j=ny-1;
    for (i = 0; i <= nx-1; i++)
    {
        dery_los[j * nx + i] = ( (3*los[j * nx + i]) + (-4*los[(j-1) * nx + i]) - (-los[(j-2) * nx + i]) )*0.5;
    }
    
    
    for (i = 0; i <= nx-1; i++)
    {
        for (j = 0; j <= ny-1; j++)
        {
            if ( bitmask[j * nx + i] < 30 ) continue;
            if isnan(los[j * nx + i])      continue;
            if isnan(los[(j+1) * nx + i])  continue;
            if isnan(los[(j-1) * nx + i])  continue;
            if isnan(los[j * nx + i-1])    continue;
            if isnan(los[j * nx + i+1])    continue;
            if isnan(derx_los[j * nx + i]) continue;
            if isnan(dery_los[j * nx + i]) continue;
            sum += sqrt( derx_los[j * nx + i]*derx_los[j * nx + i]  + dery_los[j * nx + i]*dery_los[j * nx + i] ); /* Units of Gauss */
            count_mask++;
        }
    }
    
    *mean_derivative_los_ptr = (sum)/(count_mask); // would be divided by ((nx-2)*(ny-2)) if shape of count_mask = shape of magnetogram
    
    //printf("MEANGBL=%f\n",*mean_derivative_los_ptr);
    
	return 0;
}

/*==================KEIJI'S CODE =========================*/

// #include <omp.h>
#include <math.h>

void greenpot(float *bx, float *by, float *bz, int nnx, int nny)
{
    /* local workings */
    int inx, iny, i, j, n;
    /* local array */
    float *pfpot, *rdist;
    pfpot=(float *)malloc(sizeof(float) *nnx*nny);
    rdist=(float *)malloc(sizeof(float) *nnx*nny);
    float *bztmp;
    bztmp=(float *)malloc(sizeof(float) *nnx*nny);
    /* make nan */
    //  unsigned long long llnan = 0x7ff0000000000000;
    //  float NAN = (float)(llnan);
    
    // #pragma omp parallel for private (inx)
    for (iny=0; iny < nny; iny++){for (inx=0; inx < nnx; inx++){pfpot[nnx*iny+inx] = 0.0;}}
    // #pragma omp parallel for private (inx)
    for (iny=0; iny < nny; iny++){for (inx=0; inx < nnx; inx++){rdist[nnx*iny+inx] = 0.0;}}
    // #pragma omp parallel for private (inx)
    for (iny=0; iny < nny; iny++){for (inx=0; inx < nnx; inx++){bx[nnx*iny+inx] = 0.0;}}
    // #pragma omp parallel for private (inx)
    for (iny=0; iny < nny; iny++){for (inx=0; inx < nnx; inx++){by[nnx*iny+inx] = 0.0;}}
    // #pragma omp parallel for private (inx)
    for (iny=0; iny < nny; iny++){for (inx=0; inx < nnx; inx++)
    {
        float val0 = bz[nnx*iny + inx];
        if (isnan(val0)){bztmp[nnx*iny + inx] = 0.0;}else{bztmp[nnx*iny + inx] = val0;}
    }}
    
    // dz is the monopole depth
    float dz = 0.001;
    
    // #pragma omp parallel for private (inx)
    for (iny=0; iny < nny; iny++){for (inx=0; inx < nnx; inx++)
    {
        float rdd, rdd1, rdd2;
        float r;
        rdd1 = (float)(inx);
        rdd2 = (float)(iny);
        rdd = rdd1 * rdd1 + rdd2 * rdd2 + dz * dz;
        rdist[nnx*iny+inx] = 1.0/sqrt(rdd);
    }}
    
    int iwindow;
    if (nnx > nny) {iwindow = nnx;} else {iwindow = nny;}
    float rwindow;
    rwindow = (float)(iwindow);
    rwindow = rwindow * rwindow + 0.01; // must be of square
    
    rwindow = 1.0e2; // limit the window size to be 10.
    
    rwindow = sqrt(rwindow);
    iwindow = (int)(rwindow);
    
    // #pragma omp parallel for private(inx)
    for (iny=0;iny<nny;iny++){for (inx=0;inx<nnx;inx++)
    {
        float val0 = bz[nnx*iny + inx];
        if (isnan(val0))
        {
            pfpot[nnx*iny + inx] = 0.0; // hmmm.. NAN;
        }
        else
        {
            float sum;
            sum = 0.0;
            int j2, i2;
            int j2s, j2e, i2s, i2e;
            j2s = iny - iwindow;
            j2e = iny + iwindow;
            if (j2s <   0){j2s =   0;}
            if (j2e > nny){j2e = nny;}
            i2s = inx - iwindow;
            i2e = inx + iwindow;
            if (i2s <   0){i2s =   0;}
            if (i2e > nnx){i2e = nnx;}
            
            for (j2=j2s;j2<j2e;j2++){for (i2=i2s;i2<i2e;i2++)
            {
                float val1 = bztmp[nnx*j2 + i2];
                float rr, r1, r2;
                //        r1 = (float)(i2 - inx);
                //        r2 = (float)(j2 - iny);
                //        rr = r1*r1 + r2*r2;
                //        if (rr < rwindow)
                //        {
                int   di, dj;
                di = abs(i2 - inx);
                dj = abs(j2 - iny);
                sum = sum + val1 * rdist[nnx * dj + di] * dz;
                //        }
            } }
            pfpot[nnx*iny + inx] = sum; // Note that this is a simplified definition.
        }
    } } // end of OpenMP parallelism
    
    // #pragma omp parallel for private(inx)
    for (iny=1; iny < nny - 1; iny++){for (inx=1; inx < nnx - 1; inx++)
    {
        bx[nnx*iny + inx] = -(pfpot[nnx*iny + (inx+1)]-pfpot[nnx*iny + (inx-1)]) * 0.5;
        by[nnx*iny + inx] = -(pfpot[nnx*(iny+1) + inx]-pfpot[nnx*(iny-1) + inx]) * 0.5;
    } } // end of OpenMP parallelism
    
    free(rdist);
    free(pfpot);
    free(bztmp);
} // end of void func. greenpot


/*===========END OF KEIJI'S CODE =========================*/

char *sw_functions_version() // Returns CVS version of sw_functions.c
{
    return strdup("$Id");
}

/* ---------------- end of this file ----------------*/
