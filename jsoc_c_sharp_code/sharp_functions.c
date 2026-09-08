/* sharp_functions.c */
#define PI (3.141592653589793)

/*===========================================*/

/* Example function 1: Compute total unsigned flux in units of G/cm^2, Mean Vertical Field */

//  To convert G to G/cm^2, simply multiply by the number of square centimeters per pixel.
//  As an order of magnitude estimate, we can assign 0.5 to CDELT1 and 722500m/arcsec to (RSUN_REF/RSUN_OBS).
//  Gauss(CDELT1)(RSUN_REF/RSUN_OBS)(100.)
//  = Gauss(0.5 arcsec/pix)^2(722500m/arcsec)^2(100cm/m)^2
//  = Gauss(1.30501e15)

int computeAbsFlux(float *bz, int *dims, float *absFlux, float *mean_vf_ptr, int *mask)
{

    int nx = dims[0], ny = dims[1];
    int i, j, count_mask=0;
    double sum=0.0;	
	
    if (nx <= 0 || ny <= 0) return 1;
	
    *absFlux = 0.0;
    *mean_vf_ptr =0.0;

	for (j = 0; j < ny; j++) 
	{
		for (i = 0; i < nx; i++) 
		{
                  if (mask[j * nx + i] <= 1) continue;
                  sum += (fabs(bz[j * nx + i]));
                  count_mask++;
		}	
	}

     printf("nx=%d,ny=%d,count_mask=%d,sum=%f\n",nx,ny,count_mask,sum);
     *mean_vf_ptr = sum*(1.30501e15);
     return 0;
}

/*===========================================*/
/* Example function 2: Calculate Bh in units of Gauss */
// Native units of Bh are Gauss

int computeBh(float *bx, float *by, float *bz, float *bh, int *dims, float *mean_hf_ptr, int *mask)
{

    int nx = dims[0], ny = dims[1];
    int i, j, count_mask=0;
    float sum=0.0;	
    *mean_hf_ptr =0.0;

    if (nx <= 0 || ny <= 0) return 1;

	for (j = 0; j < ny; j++) 
	  {
	    for (i = 0; i < nx; i++)
	      {
		bh[j * nx + i] = sqrt( bx[j * nx + i]*bx[j * nx + i] + by[j * nx + i]*by[j * nx + i] );
                sum += bh[j * nx + i];
                count_mask++;
	      }	
	  }
     
    *mean_hf_ptr = sum/(count_mask); // would be divided by nx*ny if shape of count_mask = shape of magnetogram
    printf("*mean_hf_ptr=%f\n",*mean_hf_ptr);
    return 0;
}

/*===========================================*/

/* Example function 3: Calculate Gamma in units of degrees 
// Native units of atan(x) are in radians; to convert from radians to degrees, multiply by (180./PI)

this is wrong, also the divide by nx*ny is wrong */

int computeGamma(float *bx, float *by, float *bz, float *bh, int *dims, float *mean_gamma_ptr, int *mask)
{
    int nx = dims[0], ny = dims[1];
    int i, j, count_mask=0;

    if (nx <= 0 || ny <= 0) return 1;
	
    *mean_gamma_ptr=0.0;
    float sum=0.0;
    int count=0;

	for (i = 0; i < nx; i++) 
	  {
	    for (j = 0; j < ny; j++) 
	      {
		if (bh[j * nx + i] > 100)
		  {
                    if (mask[j * nx + i] <= 1) continue;
		    sum += (atan (fabs( bz[j * nx + i] / bh[j * nx + i] ))* (180./PI));
		    count_mask++;
		  }
	      }
	  }

     *mean_gamma_ptr = sum/count_mask;
     printf("*mean_gamma_ptr=%f\n",*mean_gamma_ptr);
     return 0;
}

/*===========================================*/

/* Example function 4: Calculate B_Total*/
// Native units of B_Total are in gauss

int computeB_total(float *bx, float *by, float *bz, float *bt, int *dims, int *mask)
{

    int nx = dims[0], ny = dims[1];
    int i, j, count_mask=0;
	
    if (nx <= 0 || ny <= 0) return 1;

	for (i = 0; i < nx; i++) 
	  {
	    for (j = 0; j < ny; j++) 
	      {
		bt[j * nx + i] = sqrt( bx[j * nx + i]*bx[j * nx + i] + by[j * nx + i]*by[j * nx + i] + bz[j * nx + i]*bz[j * nx + i]);
	      }	
	  }
     return 0;
}

/*===========================================*/

/* Example function 5:  Derivative of B_Total SQRT( (dBt/dx)^2 + (dBt/dy)^2 ) */

int computeBtotalderivative(float *bt, int *dims, float *mean_derivative_btotal_ptr, int *mask)
{

    int nx = dims[0], ny = dims[1];
    int i, j, count_mask=0;

    if (nx <= 0 || ny <= 0) return 1;

    *mean_derivative_btotal_ptr = 0.0;
    float derx, dery, sum = 0.0;

	for (i = 1; i < nx-1; i++) 
	  {
	    for (j = 1; j < ny-1; j++) 
	      {
                if (mask[j * nx + i] <= 1) continue;
		/* Missing a (*0.5) */
		derx = bt[j * nx + i+1] - bt[j * nx + i-1];
		dery = bt[(j+1) * nx + i] - bt[(j-1) * nx + i];
		sum += sqrt(derx*derx + dery*dery);
                count_mask++;
	      }	
	  }

        *mean_derivative_btotal_ptr = (sum)/(count_mask); // would be divided by ((nx-2)*(ny-2)) if shape of count_mask = shape of magnetogram
     printf("*mean_derivative_btotal_ptr=%f\n",*mean_derivative_btotal_ptr);
     return 0;
}

/*===========================================*/
/* Example function 6:  Derivative of Bh SQRT( (dBh/dx)^2 + (dBh/dy)^2 ) */

int computeBhderivative(float *bh, int *dims, float *mean_derivative_bh_ptr, int *mask)
{

        int nx = dims[0], ny = dims[1];
        int i, j, count_mask=0;

        if (nx <= 0 || ny <= 0) return 1;

        *mean_derivative_bh_ptr = 0.0;
        float derx, dery, sum = 0.0;

        for (i = 1; i < nx-1; i++)
          {
            for (j = 1; j < ny-1; j++)
              {
                if (mask[j * nx + i] <= 1) continue;
                /* Missing a (*0.5) */
                derx = bh[j * nx + i+1] - bh[j * nx + i-1];
                //printf("derx=%f\n",derx);
                dery = bh[(j+1) * nx + i] - bh[(j-1) * nx + i];
                //                printf("dery=%f\n",dery);
                sum += sqrt(derx*derx + dery*dery);
                //printf("sum=%f\n",sum);
                count_mask++;
                //printf("count_mask=%d\n",count_mask);
              }
          }

        *mean_derivative_bh_ptr = (sum)/(count_mask); // would be divided by ((nx-2)*(ny-2)) if shape of count_mask = shape of magnetogram
        printf("*mean_derivative_bh_ptr=%f,nx=%d,ny=%d,sum=%f\n",*mean_derivative_bh_ptr,nx,ny,sum);
        return 0;
}

/*===========================================*/
/* Example function 7:  Derivative of B_vertical SQRT( (dBz/dx)^2 + (dBz/dy)^2 ) */

int computeBzderivative(float *bz, int *dims, float *mean_derivative_bz_ptr, int *mask)
{

	int nx = dims[0], ny = dims[1];
	int i, j, count_mask=0;

	if (nx <= 0 || ny <= 0) return 1;

	*mean_derivative_bz_ptr = 0.0;
	float derx, dery, sum = 0.0;

	for (i = 1; i < nx-1; i++) 
	  {
	    for (j = 1; j < ny-1; j++) 
	      {
                if (mask[j * nx + i] <= 1) continue;
		/* Missing a (*0.5) */
		derx = bz[j * nx + i+1] - bz[j * nx + i-1];
		dery = bz[(j+1) * nx + i] - bz[(j-1) * nx + i];
		sum += sqrt(derx*derx + dery*dery);
                count_mask++;
	      }	
	  }

	*mean_derivative_bz_ptr = (sum)/(count_mask); // would be divided by ((nx-2)*(ny-2)) if shape of count_mask = shape of magnetogram

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
//  (Gauss/pix)(pix/arcsec)(arcsec/meter)(Newton/Gauss*Ampere*meter)(Ampere^2/Newton)(milliAmpere/Ampere), or
//  (Gauss/pix)(1/CDELT1)(RSUN_OBS/RSUN_REF)(1 T / 10^4 Gauss)(1 / 4*PI*10^-7)( 10^3 milliAmpere/Ampere),
//  where a Tesla is represented as a Newton/Ampere*meter.
//  As an order of magnitude estimate, we can assign 0.5 to CDELT1 and 722500m/arcsec to (RSUN_REF/RSUN_OBS).
//  In that case, we would have the following:
//  (Gauss/pix)(1/0.5)(1/722500)(10^-4)(4*PI*10^7)(10^3), or
//  jz * (35.0)
//
//  The units of total unsigned vertical current (us_i) are simply in A. In this case, we would have the following:
//  (Gauss/pix)(1/0.5)(1/722500)(10^-4)(4*PI*10^7)(722500)(722500), or
//  (Gauss/pix)(1.81*10^10)
//  (Gauss/pix)(18100000000.)

int computeJz(float *bx, float *by, int *dims, float *jz, float *mean_jz_ptr, float *us_i_ptr, int *mask)
{

	int nx = dims[0], ny = dims[1];
	int i, j, count_mask=0;

	if (nx <= 0 || ny <= 0) return 1;

	*mean_jz_ptr = 0.0;
	float derx, dery, curl=0.0, us_i=0.0,test_perimeter=0.0,mean_curl=0.0;

	for (i = 1; i < nx-1; i++) 
	  {
	    for (j = 1; j < ny-1; j++) 
	      {
                if (mask[j * nx + i] <= 1) continue;
                derx = (by[j * nx + i+1] - by[j * nx + i-1])*0.5;
		dery = (bx[(j+1) * nx + i] - bx[(j-1) * nx + i])*0.5;
                curl += (derx-dery)*(35.0);               /* curl is in units of mA/m^2 */
                jz[j * nx + i] = (derx-dery);             /* jz is in units of Gauss per pixel */
                us_i += fabs(derx-dery)*(18100000000.) ;  /* us_i is in units of A */
                count_mask++;
	      }	
	  }

        mean_curl      = (curl/count_mask);
        printf("mean_curl=%f\n",mean_curl);
        *mean_jz_ptr     = curl/(count_mask);
        printf("count_mask=%d\n",count_mask);
        *us_i_ptr = (us_i);
	return 0;
}



/*===========================================*/
/* Example function 9:  Twist Parameter, alpha */

// The twist parameter, alpha, is defined as alpha = Jz/Bz and the units are in 1/Mm
// The units of Jz are in G/pix; the units of Bz are in G.
// Therefore, the units of Jz/Bz = (pix/arcsec)(arcsec/meter)(meter/Mm), or 
//                         Jz/Bz = (Gauss/pix)(1/CDELT1)(RSUN_OBS/RSUN_REF)(10^6).
//  As an order of magnitude estimate, we can assign 0.5 to CDELT1 and 722500m/arcsec to (RSUN_REF/RSUN_OBS).
//  In that case, we would have the following:
//  (Gauss/pix)(1/0.5)(1/722500)(10^6), or
//  (Gauss/pix)*2.7

int computeAlpha(float *bz, int *dims, float *jz, float *mean_alpha_ptr, int *mask)
{
	int nx = dims[0], ny = dims[1];
	int i, j, count_mask=0;

	if (nx <= 0 || ny <= 0) return 1;

	*mean_alpha_ptr = 0.0;
	float aa, bb, cc, bznew, alpha2, sum=0.0;

	for (i = 1; i < nx-1; i++) 
	  {
	    for (j = 1; j < ny-1; j++) 
	      {
                if (mask[j * nx + i] <= 1) continue;
                if isnan(jz[j * nx + i]) continue;
                if isnan(bz[j * nx + i]) continue;
                if (bz[j * nx + i] == 0.0) continue;
                sum += (jz[j * nx + i] / bz[j * nx + i])*2.7 ; /* the units for (jz) Gauss/pix; the units for bz are Gauss */
                //printf("sum=%f\n",sum);
                //printf("jz[j * nx + i]=%f\n",jz[j * nx + i]);
                //printf("bz[j * nx + i]=%f\n",bz[j * nx + i]);
                //printf("(jz[j * nx + i] / bz[j * nx + i])*2.7=%f\n",(jz[j * nx + i] / bz[j * nx + i])*2.7);
                count_mask++;
                //printf("count_mask=%d\n",count_mask);
	      }	
	  }

        printf("count_mask=%d\n",count_mask);
        printf("sum=%f\n",sum);
	*mean_alpha_ptr = sum/count_mask; /* Units are 1/Mm */
	return 0;
}

/*===========================================*/

/* Example function 10:  Helicity (mean current helicty, mean unsigned current helicity, and mean absolute current helicity) */

//  The current helicity is defined as Bz*Jz and the units are G^2 / m
//  The units of Jz are in G/pix; the units of Bz are in G.
//  Therefore, the units of Bz*Jz = (Gauss)*(Gauss/pix) = (Gauss^2/pix)(1/CDELT1)(RSUN_OBS/RSUN_REF) = G^2 / m.
//  As an order of magnitude estimate, we can assign 0.5 to CDELT1 and 722500m/arcsec to (RSUN_REF/RSUN_OBS).
//  In that case, we would have the following:
//  (Gauss/pix)(1/0.5)(1/722500), or
//  (Gauss/pix)*0.000002768166

int computeHelicity(float *bz, int *dims, float *jz, float *mean_ih_ptr, float *total_us_ih_ptr, float *total_abs_ih_ptr, int *mask)
{


	int nx = dims[0], ny = dims[1];
	int i, j, count_mask=0;
	
	if (nx <= 0 || ny <= 0) return 1;

	*mean_ih_ptr = 0.0;
	float sum=0.0, sum2=0.0;

	for (j = 0; j < ny; j++) 
	{
		for (i = 0; i < nx; i++) 
		{
                if (mask[j * nx + i] <= 1) continue;
                if isnan(jz[j * nx + i]) continue;
                if isnan(bz[j * nx + i]) continue;
                if (bz[j * nx + i] == 0.0) continue;
                if (jz[j * nx + i] == 0.0) continue;
		sum  += (jz[j * nx + i])*(bz[j * nx + i])*(0.000002768166);
		sum2 += fabs(jz[j * nx + i]*(bz[j * nx + i]))*(0.000002768166);
                count_mask++;
                printf("(jz[j * nx + i])=%f\n",(jz[j * nx + i]));
                printf("(bz[j * nx + i])=%f\n",(bz[j * nx + i]));
                printf("(jz[j * nx + i])*(bz[j * nx + i])*(0.000002768166)=%f\n",(jz[j * nx + i])*(bz[j * nx + i])*(0.000002768166));
                printf("sum=%f\n",sum);
                printf("sum2=%f\n",sum2);
                printf("count_mask=%d\n",count_mask);
	        }	
	 }

            printf("sum/count_mask=%f\n",sum/count_mask);
	    *mean_ih_ptr     = sum/count_mask; /* Units are G^2 / m ; keyword is MEANJZH */ 
	    *total_us_ih_ptr = sum2;           /* Units are G^2 / m */
	    *total_abs_ih_ptr= fabs(sum);      /* Units are G^2 / m */

	return 0;
}

/*===========================================*/

/* Example function 11:  Sum of Absolute Value per polarity */

//  The Sum of the Absolute Value per polarity is defined as the following:
//  fabs(sum(jz gt 0)) + fabs(sum(jz lt 0)) and the units are in Amperes.
//  The units of jz are in G/pix. In this case, we would have the following:
//  (Gauss/pix)(1/0.5)(1/722500)(10^-4)(4*PI*10^7)(722500)(722500), or
//  (Gauss/pix)(1.81*10^10)
//  (Gauss/pix)(18100000000.)

int computeSumAbsPerPolarity(float *bz, float *jz, int *dims, float *totaljzptr, int *mask)
{

	
	int nx = dims[0], ny = dims[1];
	int i, j, count_mask=0;
	
	if (nx <= 0 || ny <= 0) return 1;
	
	*totaljzptr=0.0;
	float sum1=0.0, sum2=0.0;

	for (i = 0; i < nx; i++) 
	  {
	    for (j = 0; j < ny; j++) 
	      {
                if (mask[j * nx + i] <= 1) continue; 
		if (bz[j * nx + i] >  0) sum1 += ( jz[j * nx + i]*18100000000.);
		if (bz[j * nx + i] <= 0) sum2 += ( jz[j * nx + i]*18100000000.);
	      }
	  }
	
	*totaljzptr = fabs(sum1)+fabs(sum2);  /* Units are A */
	return 0;
}

/*===========================================*/

/* Example function 12:  Mean photospheric excess magnetic energy and total photospheric excess magnetic energy density */
// The units for magnetic energy density in cgs are ergs per cubic centimeter. The formula B^2/8*PI integrated over all space, dV
// automatically yields erg per cubic centimeter for an input B in Gauss.
//
// Total magnetic energy is the magnetic energy density times dA, or the area, and the units are thus ergs/cm. To convert
// ergs per centimeter cubed to ergs per centimeter, simply multiply by the area per pixel in cm:
// erg/cm^3(CDELT1)(RSUN_REF/RSUN_OBS)(100.)
// = erg/cm^3(0.5 arcsec/pix)^2(722500m/arcsec)^2(100cm/m)^2
// = erg/cm^3(1.30501e15)
// = erg/cm(1/pix^2)

int computeFreeEnergy(float *bx, float *by, float *bpx, float *bpy, int *dims, float *meanpotptr, float *totpotptr, int *mask)
{
	int nx = dims[0], ny = dims[1];
	int i, j, count_mask=0;
	
	if (nx <= 0 || ny <= 0) return 1;
	
        *totpotptr=0.0;
	*meanpotptr=0.0;
	float sum=0.0;

	for (i = 0; i < nx; i++) 
	  {
	    for (j = 0; j < ny; j++) 
	      {
                 if (mask[j * nx + i] <= 1) continue;
                 sum += ((    ((bx[j * nx + i])*(bx[j * nx + i]) + (by[j * nx + i])*(by[j * nx + i]) ) -  ((bpx[j * nx + i])*(bpx[j * nx + i]) + (bpy[j * nx + i])*(bpy[j * nx + i]))  )/8.*PI);
                 count_mask++;
	      }
	  }

	*meanpotptr = (sum)/(count_mask);              /* Units are ergs per cubic centimeter */
        *totpotptr  = sum*(1.30501e15)*(count_mask);   /* Units of sum are ergs/cm^3, units of factor are cm^2/pix^2, units of count_mask are pix^2; therefore, units of totpotptr are ergs per centimeter */
	return 0;
}

/*===========================================*/
/* Example function 13:  Mean 3D shear angle, area with shear greater than 45, mean horizontal shear angle, area with horizontal shear angle greater than 45 */

int computeShearAngle(float *bx, float *by, float *bz, float *bpx, float *bpy, float *bpz, int *dims, float *meanshear_angleptr, float *area_w_shear_gt_45ptr, float *meanshear_anglehptr, float *area_w_shear_gt_45hptr, int *mask)
{	
	int nx = dims[0], ny = dims[1];
	int i, j, count_mask=0;
	
	if (nx <= 0 || ny <= 0) return 1;
	
        *area_w_shear_gt_45ptr=0.0;
	*meanshear_angleptr=0.0;
	float dotproduct, magnitude_potential, magnitude_vector, shear_angle=0.0, sum = 0.0, count=0.0;
        float dotproducth, magnitude_potentialh, magnitude_vectorh, shear_angleh=0.0, sum1 = 0.0, counth = 0.0;

	for (i = 0; i < nx; i++) 
	  {
	    for (j = 0; j < ny; j++) 
	      {
                 if (mask[j * nx + i] <= 1) continue;
                 /* For mean 3D shear angle, area with shear greater than 45*/
                 dotproduct            = (bpx[j * nx + i])*(bx[j * nx + i]) + (bpy[j * nx + i])*(by[j * nx + i]) + (bpz[j * nx + i])*(bz[j * nx + i]);
                 magnitude_potential   = sqrt((bpx[j * nx + i]*bpx[j * nx + i]) + (bpy[j * nx + i]*bpy[j * nx + i]) + (bpz[j * nx + i]*bpz[j * nx + i]));
                 magnitude_vector      = sqrt( (bx[j * nx + i]*bx[j * nx + i]) + (by[j * nx + i]*by[j * nx + i]) + (bz[j * nx + i]*bz[j * nx + i]) );
                 shear_angle           = acos(dotproduct/(magnitude_potential*magnitude_vector))*(180./PI);
                 sum += shear_angle ;
                 if (shear_angle > 45) count++;

                 /* For mean horizontal shear angle, area with horizontal shear angle greater than 45 */ 
                 dotproducth           = (bpx[j * nx + i])*(bx[j * nx + i]) + (bpy[j * nx + i])*(by[j * nx + i]);
                 magnitude_potentialh  = sqrt((bpx[j * nx + i]*bpx[j * nx + i]) + (bpy[j * nx + i]*bpy[j * nx + i]));
                 magnitude_vectorh     = sqrt( (bx[j * nx + i]*bx[j * nx + i]) + (by[j * nx + i]*by[j * nx + i]) );
                 shear_angleh          = acos(dotproduct/(magnitude_potential*magnitude_vector))*(180./PI);
                 sum1 += shear_angleh ;
                 if (shear_angleh > 45) counth++;
                 count_mask++;
	      }
	  }
	
        /* For mean 3D shear angle, area with shear greater than 45*/
	*meanshear_angleptr = (sum)/(count_mask);              /* Units are degrees */
        *area_w_shear_gt_45ptr = (count/(count_mask))*(100.);  /* The area here is a fractional area -- the % of the total area */

        //        /* For mean horizontal shear angle, area with horizontal shear angle greater than 45 */ 
        //        *meanshear_anglehptr = (sum1)/(count_mask);              /* Units are degrees */
        //        *area_w_shear_gt_45hptr = (counth/(count_mask))*(100.);  /* The area here is a fractional area -- the % of the total area */
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
/* make nan */
  unsigned long long llnan = 0x7ff0000000000000;
  //  float NAN = (float)(llnan);

// #pragma omp parallel for private (inx)
  for (iny=0; iny < nny; iny++){for (inx=0; inx < nnx; inx++){pfpot[nnx*iny+inx] = 0.0;}}
// #pragma omp parallel for private (inx)
  for (iny=0; iny < nny; iny++){for (inx=0; inx < nnx; inx++){rdist[nnx*iny+inx] = 0.0;}}
// #pragma omp parallel for private (inx)
  for (iny=0; iny < nny; iny++){for (inx=0; inx < nnx; inx++){bx[nnx*iny+inx] = 0.0;}}
// #pragma omp parallel for private (inx)
  for (iny=0; iny < nny; iny++){for (inx=0; inx < nnx; inx++){by[nnx*iny+inx] = 0.0;}}

float dz;
dz = params_get_float(&cmdparams, "dzvalue");

// float dz = 0.0001;

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
  rwindow = rwindow * rwindow + 0.01;
// #pragma omp parallel for private(inx)
  for (iny=0;iny<nny;iny++){for (inx=0;inx<nnx;inx++)
  {
    float val0 = bz[nnx*iny + inx];
    if (isnan(val0))
    {
      pfpot[nnx*iny + inx] = NAN;
    }
    else
    {
      float sum;
      sum = 0.0;
      int j2, i2;
      for (j2=0;j2<nny;j2++){for (i2=0;i2<nnx;i2++)
      {
        float val1 = bz[nnx*j2 + i2];
        float rr, r1, r2;
        r1 = (float)(i2 - inx);
        r2 = (float)(j2 - iny);
        rr = r1*r1 + r2*r2;
        if ((!isnan(val1)) && (rr < rwindow))
        {
          int   di, dj;
          di = abs(i2 - inx);
          dj = abs(j2 - iny);
          sum = sum + val1 * rdist[nnx * dj + di] * dz;
        }
      } }
      pfpot[nnx*iny + inx] = sum; // Note that this is a simplified definition.
    }
  } } // end of OpenMP parallelism

// #pragma omp parallel for private(inx)
  for (iny=1; iny < nny - 1; iny++){for (inx=1; inx < nnx - 1; inx++)
  {
    bx[nnx*iny + inx] = -(pfpot[nnx*iny + (inx+1)]-pfpot[nnx*iny + (inx-1)]) / 2.0;
    by[nnx*iny + inx] = -(pfpot[nnx*(iny+1) + inx]-pfpot[nnx*(iny-1) + inx]) / 2.0;
  } }
  free(pfpot);
} // end of void func. greenpot


/*===========END OF KEIJI'S CODE =========================*/
/* ---------------- end of this file ----------------*/
