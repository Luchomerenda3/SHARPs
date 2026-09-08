#include <math.h>

int img2helioVector (double bxImg, double byImg, double bzImg, double *bxHelio, 
                     double *byHelio, double *bzHelio, double lon, double lat,
                     double lonc, double latc, double pAng) {
/*
*     perform tramsformation of a vector from image location (lon, lat) to
*     heliographic center. The formula is from Hagyard (1987), and further
*     developed by Gary & Hagyard (1990).
*
* Arguments:
*
*    bxImg, byImg, bzImg: three components of vector magnetic field on image
*                         coordinates.
*    lon, lat:            heliographic coordinates of the location where the vector field
*                         measured. They are in radians.
*    lonc, latc:          heliographic coordinates of the image disk center. They are in
*                         radians.
*    pAng:                position angle of the heliographic north pole, measured eastward
*                         from the north. It's in radians.   
*/

    static double raddeg = M_PI / 180.;
    double a11, a12, a13, a21, a22, a23, a31, a32, a33;

    a11 = -sin(latc) * sin(pAng) * sin(lon - lonc) + cos(pAng) * cos(lon - lonc);
    a12 =  sin(latc) * cos(pAng) * sin(lon - lonc) + sin(pAng) * cos(lon - lonc);
    a13 = -cos(latc) * sin(lon - lonc);
    a21 = -sin(lat) * (sin(latc) * sin(pAng) * cos(lon - lonc) + cos(pAng) * sin(lon - lonc)) 
          - cos(lat) * cos(latc) * sin(pAng);
    a22 =  sin(lat) * (sin(latc) * cos(pAng) * cos(lon - lonc) - sin(pAng) * sin(lon - lonc))
          + cos(lat) * cos(latc) * cos(pAng);
    a23 = -cos(latc) * sin(lat) * cos(lon - lonc) + sin(latc) * cos(lat);
    a31 =  cos(lat) * (sin(latc) * sin(pAng) * cos(lon - lonc) + cos(pAng) * sin(lon - lonc))
          - sin(lat) * cos(latc) * sin(pAng);
    a32 = -cos(lat) * (sin(latc) * cos(pAng) * cos(lon - lonc) - sin(pAng) * sin(lon - lonc))
          + sin(lat) * cos(latc) * cos(pAng);
    a33 =  cos(lat) * cos(latc) * cos(lon - lonc) + sin(lat) * sin(latc);

    *bxHelio = a11 * bxImg + a12 * byImg + a13 * bzImg;
    *byHelio = a21 * bxImg + a22 * byImg + a23 * bzImg;
    *bzHelio = a31 * bxImg + a32 * byImg + a33 * bzImg;

return 0;
}
