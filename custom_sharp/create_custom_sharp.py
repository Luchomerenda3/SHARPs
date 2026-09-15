import argparse
import typing

import astropy.units as unit
import matplotlib.pyplot as plt
import sunpy.map
from astropy.coordinates import SkyCoord
from sunpy.coordinates import HeliographicCarrington, propagate_with_solar_surface

from . import bvec2cea, bvecerr2cea
from utils.lu_tools import AreaSelector, cornerl_order


def get_custom_area_center_and_size(b_field: str):
    # select area to crop
    print("Choose the area to crop from the Br map")
    preview = sunpy.map.Map(b_field)
    preview.plot(title="Use mouse to select area and press Q after selection is made")

    selected_area = AreaSelector(plt.gcf(), plt.gca())

    final_coords = cornerl_order(selected_area.data_coords, "bltp", "np")

    bottom_left = preview.pixel_to_world(final_coords[0][0] * unit.pix,
                                         final_coords[0][1] * unit.pix)
    top_right = preview.pixel_to_world(final_coords[1][0] * unit.pix,
                                       final_coords[1][1] * unit.pix)

    # Make the submap and get the needed data from the map
    cropped_map = preview.submap(bottom_left=bottom_left, top_right=top_right)
    nx, ny = cropped_map.dimensions

    center = cropped_map.center.transform_to(new_frame=HeliographicCarrington(obstime=cropped_map.meta['date-obs']))

    return center, nx, ny


def create_custom_sharp_data(
        b_field: str, b_azi: str, b_incli: str, b_disambig: str,
        b_err_field: str, b_err_azi: str, b_err_incli: str,
        b_cc_field_incli: str, b_cc_field_azi: str, b_cc_incli_azi: str,
        coords: tuple[SkyCoord, int, int]
):
    center, nx, ny = coords
    phi_c, lambda_c = center.lon.to_value(unit.deg), center.lat.to_value(unit.deg)

    print("Creating custom bp, bt, br maps...")
    bp, bt, br, hd_out = bvec2cea.bvec2cea(
        infile_fld=b_field, infile_azi=b_azi, infile_inc=b_incli,
        infile_disamb=b_disambig,
        phi_c=phi_c, lambda_c=lambda_c,
        nx=nx, ny=ny,
    )
    print("Creating custom bp_err, bt_err, bp_err maps...")
    bp_err, bt_err, br_err, hd_err_out = bvecerr2cea.bvecerr2cea(
        infile_fld=b_field, infile_azi=b_azi, infile_inc=b_incli,
        infile_err_fld=b_err_field, infile_err_inc=b_err_incli, infile_err_azi=b_err_azi,
        infile_cc_fld_inc=b_cc_field_incli, infile_cc_fld_azi=b_cc_field_azi, infile_cc_inc_azi=b_cc_incli_azi,
        infile_disamb=b_disambig,
        phi_c=phi_c, lambda_c=lambda_c,
        nx=nx, ny=ny
    )

    return br, br_err, bp, bp_err, bt, bt_err, hd_out, hd_err_out


def run(b_field: str, b_azi: str, b_incli: str, b_disambig: str,
        b_err_field: str, b_err_azi: str, b_err_incli: str,
        b_cc_field_incli: str, b_cc_field_azi: str, b_cc_incli_azi: str,
        coords: tuple[SkyCoord, int, int] | None = None,
        output_path: str = '') -> tuple:
    """Create custom SHARP maps from full disk FITS set of field, azimuth, inclination & disambiguation data"""
    # If no coords input present force selection using the interactive selector
    if not coords:
        coords = get_custom_area_center_and_size(b_field)

    print("Creating custom SHARP maps...")
    br, br_err, bp, bp_err, bt, bt_err, hd_out, hd_err_out = create_custom_sharp_data(
        b_field, b_azi, b_incli,
        b_disambig,
        b_err_field, b_err_azi, b_err_incli,
        b_cc_field_incli, b_cc_field_azi, b_cc_incli_azi,
        coords)

    br_map = sunpy.map.Map(br, hd_out)
    bt_map = sunpy.map.Map(bt, hd_out)
    bp_map = sunpy.map.Map(bp, hd_out)
    br_err = sunpy.map.Map(br_err, hd_err_out)
    bt_err = sunpy.map.Map(bt_err, hd_err_out)
    bp_err = sunpy.map.Map(bp_err, hd_err_out)

    if output_path == '':
        # if no output_path let's save this in current working dir
        output_path = './'

    print('Saving custom SHARP maps...')
    pre_filename = f"{output_path}/hmi.sharp_cea_720s.custom_harp.{hd_out['t_rec']}"
    br_map.save(pre_filename + ".Br.fits")
    bt_map.save(pre_filename + ".Bt.fits")
    bp_map.save(pre_filename + ".Bp.fits")
    br_err.save(pre_filename + ".Br_err.fits")
    bt_err.save(pre_filename + ".Bt_err.fits")
    bp_err.save(pre_filename + ".Bp_err.fits")

    return br_map,bt_map,bp_map,br_err,bt_err,bp_err


def run_series(b_field_arr: list[str], b_azi_arr: list[str], b_incli_arr: list[str], b_disambig_arr: list[str],
               b_err_field_arr: str, b_err_azi_arr: str, b_err_incli_arr: str,
               b_cc_field_incli_arr: str, b_cc_field_azi_arr: str, b_cc_incli_azi_arr: str,
               coords: tuple[SkyCoord, int, int] | None = None,
               output_path: str = ''):
    # If no coords input present force selection using the interactive selector from the first image in the series.
    if not coords:
        coords = get_custom_area_center_and_size(b_field_arr[0])

    for i, b_field, b_azi, b_incli, b_disambig, b_err_field, b_err_azi, b_err_incli, b_cc_field_incli, b_cc_field_azi, b_cc_incli_azi in zip(
            range(len(b_field_arr)), b_field_arr, b_azi_arr, b_incli_arr,
            b_disambig_arr, b_err_field_arr, b_err_azi_arr, b_err_incli_arr,
            b_cc_field_incli_arr, b_cc_field_azi_arr, b_cc_incli_azi_arr):

        # Adjust every frame crop coordinates for solar rotation before running
        new_final_coords = coords

        if i != 0:
            hdr = sunpy.map.Map(b_field_arr[i]).meta
            map_newframe = HeliographicCarrington(obstime=hdr['date-obs'])
            with propagate_with_solar_surface():
                center_coord, nx, ny = coords
                updated_center_coord = center_coord.transform_to(map_newframe)
                new_final_coords = (updated_center_coord, nx, ny)

        run(b_field, b_azi, b_incli, b_disambig, b_err_field, b_err_azi, b_err_incli, b_cc_field_incli, b_cc_field_azi,
            b_cc_incli_azi, coords=new_final_coords, output_path=output_path)


def main(argv: typing.Sequence[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Create custom SHARP maps from full disk vector magnetic field FITS data."
    )

    # Required FITS input files
    parser.add_argument("-f", "--b-field", "--b_field", dest="b_field", required=True, type=str,
                        help="Path to the magnetic field strength FITS file")
    parser.add_argument("-a", "--b-azi", "--b_azi", dest="b_azi", required=True, type=str,
                        help="Path to the magnetic field azimuth FITS file")
    parser.add_argument("-i", "--b-incli", "--b_incli", dest="b_incli", required=True, type=str,
                        help="Path to the magnetic field inclination FITS file")
    parser.add_argument("-d", "--b-disambig", "--b_disambig", dest="b_disambig", required=True, type=str,
                        help="Path to the disambiguation FITS file")
    parser.add_argument("--b-err-field", "--b_err_field", dest="b_err_field", required=True, type=str,
                        help="Path to the magnetic field strength error FITS file")
    parser.add_argument("--b-err-azi", "--b_err_azi", dest="b_err_azi", required=True, type=str,
                        help="Path to the magnetic field azimuth error FITS file")
    parser.add_argument("--b-err-incli", "--b_err_incli", dest="b_err_incli", required=True, type=str,
                        help="Path to the magnetic field inclination error FITS file")
    parser.add_argument("--b-cc-field-incli", "--b_cc_field_incli", dest="b_cc_field_incli", required=True,
                        type=str, help="Path to the field-strength/inclination correlation FITS file")
    parser.add_argument("--b-cc-field-azi", "--b_cc_field_azi", dest="b_cc_field_azi", required=True,
                        type=str, help="Path to the field-strength/azimuth correlation FITS file")
    parser.add_argument("--b-cc-incli-azi", "--b_cc_incli_azi", dest="b_cc_incli_azi", required=True,
                        type=str, help="Path to the inclination/azimuth correlation FITS file")
    parser.add_argument("--center-phi", "--center_phi", dest="center_phi", type=float,
                        help="Carrington longitude of the crop center, in degrees")
    parser.add_argument("--center-lambda", "--center_lambda", dest="center_lambda", type=float,
                        help="Carrington latitude of the crop center, in degrees")
    parser.add_argument("--nx", type=int, help="Crop width in pixels")
    parser.add_argument("--ny", type=int, help="Crop height in pixels")
    parser.add_argument("--output-path", "--output_path", dest="output_path", required=True, type=str)

    args = parser.parse_args(argv)

    # Check coords input in cli command.
    crop_definition = (args.center_phi, args.center_lambda, args.nx, args.ny)
    if any(value is not None for value in crop_definition) and not all(value is not None for value in crop_definition):
        parser.error("--center-phi, --center-lambda, --nx, and --ny must be provided together")

    coords = None
    if all(value is not None for value in crop_definition):
        center_coords = SkyCoord(
            lon=args.center_phi * unit.deg,
            lat=args.center_lambda * unit.deg,
            frame=HeliographicCarrington,
        )
        coords = (center_coords, args.nx, args.ny)

    run(
        b_field=args.b_field,
        b_azi=args.b_azi,
        b_incli=args.b_incli,
        b_disambig=args.b_disambig,
        b_err_field=args.b_err_field,
        b_err_azi=args.b_err_azi,
        b_err_incli=args.b_err_incli,
        b_cc_field_incli=args.b_cc_field_incli,
        b_cc_field_azi=args.b_cc_field_azi,
        b_cc_incli_azi=args.b_cc_incli_azi,
        coords=coords,
        output_path=args.output_path,
    )

    print(f"All Br, Bp, Bt, Br_err, Bp_err, Bt_err generated & saved in {args.output_path}")


if __name__ == "__main__":
    main()
