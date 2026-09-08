import argparse
import typing

import astropy.units as unit
import matplotlib.pyplot as plt
import sunpy.map
from sunpy.coordinates import HeliographicCarrington, propagate_with_solar_surface

from utils.lu_tools import AreaSelector, cornerl_order


def select_custom_area(b_field: str):
    # select area to crop
    print("Choose the area to crop from the Br map")
    preview = sunpy.map.Map(b_field)
    preview.plot(title="Use mouse to select and press Q after selection is made")
    area_to_select = AreaSelector(plt.gcf(), plt.gca())

    # Give crop coordinates in bottomleft, topright order,
    # in order to use them with sunpy's map submap method.
    # Crop coordinates are selected for first image, all following crop
    # coordinates are corrected for solar rotation
    final_coords = cornerl_order(area_to_select.data_coords, "bltp", "np")

    bottomleft = preview.pixel_to_world(final_coords[0][0] * unit.pix,
                                        final_coords[0][1] * unit.pix)
    topright = preview.pixel_to_world(final_coords[1][0] * unit.pix,
                                      final_coords[1][1] * unit.pix)

    # Make the submap
    crop = preview.submap(bottom_left=bottomleft, top_right=topright)
    pass


def create_custom_map():
    return br, br_err, bp, bp_err, bt, bt_err


def run(b_field: str, b_azi: str, b_incli: str, b_disambig: str, coords=None):
    """If no coords input present force selection using the interactive selector."""
    if not coords:
        coords = select_custom_area(b_field)

    br, br_err, bp, bp_err, bt, bt_err = create_custom_map(b_field, b_azi, b_incli, b_disambig, coords)


def run_series(b_field_arr: list[str], b_azi_arr: list[str], b_incli_arr: list[str], b_disambig_arr: list[str],
               coords=None) -> None:
    # If no coords input present force selection using the interactive selector from the first image.
    if not coords:
        coords = select_custom_area(b_field_arr[0])

    for i, b_field, b_azi, b_incli, b_disambig in zip(range(len(b_field_arr)), b_field_arr, b_azi_arr, b_incli_arr,
                                                      b_disambig_arr):

        # Adjust every frame crop coordinates for solar rotation before running
        hdr = sunpy.map.Map(b_field_arr[i]).meta

        if i == 0:
            new_final_coords = coords
        else:
            bottomleft, topright = coords
            map_newframe = HeliographicCarrington(obstime=hdr['date-obs'])
            with propagate_with_solar_surface():
                new_final_coords = (bottomleft.transform_to(map_newframe), topright.transform_to(map_newframe))

        run(b_field, b_azi, b_incli, b_disambig, new_final_coords)


def main(argv: typing.Sequence[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Create custom SHARP maps from full disk vector magnetic field FITS data."
    )

    # Required FITS input files
    parser.add_argument("-f", "--b_field", required=True, type=str,
                        help="Path to the magnetic field strength FITS file")
    parser.add_argument("-a", "--b_azi", required=True, type=str,
                        help="Path to the magnetic field azimuth FITS file")
    parser.add_argument("-i", "--b_incli", required=True, type=str,
                        help="Path to the magnetic field inclination FITS file")
    parser.add_argument("-d", "--b_disambig", required=True, type=str,
                        help="Path to the disambiguation FITS file (optional)")
    # Optional inputs
    parser.add_argument("-c", "--coords", nargs=4, type=float, default=None,
                        metavar=("XMIN", "XMAX", "YMIN", "YMAX"),
                        help="Crop coordinates (xmin xmax ymin ymax). If omitted, interactive selection is launched.")

    args = parser.parse_args(argv)

    run(
        b_field=args.b_field,
        b_azi=args.b_azi,
        b_incli=args.b_incli,
        b_disambig=args.b_disambig,
        coords=args.coords,
    )


if __name__ == "__main__":
    main()
