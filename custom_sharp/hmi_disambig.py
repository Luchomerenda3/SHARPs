import numpy as np


def hmi_disambig(azimuth, disambig, method: int = 2, inplace: bool = False):
    """
    Combine HMI disambiguation result with azimuth.
    For details, see Section 5 of Hoeksema et al. (2014, SoPh 289, 3483).
    Copied/translated from SSWIDL/SDO/HMI/IDL code by Xudong Sun (xudongs@hawaii.edu).

    Arguments:
        azimuth: Azimuth image with values between 0 and 180 (degrees).
        disambig: Bit mask with the same size as azimuth.
        method: Integer from 0 to 2 indicating the bit used:
                0 for potential acute, 1 for random, 2 for radial acute (default).
                Out-of-range values default to 2.
        inplace: If True, modifies azimuth in place. If False, returns a modified copy.

    Returns:
        azimuth: Modified azimuth image with values between 0 and 360 (degrees).
    """
    if azimuth.shape != disambig.shape:
        raise ValueError(f"Dimension mismatch: azimuth shape {azimuth.shape} != disambig shape {disambig.shape}")

    # Check method
    if method is None or method < 0 or method > 2:
        method = 2

    # Work on a copy if not inplace
    out_azi = azimuth if inplace else np.array(azimuth, copy=True, dtype=np.float64)

    # Cleanly cast disambig bits to integers, handling any NaNs safely
    disamb = np.nan_to_num(disambig, nan=0).astype(np.int32)

    # Perform disambiguation: move target bit to lowest position and check parity
    disamb = disamb // (2 ** method)
    idx = np.where(disamb % 2 != 0)

    if len(idx[0]) > 0:
        out_azi[idx] += 180.0

    return out_azi
