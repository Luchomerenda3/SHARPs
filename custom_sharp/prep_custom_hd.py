"""Prepare source-derived metadata for custom CEA maps, using SHARP keyword names.

This module is opt-in: it does not change prep_hd, the creation pipeline, any
input map/header/array, or any file. The first eight arguments of prep_custom_hd
are the same as prep_hd. Pass ``reference_header=sharp_map.meta`` to use the
original SHARP's keyword names, order and comments as the building template.
Its region measurements are never copied into a different custom region.

Example for the notebook (no existing map or file is overwritten)::

    from custom_sharp.prep_custom_hd import prep_custom_hd

    header, report = prep_custom_hd(
        field_map.meta, custom_br_map.data,
        custom_br_map.meta['crval1'], custom_br_map.meta['crval2'],
        custom_br_map.data.shape[1], custom_br_map.data.shape[0],
        custom_br_map.meta['cdelt1'], custom_br_map.meta['cdelt2'],
        reference_header=sharp_map.meta, return_report=True,
    )
    enriched_br_map = sunpy.map.Map(custom_br_map.data, header)
    print(report.unavailable)

To calculate the space-weather fields, also supply aligned CEA arrays ``br``,
``bt``, ``bp``, ``br_err``, ``bt_err``, ``bp_err``, ``bitmap`` and
``conf_disambig``. For segment='Br', br defaults to data_out. Optional ``los``
and ``los_err`` enable LOS indices. Errors are standard deviations; all field
and error arrays must be in G (= Mx/cm^2). Bt must be southward: the conversion
to calculate_sharpkeys is Bx=Bp, By=-Bt, Bz=Br, as in its get_data function.
Arrays must describe the same grid/time; bare arrays cannot verify registration.

Observation, calibration, inversion and disambiguation metadata come from
hd_in (the actual source field file/header/map). Geometry and data statistics
are rebuilt; region indices are recomputed using calculate_sharpkeys. Unknown
science/template values are FITS undefined (None), with reasons in the report.
Record identifiers and obsolete storage, scaling/checksum and WCS cards are
excluded. No arbitrary HARPNUM, NOAA association, bitmap or noise is invented.
Explicit, independently verified tracking metadata may be supplied through
region_metadata. Existing INV*/AMB* values describe upstream processing.

DATARMS is sample standard deviation (N-1), as in the reference SHARP, not RMS
about zero. The local field-gradient functions return G/pixel; this wrapper
converts their outputs to G/Mm. Other indices retain calculate_sharpkeys'
conventions and limitations, and are not guaranteed identical to JSOC values.
Potential/energy/shear calculations require compute_potential=True because
greenpot is expensive. ERRMSHA is left undefined: the local computeShearAngle
repeats one uncertainty term and overwrites its accumulator inside the loop.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from os import PathLike

import numpy as np
from astropy.io import fits
from astropy.time import Time

from .prep_hd import prep_hd

__all__ = ["HeaderReport", "prep_custom_hd", "remap_source_segment"]


@dataclass
class HeaderReport:
    """Per-key provenance; unavailable fields have None values in the header."""

    inherited: list[str] = field(default_factory=list)
    computed: list[str] = field(default_factory=list)
    supplied: list[str] = field(default_factory=list)
    unavailable: dict[str, str] = field(default_factory=dict)
    excluded: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


_GROUPS = {
    "flux": ("USFLUX", "ERRVF", "CMASK"),
    "inclination": ("MEANGAM", "ERRGAM"),
    "gradient_br": ("MEANGBZ", "ERRBZ"),
    "gradient_bh": ("MEANGBH", "ERRBH"),
    "gradient_total": ("MEANGBT", "ERRBT"),
    "current": ("MEANJZD", "ERRJZ", "TOTUSJZ", "ERRUSI"),
    "twist": ("MEANALP", "ERRALP"),
    "helicity": ("MEANJZH", "ERRMIH", "TOTUSJH", "ERRTUI", "ABSNJZH", "ERRTAI"),
    "polarity_current": ("SAVNCPP", "ERRJHT"),
    "energy": ("MEANPOT", "ERRMPOT", "TOTPOT", "ERRTPOT"),
    "shear": ("MEANSHR", "ERRMSHA", "SHRGT45"),
    "r_value": ("R_VALUE",),
    "los_gradient": ("MEANGBL",),
    "los_flux": ("USFLUXL", "CMASKL"),
}
_SCIENCE = {key for keys in _GROUPS.values() for key in keys}
_STATS = {
    "DATAVALS",
    "MISSVALS",
    "DATAMIN",
    "DATAMAX",
    "DATAMEDN",
    "DATAMEAN",
    "DATARMS",
    "DATASKEW",
    "DATAKURT",
}
_REGION = {
    "HARPNUM",
    "MASK",
    "ARM_QUAL",
    "ARM_NCLN",
    "H_MERGE",
    "H_FAINT",
    "ARM_MODL",
    "ARM_EDGE",
    "ARM_BETA",
    "LATDTMIN",
    "LONDTMIN",
    "LATDTMAX",
    "LONDTMAX",
    "OMEGA_DT",
    "NPIX",
    "SIZE",
    "AREA",
    "NACR",
    "SIZE_ACR",
    "AREA_ACR",
    "MTOT",
    "MNET",
    "MPOS_TOT",
    "MNEG_TOT",
    "MMEAN",
    "MSTDEV",
    "MSKEW",
    "MKURT",
    "LAT_MIN",
    "LON_MIN",
    "LAT_MAX",
    "LON_MAX",
    "LAT_FWT",
    "LON_FWT",
    "LATFWTPO",
    "LONFWTPO",
    "LATFWTNE",
    "LONFWTNE",
    "T_FRST",
    "T_FRST1",
    "T_LAST",
    "T_LAST1",
    "N_PATCH",
    "N_PATCH1",
    "N_PATCHM",
    "NOAA_AR",
    "NOAA_NUM",
    "NOAA_ARS",
    "GWILL",
    "AMBPATCH",
    "AMBWEAK",
}
_STORAGE = {
    "SIMPLE",
    "XTENSION",
    "BITPIX",
    "PCOUNT",
    "GCOUNT",
    "EXTEND",
    "BSCALE",
    "BZERO",
    "BLANK",
    "CHECKSUM",
    "DATASUM",
    "HEADSUM",
    "END",
    "THEAP",
    "TFIELDS",
    "EXTNAME",
    "EXTVER",
}
_IDENTIFIERS = {"RECNUM", "DRMS_ID", "PRIMARYK"}
_PRODUCT = {"DATE", "CONTENT", "CODEVER7"}
_WCS = re.compile(
    r"^(?:WCSAXES|WCSNAME|LONPOLE|LATPOLE|RADESYS|EQUINOX|"
    r"(?:CTYPE|CUNIT|CRPIX|CRVAL|CDELT|CROTA|CRDER|CSYSER)\d+|"
    r"(?:PC|CD|PV|PS)\d+_\d+)[A-Z]?$|^(?:A|B|AP|BP)_(?:ORDER|\d+_\d+)$"
)


def _as_header(value):
    """Copy a Header, metadata mapping, map, or the first 2-D FITS HDU header."""
    if isinstance(value, (str, PathLike)):
        with fits.open(value, memmap=True) as hdus:
            for hdu in hdus:
                if hdu.header.get("NAXIS") == 2:
                    return hdu.header.copy()
        raise ValueError(f"No 2-D image found in {value}")
    if isinstance(value, fits.Header):
        return value.copy()
    if hasattr(value, "meta"):
        value = value.meta
    if not isinstance(value, Mapping):
        raise TypeError("Expected a FITS Header, metadata mapping, map or FITS path")
    entries = {str(k).upper(): v for k, v in value.items()}
    comments = {str(k).upper(): v for k, v in entries.pop("KEYCOMMENTS", {}).items()}
    header = fits.Header()
    for key, item in entries.items():
        if key in {"HISTORY", "COMMENT"}:
            lines = item if isinstance(item, (list, tuple)) else str(item).splitlines()
            for line in lines:
                header.append((key, str(line)))
        else:
            if isinstance(item, np.generic):
                item = item.item()
            if isinstance(item, float) and not np.isfinite(item):
                item = None
            header[key] = (item, comments.get(key, ""))
    return header


def _excluded(key):
    if key in _STORAGE or re.match(r"^(?:NAXIS\d*|Z[A-Z0-9_]+|BKEY[SID]\d+)$", key):
        return "Source storage/scaling/compression/checksum is not valid for the output"
    if key in _IDENTIFIERS:
        return "Source record identifier is not a custom product identifier"
    if _WCS.match(key) or key.startswith("IMCR"):
        return "Coordinates must be rebuilt for the new grid"
    if key == "_BUNIT" or (key.startswith("_") and key[1:] in _STATS):
        return "Replaced by standard output BUNIT/DAT* keyword names"
    return None


def _array(value, name, shape, *, error=False, codes=False):
    if value is None:
        return None
    a = np.ma.asarray(value)
    if a.shape != shape:
        raise ValueError(f"{name} must be aligned to shape {shape}; got {a.shape}")
    if a.dtype.kind not in "iuf":
        raise TypeError(
            f"{name} must be numeric; masks require SHARP codes, not booleans"
        )
    a = np.array(np.ma.asarray(a, dtype=float).filled(np.nan), copy=True)
    finite = np.isfinite(a)
    if error and np.any(a[finite] < 0):
        raise ValueError(f"{name} contains negative standard deviations")
    if codes and np.any(a[finite] != np.floor(a[finite])):
        raise ValueError(
            f"{name} has fractional codes; use nearest-neighbor resampling"
        )
    a[~finite] = 0 if codes else np.nan
    return a


def _put(header, report, key, value, reason, comment=""):
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        header[key] = (None, comment or "Undefined; see returned HeaderReport")
        report.unavailable[key] = reason
        if key in report.computed:
            report.computed.remove(key)
    else:
        header[key] = (value, comment or reason)
        report.unavailable.pop(key, None)
        if key not in report.computed:
            report.computed.append(key)


def _geometry(phi_c, lambda_c, nx, ny, dx, dy):
    if (
        not np.all(np.isfinite([phi_c, lambda_c, nx, ny, dx, dy]))
        or isinstance(nx, (bool, np.bool_))
        or isinstance(ny, (bool, np.bool_))
        or int(nx) != nx
        or int(ny) != ny
        or nx < 1
        or ny < 1
        or dx <= 0
        or dy <= 0
        or abs(lambda_c) > 90
    ):
        raise ValueError(
            "Use finite coordinates, latitude in [-90,90], positive integer dimensions and positive pixel scales"
        )
    return int(nx), int(ny)


def prep_custom_hd(
    hd_in,
    data_out,
    phi_c,
    lambda_c,
    nx,
    ny,
    dx,
    dy,
    *,
    reference_header=None,
    segment="Br",
    br=None,
    bt=None,
    bp=None,
    br_err=None,
    bt_err=None,
    bp_err=None,
    bitmap=None,
    conf_disambig=None,
    los=None,
    los_err=None,
    compute_potential=False,
    region_metadata=None,
    return_report=False,
):
    """Return a new Header, or (Header, HeaderReport) when return_report=True.

    hd_in describes the source observation (e.g. field_map.meta).
    reference_header describes the desired schema (e.g. sharp_map.meta), not
    the values of a different region. FITS keyword names are case-insensitive:
    this function writes uppercase; SunPy exposes them as lowercase metadata.

    data_out is the selected 2-D float32/float64 output segment. All optional
    arrays must have its shape and the same CEA registration, epoch and units.
    segment is Br, Bt, Bp or one of their *_err maps. Masked/nonfinite pixels
    are omitted from statistics; callers writing masked data should encode
    missing pixels consistently. Input arrays and headers are never changed.

    region_metadata accepts only explicitly verified region/tracking keys,
    such as NOAA_ARS. No default region, noise, confidence or mask is assumed.
    The module docstring contains a notebook example and algorithm caveats.
    """
    nx, ny = _geometry(phi_c, lambda_c, nx, ny, dx, dy)
    raw = np.ma.asarray(data_out)
    if raw.ndim != 2 or raw.dtype.kind != "f" or raw.dtype.itemsize not in (4, 8):
        raise ValueError("data_out must be a 2-D float32 or float64 array")
    if segment not in {"Br", "Bt", "Bp", "Br_err", "Bt_err", "Bp_err"}:
        raise ValueError("segment must be Br, Bt, Bp, Br_err, Bt_err or Bp_err")
    a = _array(raw, "data_out", (ny, nx), error=segment.endswith("_err"))
    source = _as_header(hd_in)
    reference = (
        _as_header(reference_header) if reference_header is not None else fits.Header()
    )
    supplied = (
        _as_header(region_metadata) if region_metadata is not None else fits.Header()
    )
    if set(supplied) - _REGION:
        raise ValueError(
            f"region_metadata cannot override {sorted(set(supplied) - _REGION)}"
        )
    report = HeaderReport()
    header = fits.Header()

    # Start with the reference's schema and comments, with no copied measurements.
    for card in reference.cards:
        key = card.keyword
        reason = _excluded(key)
        if reason:
            report.excluded[key] = reason
        elif key not in {"", "COMMENT", "HISTORY"}:
            _put(
                header,
                report,
                key,
                None,
                "Not available from the source observation or supplied inputs",
                card.comment,
            )

    # Actual values come from the file used for creating the custom map.
    for card in source.cards:
        key = card.keyword
        reason = _excluded(key)
        if reason:
            report.excluded[key] = reason
        elif key in _REGION | _SCIENCE | _STATS | _PRODUCT | {"BUNIT"}:
            _put(
                header,
                report,
                key,
                None,
                "Must be supplied or recomputed for the custom region",
                card.comment,
            )
        elif key in {"", "COMMENT", "HISTORY"}:
            header.append(card)
        else:
            header[key] = (
                card.value,
                card.comment or (reference.comments[key] if key in reference else ""),
            )
            if card.value is not None:
                report.inherited.append(key)
                report.unavailable.pop(key, None)
            else:
                report.unavailable[key] = "Undefined in the source observation"

    # Reuse prep_hd's projection convention; replace source WCS and FITS structure.
    base = prep_hd(source, a, phi_c, lambda_c, nx, ny, dx, dy)
    for key in base:
        if _WCS.match(key) or key in {
            "SIMPLE",
            "BITPIX",
            "NAXIS",
            "NAXIS1",
            "NAXIS2",
            "BUNIT",
        }:
            value = -8 * raw.dtype.itemsize if key == "BITPIX" else base[key]
            _put(header, report, key, value, "Custom CEA output")
            report.excluded.pop(key, None)
    if str(source.get("CTYPE1", "")).startswith("HPLN"):
        for key in ("CRPIX1", "CRPIX2", "CRVAL1", "CRVAL2"):
            if key in source:
                _put(
                    header,
                    report,
                    "IM" + key,
                    source[key],
                    "Reference coordinate of the source image",
                )
                report.excluded.pop("IM" + key, None)
    _put(
        header,
        report,
        "DATE",
        Time.now().utc.isot,
        "UTC creation time of this custom header",
    )
    _put(
        header,
        report,
        "CONTENT",
        f"Custom CEA {segment}",
        "Derived from source observation",
    )

    finite = a[np.isfinite(a)]
    stats = dict.fromkeys(sorted(_STATS))
    stats.update(DATAVALS=int(finite.size), MISSVALS=int(a.size - finite.size))
    if finite.size:
        stats.update(
            DATAMIN=finite.min(),
            DATAMAX=finite.max(),
            DATAMEDN=np.median(finite),
            DATAMEAN=finite.mean(),
            DATARMS=finite.std(ddof=1) if finite.size > 1 else None,
        )
        if finite.std() > 0:
            z = (finite - finite.mean()) / finite.std()
            stats.update(DATASKEW=np.mean(z**3), DATAKURT=np.mean(z**4) - 3)
    for key, value in stats.items():
        _put(
            header,
            report,
            key,
            value,
            "Finite, unmasked output pixel statistic"
            if value is not None
            else "Insufficient finite pixels or zero variance",
        )
    header.comments["DATARMS"] = "Sample standard deviation (N-1)"
    header.comments["DATASKEW"] = "Population standardized third moment"
    header.comments["DATAKURT"] = "Population excess kurtosis (normal = 0)"
    for key in sorted(_SCIENCE):
        comment = reference.comments[key] if key in reference else ""
        _put(
            header,
            report,
            key,
            None,
            "Required aligned data/masks not supplied",
            comment,
        )
    _compute_indices(
        header,
        report,
        nx,
        ny,
        dx,
        dy,
        compute_potential,
        br=a if br is None and segment == "Br" else br,
        bt=bt,
        bp=bp,
        br_err=br_err,
        bt_err=bt_err,
        bp_err=bp_err,
        bitmap=bitmap,
        conf_disambig=conf_disambig,
        los=los,
        los_err=los_err,
    )

    for key in set(header) & _REGION:
        report.unavailable[key] = (
            "Requires verified region/tracking metadata; not copied from the original region"
        )
    for card in supplied.cards:
        header[card.keyword] = (card.value, card.comment)
        if card.value is not None:
            report.supplied.append(card.keyword)
            report.unavailable.pop(card.keyword, None)
    if "CODEVER7" in header:
        report.unavailable["CODEVER7"] = (
            "Original SHARP processing version does not describe this custom product"
        )
    header.add_history(
        "Prepared by custom_sharp.prep_custom_hd; source metadata retained."
    )
    header.add_history(
        "Reference SHARP used for keyword schema only; region values not copied."
    )
    header.add_history(
        "INV*/AMB*/CODEVER* describe source processing; DAT* describe this output."
    )
    for key in ("RECNUM", "DRMS_ID", "HARPNUM", "DATE"):
        if source.get(key) is not None:
            header.add_history(f"Source {key}: {source[key]}")
    for note in report.notes:
        header.add_history(note)
    if report.unavailable:
        header.add_history(
            "Undefined fields (see HeaderReport): "
            + ", ".join(sorted(report.unavailable))
        )
    # FITS primary-image structural cards must lead, regardless of template order.
    structural = ("SIMPLE", "BITPIX", "NAXIS", "NAXIS1", "NAXIS2")
    ordered = fits.Header([header.cards[key] for key in structural])
    ordered.extend(card for card in header.cards if card.keyword not in structural)
    header = ordered
    return (header, report) if return_report else header


def _compute_indices(header, report, nx, ny, dx, dy, compute_potential, **inputs):
    """Call calculate_sharpkeys directly when a group's dependencies exist."""
    arrays = {
        name: _array(
            value,
            name,
            (ny, nx),
            error=name.endswith("_err"),
            codes=name in {"bitmap", "conf_disambig"},
        )
        for name, value in inputs.items()
    }

    def unavailable(group, reason):
        for key in _GROUPS[group]:
            _put(header, report, key, None, reason)

    ephemeris = ("RSUN_REF", "RSUN_OBS", "DSUN_OBS")
    reason = None
    if not np.isclose(dx, dy, rtol=1e-10, atol=0):
        reason = "calculate_sharpkeys requires equal positive CEA pixel scales"
    elif any(header.get(key) is None for key in ephemeris):
        reason = "Requires source RSUN_REF, RSUN_OBS and DSUN_OBS"
    if reason:
        for group in _GROUPS:
            unavailable(group, reason)
        return
    rsun, angular_rsun, dsun = (float(header[key]) for key in ephemeris)
    if (
        not np.all(np.isfinite([rsun, angular_rsun, dsun]))
        or min(rsun, angular_rsun, dsun) <= 0
    ):
        raise ValueError("RSUN_REF, RSUN_OBS and DSUN_OBS must be positive and finite")
    pixel_arcsec = np.rad2deg(np.arctan(rsun * np.deg2rad(dx) / dsun)) * 3600
    gradient_scale = angular_rsun / rsun / pixel_arcsec * 1e6

    # Local import keeps basic header preparation independent of this module's CLI.
    import calculate_sharpkeys as sk

    def ready(group, names):
        missing = [name for name in names if arrays[name] is None]
        if missing:
            unavailable(group, "Requires aligned " + ", ".join(missing))
        return not missing

    def run(group, function, args, *, scale=1, positions=None):
        try:
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                values = function(*args)
        except (ZeroDivisionError, FloatingPointError, ValueError) as exc:
            unavailable(group, f"{function.__name__} undefined: {exc}")
            return
        if positions is not None:
            values = [values[i] for i in positions]
        for key, value in zip(_GROUPS[group], values, strict=True):
            value *= scale
            _put(
                header,
                report,
                key,
                value,
                f"Custom: {function.__name__}"
                if np.isfinite(value)
                else f"{function.__name__} returned a nonfinite result; insufficient/invalid data",
            )

    bz, bt, bx = arrays["br"], arrays["bt"], arrays["bp"]
    by = -bt if bt is not None else None
    ez, ey, ex = arrays["br_err"], arrays["bt_err"], arrays["bp_err"]
    conf, mask = arrays["conf_disambig"], arrays["bitmap"]
    geometry = (nx, ny, rsun, angular_rsun, pixel_arcsec)
    radial = ("br", "br_err", "conf_disambig", "bitmap")
    vector = (*radial, "bt", "bp", "bt_err", "bp_err")
    if ready("flux", radial):
        run("flux", sk.compute_abs_flux, (bz, ez, conf, mask, *geometry))
    if ready("gradient_br", radial):
        if min(nx, ny) >= 3:
            run(
                "gradient_br",
                sk.computeBzderivative,
                (bz, ez, nx, ny, conf, mask),
                scale=gradient_scale,
            )
        else:
            unavailable("gradient_br", "Gradient stencil requires at least 3x3 pixels")
    groups = (
        "inclination",
        "gradient_bh",
        "gradient_total",
        "current",
        "twist",
        "helicity",
        "polarity_current",
        "energy",
        "shear",
    )
    available = [ready(group, vector) for group in groups]
    if all(available):
        with np.errstate(divide="ignore", invalid="ignore"):
            bh, eh = sk.compute_bh(bx, by, bz, ex, ey, ez, conf, mask, nx, ny)
            total, et = sk.compute_bt(bx, by, bz, ex, ey, ez, conf, mask, nx, ny)
        run(
            "inclination",
            sk.compute_gamma,
            (bx, by, bz, bh, ez, eh, conf, mask, *geometry),
        )
        if min(nx, ny) >= 3:
            run(
                "gradient_bh",
                sk.computeBhderivative,
                (bh, eh, nx, ny, conf, mask),
                scale=gradient_scale,
            )
            run(
                "gradient_total",
                sk.computeBtderivative,
                (total, et, nx, ny, conf, mask),
                scale=gradient_scale,
            )
            jz, je, derx, dery = sk.computeJz(bx, by, ex, ey, conf, mask, nx, ny)
            run(
                "current",
                sk.computeJzmoments,
                (jz, je, derx, dery, conf, mask, *geometry, sk.munaught),
            )
            run("twist", sk.computeAlpha, (jz, je, bz, ez, conf, mask, *geometry))
            run("helicity", sk.computeHelicity, (jz, je, bz, ez, conf, mask, *geometry))
            run(
                "polarity_current",
                sk.computeSumAbsPerPolarity,
                (jz, je, bz, ez, conf, mask, *geometry, sk.munaught),
            )
        else:
            for group in groups[1:7]:
                unavailable(group, "Derivative stencil requires at least 3x3 pixels")
        if compute_potential and min(nx, ny) >= 3:
            px, py = sk.greenpot(bz, nx, ny)
            run(
                "energy",
                sk.computeFreeEnergy,
                (ex, ey, bx, by, px, py, *geometry, conf, mask),
            )
            run(
                "shear",
                sk.computeShearAngle,
                (ex, ey, ez, bx, by, bz, px, py, nx, ny, conf, mask),
            )
            _put(
                header,
                report,
                "ERRMSHA",
                None,
                "Local computeShearAngle uncertainty accumulation is unreliable",
            )
        else:
            for group in ("energy", "shear"):
                unavailable(
                    group, "Requires compute_potential=True and at least 3x3 pixels"
                )
    if conf is not None and mask is not None:
        report.notes.append("Vector selection: conf_disambig >= 70 and bitmap >= 30.")
    report.notes.append(
        "Bx=Bp, By=-Bt, Bz=Br; computed field gradients converted to G/Mm."
    )
    los, le = arrays["los"], arrays["los_err"]
    if ready("los_flux", ("los", "los_err", "bitmap")):
        run(
            "los_flux",
            sk.compute_abs_flux_los,
            (los, le, mask, *geometry),
            positions=(0, 2),
        )
    if ready("los_gradient", ("los", "los_err", "bitmap")):
        if min(nx, ny) >= 3:
            run(
                "los_gradient",
                sk.computeLOSderivative,
                (los, le, nx, ny, mask, rsun, angular_rsun, pixel_arcsec),
                scale=gradient_scale,
                positions=(0,),
            )
        else:
            unavailable("los_gradient", "Gradient stencil requires at least 3x3 pixels")
    if ready("r_value", ("los", "los_err")):
        scale = round(2 / pixel_arcsec)
        reduced = np.ceil(np.array([ny, nx]) / max(scale, 1)).astype(int)
        if scale < 1 or any(round(n / 2) + 3 > n for n in reduced):
            unavailable(
                "r_value", "Grid too small/coarse for computeR's reduced boxcar kernel"
            )
        else:
            run("r_value", sk.computeR, (los, le, nx, ny, pixel_arcsec), positions=(0,))


def remap_source_segment(
    hd_in, data, phi_c, lambda_c, nx, ny, dx=0.03, dy=0.03, *, categorical=False
):
    """Resample a scalar source image with the existing bvec2cea CEA mapping.

    categorical=True preserves bitmap/conf_disambig codes with nearest-neighbor
    interpolation and missing=0. Continuous scalars use bilinear interpolation
    and missing=NaN. This does not rotate vectors or propagate uncertainties.
    hd_in must describe this data's helioprojective source grid/observation.
    """
    from scipy.ndimage import map_coordinates

    from .find_cea_coord import find_cea_coord

    nx, ny = _geometry(phi_c, lambda_c, nx, ny, dx, dy)
    source = _as_header(hd_in)
    if not str(source.get("CTYPE1", "")).startswith("HPLN") or not str(
        source.get("CTYPE2", "")
    ).startswith("HPLT"):
        raise ValueError("Expected the helioprojective source image grid")
    shape = (int(source["NAXIS2"]), int(source["NAXIS1"]))
    a = _array(data, "source segment", shape, codes=categorical)
    if a is None:
        raise ValueError("Source data are required")
    xi, eta, _, _ = find_cea_coord(source, phi_c, lambda_c, nx, ny, dx, dy)
    finite = np.isfinite(xi) & np.isfinite(eta)
    coords = [np.where(finite, eta, -1), np.where(finite, xi, -1)]
    fill = 0.0 if categorical else np.nan
    output = map_coordinates(
        a, coords, order=0 if categorical else 1, mode="constant", cval=fill
    )
    output[~finite] = fill
    return output
