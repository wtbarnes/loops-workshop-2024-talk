import pathlib

import astropy.units as u
from sunpy.net import Fido, attrs as a
import sunpy.map
from aiapy.calibrate import register, update_pointing
from sunkit_image.coalignment import _calculate_shift as calculate_shift
from eispac.core import EISMap

__all__ = ['cross_correlate_with_aia', 'fix_eis_pointing']


def cross_correlate_with_aia(m_eis, m_aia):
    """
    Calculate cross-correlation between EIS and AIA data and find the
    bottom left coordinate of the EIS map in the coordinate system of
    the AIA image.
    
    Parameters
    ----------
    m_eis : `~sunpy.map.Map`
    m_aia : `~sunpy.map.Map`
    return_aia_map : `bool`
        If True, return the AIA map resampled at the EIS resolution. This is
        useful for calculating the cross-correlation between the two maps
        later on (if needed).
    
    Returns
    -------
    reference_coord : `~astropy.coordinate.SkyCoord`
        World coordinate corresponding to the center of the lower left corner
        pixel in the EIS image. The corresponding FITS WCS pixel coordinate
        is (1,1).
    m_aia_r : `~sunpy.map.Map`
        The resampled AIA map. This is used to compute the cross-correlation
        between the the EIS and AIA maps.
    """
    n_x = (m_aia.scale.axis1 * m_aia.dimensions.x) / m_eis.scale.axis1
    n_y = (m_aia.scale.axis2 * m_aia.dimensions.y) / m_eis.scale.axis2
    m_aia_r = m_aia.resample(u.Quantity([n_x, n_y]))
    # Cross-correlate 
    # The resulting "shifts" can be interpreted as the location of the bottom left pixel of the
    # EIS raster in the pixel coordinates of the AIA image. Note that these are in array index
    # coordinates.
    yshift, xshift = calculate_shift(m_aia_r.data, m_eis.data)
    # Find the corresponding coordinate at this pixel position in the resampled AIA map
    return m_aia_r.pixel_to_world(xshift, yshift)


def fix_eis_pointing(l2_dir, corrected_l2_dir):
    """
    Given a directory of EIS files, find the 195.119 observation, download a corresponding AIA 193
    image, correct the pointing, and save out the corrected files.

    Parameters
    ----------
    """
    eis_files = sorted(pathlib.Path(l2_dir).glob('*.fits'))
    eis_maps = sunpy.map.Map(eis_files)

    # Select reference Fe XII 195.119 map
    m_eis_ref = [m for m in eis_maps if m.wavelength == 195.119*u.AA and m.measurement=='intensity'][0]
    # Download AIA 193 map
    q = Fido.search(
            a.Instrument.aia,
            a.Wavelength(193*u.angstrom),
            a.Physobs.intensity,
            a.Time(m_eis_ref.date_start, end=m_eis_ref.date_end, near=m_eis_ref.date_average),
    )
    aia_file = Fido.fetch(q)
    m_aia_ref = sunpy.map.Map(aia_file)
    m_aia_ref = register(update_pointing(m_aia_ref))
    # Cross-correlate 193 and 195.119 observations
    ref_coord = cross_correlate_with_aia(m_eis_ref, m_aia_ref)
    # Compute the reference coord offset from the 195.119 line
    shift_x = ref_coord.Tx - m_eis_ref.bottom_left_coord.Tx
    shift_y = ref_coord.Ty - m_eis_ref.bottom_left_coord.Ty
    eis_maps_fixed = [m.shift_reference_coord(shift_x, shift_y) for m in eis_maps]
    # Save out EIS maps
    fixed_dir = pathlib.Path(corrected_l2_dir)
    fixed_dir.mkdir(exist_ok=True)
    for m, f in zip(eis_maps_fixed, eis_files):
        m.save(fixed_dir / f.name)


def fix_eis_calibration(l2_dir, corrected_l2_dir):
    """
    Correct EIS calibration using the updated 2023 calibration
    """
    import sys
    sys.path.append('../../../../projects/EISPAC-Tutorial___Calibrations/')
    from eis_calibration.eis_calib_2023 import calib_2023
    
    eis_files = sorted(pathlib.Path(l2_dir).glob('*.int.fits'))
    eis_maps = sunpy.map.Map(eis_files)
    eis_maps_fixed = [calib_2023(m) for m in eis_maps]

    fixed_dir = pathlib.Path(corrected_l2_dir)
    fixed_dir.mkdir(exist_ok=True)
    for m, f in zip(eis_maps_fixed, eis_files):
        m.save(fixed_dir / f.name)
