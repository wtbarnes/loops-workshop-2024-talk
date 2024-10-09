"""
Code to process L1 AIA files to L2
"""
import pathlib

import aiapy.calibrate
import astropy.units as u
import astropy.table
import astropy.wcs
import numpy as np
import sunpy.map
import eispac.core  # Register EISMap source
import xarray

from astropy.coordinates import SkyCoord
from sunpy.coordinates import propagate_with_solar_surface, get_horizons_coord, Helioprojective
from sunpy.map.header_helper import _set_instrument_meta
from tqdm import tqdm

import sunpy
sunpy.log.setLevel('WARN')  # Makes processing less noisy

TARGET_SCALE = 0.6 * u.arcsec/u.pix
CORRECTION_TABLE = aiapy.calibrate.util.get_correction_table()
ROOT_DIR = pathlib.Path('../data')
AR_TABLE_PATH = ROOT_DIR / 'tables' / 'mason_ars_region_1.asdf'
AIA_CHANNELS = ['94', '131', '171', '193', '211', '335']
CADENCE = 30*u.s  # This will change depending on your dataset


def aia_l1_to_l2(level_1_filenames, output_dir, shape, ref_coord):
    """
    Align all L1 images to a common FOV at specified time with corrections for exposure time
    and degradation.

    Parameters
    ----------
    top_dir: path-like
    shape: `tuple`
    ref_coord: `~astropy.coordinates.SkyCoord`
    """
    output_dir = pathlib.Path(output_dir)
    for file in tqdm(level_1_filenames):
        m_l1 = sunpy.map.Map(file)
        target_header = sunpy.map.make_fitswcs_header(
            shape,
            ref_coord,
            scale=u.Quantity([TARGET_SCALE,TARGET_SCALE]),
            rotation_angle=0*u.deg,
            instrument=m_l1.meta['instrume'],
            telescope=m_l1.meta['telescop'],
            wavelength=m_l1.wavelength,
            exposure=m_l1.exposure_time,
            #unit=m_l1.unit,  # FIXME: revert back to this once its merged in 6.0
        )
        # Add extra keys
        target_header['bunit'] = m_l1.unit.to_string()  # FIXME: remove this once it can be passed above
        target_header['lvl_num'] = 2
        target_header['quality'] = m_l1.meta['quality']
        target_header['t_obs'] = m_l1.date.isot
        target_header['rsun_ref'] = m_l1.meta['rsun_ref']  # needed for reprojection
        # Remove any negative values
        m_l1 = sunpy.map.Map(np.where(m_l1.data<0,0,m_l1.data), m_l1.meta)
        # Reproject
        with propagate_with_solar_surface():
            m_l2 = m_l1.reproject_to(target_header, parallel=False)
        # NOTE: This is needed in order to preserve the keys in the above header
        m_l2 = sunpy.map.Map(m_l2.data, target_header)
        # Additional prep steps
        m_l2 = aiapy.calibrate.correct_degradation(m_l2, correction_table=CORRECTION_TABLE)
        m_l2 = m_l2 / m_l2.exposure_time
        # Save to L2 dir
        output_dir.mkdir(exist_ok=True, parents=True)
        file_l2 = output_dir / file.name.replace('lev1', 'lev2')
        m_l2.save(file_l2, overwrite=True)


def aia_l1_to_l2_eis(level_1_filenames, output_dir, m_eis_ref):
    """
    Align all L1 images to EIS FOV and resolution at specified time with corrections for exposure
    time and degradation. 
    """
    for file in tqdm(level_1_filenames):
        m_l1 = sunpy.map.Map(file)
        m_l1 = sunpy.map.Map(np.where(m_l1.data<0,0,m_l1.data), m_l1.meta)
        # Additional prep steps
        m_l1 = aiapy.calibrate.correct_degradation(m_l1, correction_table=CORRECTION_TABLE)
        m_l1 = m_l1 / m_l1.exposure_time
        m_l1.meta['rsun_ref'] = m_eis_ref.rsun_meters.to_value('meter')
        # Reproject
        with propagate_with_solar_surface():
            m_l2 = m_l1.reproject_to(m_eis_ref.wcs, parallel=False)
        target_header = m_l2.meta.copy()
        # Propagate relevant non-WCS AIA metadata
        target_header = _set_instrument_meta(
            target_header,
            m_l1.meta['instrume'],
            m_l1.meta['telescop'],
            None,
            None,
            m_l1.wavelength,
            m_l1.exposure_time,
            None, # m_l1.unit,
        )
        target_header['bunit'] = m_l1.unit.to_string()  # FIXME: remove this once it can be passed above
        target_header['lvl_num'] = 2
        target_header['quality'] = m_l1.meta['quality']
        target_header['t_obs'] = m_l1.date.isot
        m_l2 = sunpy.map.Map(m_l2.data, target_header)
        # Save to L2 dir
        output_dir.mkdir(exist_ok=True)
        m_l2.save(output_dir / file.name.replace('lev1', 'lev2'), overwrite=True)


def build_aligned_datacube(level_2_dir, output_filename, time_common, ref_date):
    """
    Given a set of L2 AIA files, build a Zarr dataset of the interpolated and aligned images for each
    AIA channel.
    """
    ds = {}
    for channel in AIA_CHANNELS:
        file_list = sorted(level_2_dir.glob(f'aia.lev2_euv_12s.*.{channel}.image.fits'))
        # NOTE: This is done as a for loop because reading all maps in at once can result
        # in too many open files, depending on the OS
        times = []
        data_array = []
        for file in file_list:
            _map = sunpy.map.Map(file, memmap=False)
            if _map.meta['quality']==0:
                times.append(_map.meta['t_obs'])
                data_array.append(_map.data)
        times = (astropy.time.Time(times) - ref_date).to('s')
        data_array = np.array(data_array)
        time_coord = xarray.DataArray(times.value,
                                      dims=['time'],
                                      attrs={'unit': times.unit.to_string()})
        # NOTE; WCS and unit will be the same for all maps so just use the last one
        da = xarray.DataArray(data_array,
                              dims=['time', 'pixel_y', 'pixel_x'],
                              coords={'time': time_coord},
                              attrs={'unit': _map.unit.to_string(),
                                     **dict(_map.wcs.to_header())})
        # Fill NaN values
        da = da.ffill('time')
        # Replace negative values with 0s (should have done this in the L1->L2 step)
        da = da.where(da>0, 0.0)
        # Interpolate datacube
        da = da.interp(time=time_common.to_value(times.unit), kwargs={'fill_value': 'extrapolate'})
        # Build dataset
        ds[channel] = da
    ds = xarray.Dataset(ds).chunk({'pixel_x': 'auto', 'pixel_y': 'auto'})
    ds.to_zarr(output_filename)


if __name__ == '__main__':
    ar_table = astropy.table.QTable.read(AR_TABLE_PATH)
    for row in ar_table:
        print(row['NOAA AR'])
        # Get list of L1 files
        top_dir = ROOT_DIR / f'noaa_{row["NOAA AR"]}'
        l1_filenames = sorted((top_dir / 'AIA' / 'level_1').glob('aia.lev1_euv_12s.*.image.fits'))
        # Build reference coordinate and output shape
        observer = get_horizons_coord('SDO', row['Date mid'])
        frame = Helioprojective(observer=observer, obstime=observer.obstime)
        blc = row['bottom left']
        trc = row['top right']
        ref_coord = SkyCoord(Tx=(blc[0] + trc[0])/2, Ty=(blc[1] + trc[1])/2, frame=frame)
        shape = u.Quantity([trc[1] - blc[1], trc[0] - blc[0]]) / TARGET_SCALE
        shape = np.ceil(shape.to_value('pixel')).astype(int).tolist()
        # L1->L2 promotion
        level_2_dir = top_dir / 'AIA' / 'level_2'
        aia_l1_to_l2(l1_filenames, level_2_dir, shape, ref_coord)
        # L1->L2-EIS promotion
        level_2_eis_dir = top_dir / 'AIA' / 'level_2_EIS'
        m_eis_ref = sunpy.map.Map(top_dir / 'EIS' / 'level_2.5' / 'eis_*.fe_12_195_119.2c-0.int.fits')
        aia_l1_to_l2_eis(l1_filenames, level_2_eis_dir, m_eis_ref)
        # Aligned datacubes
        time_common = np.arange(0, (row['Date end']-row['Date start']).to_value('h'), CADENCE.to_value('h')) * u.h
        build_aligned_datacube(level_2_dir, top_dir / 'AIA' / 'level_2_datacubes.zarr', time_common, row['Date start'])
        build_aligned_datacube(level_2_eis_dir, top_dir / 'AIA' / 'level_2_EIS_datacubes.zarr', time_common, row['Date start'])
