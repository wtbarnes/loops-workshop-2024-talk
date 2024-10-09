import warnings

import astropy.units as u
import astropy.wcs
import ndcube
import numpy as np
import sunpy.map
import xarray

from sunkit_dem import GenericModel
from demregpy.dn2dem import dn2dem
from demcmc import EmissionLine, TempBins, ContFuncDiscrete, predict_dem_emcee
from synthesizAR.instruments.util import extend_celestial_wcs


__all__ = ['HK12Model',
           'make_slope_map',
           'read_cube_with_xarray',
           'write_cube_with_xarray']


class HK12Model(GenericModel):
    
    def _model(self, alpha=1.0, increase_alpha=1.5, max_iterations=10, guess=None, use_em_loci=False, **kwargs):
        errors = np.array([self.data[k].uncertainty.array.squeeze() for k in self._keys]).T
        dem, edem, elogt, chisq, dn_reg = dn2dem(
            self.data_matrix.T,
            errors,
            self.kernel_matrix.T,
            np.log10(self.kernel_temperatures.to_value('K')),
            self.temperature_bin_edges.to_value('K'),
            max_iter=max_iterations,
            reg_tweak=alpha,
            rgt_fact=increase_alpha,
            dem_norm0=guess,
            gloci=use_em_loci,
            **kwargs,
        )
        _key = self._keys[0]
        dem_unit = self.data[_key].unit / self.kernel[_key].unit / self.temperature_bin_edges.unit
        uncertainty = edem.T * dem_unit
        em = (dem * np.diff(self.temperature_bin_edges)).T * dem_unit
        dem = dem.T * dem_unit
        T_error_upper = self.temperature_bin_centers * (10**elogt - 1 )
        T_error_lower = self.temperature_bin_centers * (1 - 1 / 10**elogt)
        return {'dem': dem,
                'uncertainty': uncertainty,
                'em': em,
                'temperature_errors_upper': T_error_upper.T,
                'temperature_errors_lower': T_error_lower.T,
                'chi_squared': np.atleast_1d(chisq).T}

    @classmethod
    def defines_model_for(self, *args, **kwargs):
        return kwargs.get('model') == 'hk12'
    

class MCMCModel(GenericModel):

    def _model(self, nwalkers=100, nsteps=100, **kwargs):
        emission_lines = []
        for k in self._keys:
            # NOTE: There is a bug in demmcmc that requires all respones to have units
            # of 'cm5 K-1' 
            cont_func=ContFuncDiscrete(self.kernel_temperatures,
                                       self.kernel[k].value*u.Unit('cm5 K-1'),
                                       name=k)
            intensity_unit = u.cm**(-5) * self.kernel[k].unit
            intensity = self.data[k].data * self.data[k].unit
            uncertainty = self.data[k].uncertainty.array.squeeze() * self.data[k].unit
            eline = EmissionLine(
                cont_func=cont_func,
                intensity_obs=intensity.to_value(intensity_unit),
                sigma_intensity_obs=uncertainty.to_value(intensity_unit),
                name=k,
            )
            emission_lines.append(eline)
        temperature_bins = TempBins(self.temperature_bin_edges)
        result = predict_dem_emcee(emission_lines,
                                   temperature_bins,
                                   nwalkers=nwalkers,
                                   nsteps=nsteps)
        return result

    @classmethod
    def defines_model_for(self, *args, **kwargs):
        return kwargs.get('model') == 'mcmc'
    

def make_slope_map(dem,
                   temperature_bounds=None,
                   em_threshold=None,
                   rsquared_tolerance=0.5,
                   mask_negative=True):
    r"""
    Calculate emission measure slope :math:`a` in each pixel

    Create map of emission measure slopes by fitting :math:`\mathrm{EM}\sim T^a` for a
    given temperature range. A slope is masked if a value between the `temperature_bounds`
    is less than :math:`\mathrm{EM}`. Additionally, the "goodness-of-fit" is evaluated using
    the correlation coefficient, :math:`r^2=1 - R_1/R_0`, where :math:`R_1` and :math:`R_0`
    are the residuals from the first and zeroth order polynomial fits, respectively. We mask
    the slope if :math:`r^2` is less than `rsquared_tolerance`.

    Parameters
    ----------
    temperature_bounds : `~astropy.units.Quantity`, optional
    em_threshold : `~astropy.units.Quantity`, optional
        Mask slope if any emission measure in the fit interval is below this value
    rsquared_tolerance : `float`
        Mask any slopes with a correlation coefficient, :math:`r^2`, below this value
    mask_negative : `bool`
    """
    # TODO: move this somewhere more visible, e.g. synthesizAR
    if temperature_bounds is None:
        temperature_bounds = u.Quantity((1e6, 4e6), u.K)
    if em_threshold is None:
        em_threshold = u.Quantity(1e25, u.cm**(-5))
    # Get temperature fit array
    temperature_bin_centers = dem.axis_world_coords(0)[0]
    index_temperature_bounds = np.where(np.logical_and(
        temperature_bin_centers >= temperature_bounds[0],
        temperature_bin_centers <= temperature_bounds[1]
    ))
    temperature_fit = np.log10(
        temperature_bin_centers[index_temperature_bounds].to_value(u.K))
    if temperature_fit.size < 3:
        warnings.warn(f'Fitting to fewer than 3 points in temperature space: {temperature_fit}')
    # Cut on temperature
    data = u.Quantity(dem.data, dem.unit).T
    data = data[...,index_temperature_bounds].squeeze()
    # Get EM fit array
    em_fit = np.log10(data.value.reshape((np.prod(data.shape[:2]),) + data.shape[2:]).T)
    em_fit[np.logical_or(np.isinf(em_fit), np.isnan(em_fit))] = 0.0  # Filter before fitting
    # Fit to first-order polynomial
    coefficients, rss, _, _, _ = np.polyfit(temperature_fit, em_fit, 1, full=True,)
    slope_data = coefficients[0, :].reshape(data.shape[:2])
    # Apply masks
    _, rss_flat, _, _, _ = np.polyfit(temperature_fit, em_fit, 0, full=True,)
    rss = 0.*rss_flat if rss.size == 0 else rss  # returns empty residual when fit is exact
    rsquared = 1. - rss/rss_flat
    rsquared_mask = rsquared.reshape(data.shape[:2]) < rsquared_tolerance
    em_mask = np.any(data < em_threshold, axis=2)
    mask_list = [rsquared_mask, em_mask]
    if dem.mask is not None:
        mask_list.append(dem.mask.T[..., index_temperature_bounds].squeeze().any(axis=2))
    if mask_negative:
        mask_list.append(slope_data < 0)
    combined_mask = np.stack(mask_list, axis=2).any(axis=2).T
    # Build new map
    header = dem.wcs.low_level_wcs._wcs[0].to_header()
    header['temp_a'] = 10.**temperature_fit[0]
    header['temp_b'] = 10.**temperature_fit[-1]
    slope_map = sunpy.map.GenericMap(slope_data.T, header, mask=combined_mask)
    return slope_map


def read_cube_with_xarray(filename, axis_name, physical_type):
    """
    Read an xarray netCDF file and rebuild an NDCube

    This function reads a data cube from a netCDF file and rebuilds it
    as an NDCube. The assumption is that the attributes on the stored
    data array have the keys necessary to reconstitute a celestial FITS
    WCS and that the axis denoted by `axis_name` is the additional axis
    along which to extend that celestial WCS. This works only for 3D cubes
    where two of the axes correspond to spatial, celestial axes.

    Parameters
    ----------
    filename: `str`, path-like
        File to read from, usually a netCDF file
    axis_name: `str`
        The addeded coordinate along which to extend the celestial WCS.
    physical_type: `str`
        The physical type of `axis_name` as denoted by the IVOA designation.
    """
    ds = xarray.load_dataset(filename)
    meta = ds.attrs
    data = u.Quantity(ds['data'].data, meta.pop('unit'))
    mask = ds['mask'].data
    celestial_wcs = astropy.wcs.WCS(header=meta)
    axis_array = u.Quantity(ds[axis_name].data, ds[axis_name].attrs.get('unit'))
    combined_wcs = extend_celestial_wcs(celestial_wcs, axis_array, axis_name, physical_type)
    return ndcube.NDCube(data, wcs=combined_wcs, meta=meta, mask=mask)


def write_cube_with_xarray(cube, axis_name, filename):
    """
    Write an NDCube to a netCDF file

    This function writes an NDCube to a netCDF file by first expressing
    it as an xarray DataArray. This works only for 3D cubes where two of
    the axes correspond to spatial, celestial axes.

    Parameters
    ----------
    cube: `ndcube.NDCube`
    axis_name: `str`
    celestial_wcs: `astropy.wcs.WCS`
    filename: `str` or path-like
    """
    # FIXME: This is not a general solution and is probably really brittle
    celestial_wcs = cube.wcs.low_level_wcs._wcs[0]
    wcs_keys = dict(celestial_wcs.to_header())
    axis_array = cube.axis_world_coords(axis_name)[0]
    axis_coord = xarray.Variable(axis_name,
                                 axis_array.value,
                                 attrs={'unit': axis_array.unit.to_string()})
    if (mask:=cube.mask) is None:
        mask = np.full(cube.data.shape, False)
    cube_xa = xarray.Dataset(
        {'data': ([axis_name, 'lat', 'lon'], cube.data),
         'mask': ([axis_name, 'lat', 'lon'], mask)},
        coords={
            axis_name: axis_coord,
        },
        attrs={**wcs_keys, 'unit': cube.unit.to_string()}
    )
    cube_xa.to_netcdf(filename)