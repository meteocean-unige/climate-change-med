import numpy as np
import xarray as xr
from pathlib import Path
from f_read import info_models, set_time
from scipy.stats import circmean, circstd


def stats(exp, ds, var, period):
    fn_lst = dir_data / f'{exp}_{var}_{period[0]}_{period[1]}__seasonalquantiles.nc'
    if not fn_lst.is_file():
        ds_var_gr = ds[var].groupby('time.month')
        ds_var_gr.mean().to_netcdf(dir_data / f'{exp}_{var}_{period[0]}_{period[1]}__monthlymean.nc')
        ds_var_gr.max().to_netcdf(dir_data / f'{exp}_{var}_{period[0]}_{period[1]}__monthlymax.nc')
        ds[var].chunk({'time': -1}).groupby('time.month').quantile([0.1, 0.5, 0.9, 0.95, 0.99]).to_netcdf(
            dir_data / f'{exp}_{var}_{period[0]}_{period[1]}__monthlyquantiles.nc')

        ds_var_gr = ds[var].groupby('time.season')
        ds_var_gr.mean().to_netcdf(dir_data / f'{exp}_{var}_{period[0]}_{period[1]}__seasonalmean.nc')
        ds_var_gr.max().to_netcdf(dir_data / f'{exp}_{var}_{period[0]}_{period[1]}__seasonalmax.nc')
        ds[var].chunk({'time': -1}).groupby('time.season').quantile([0.1, 0.5, 0.9, 0.95, 0.99]).to_netcdf(
            dir_data / f'{exp}_{var}_{period[0]}_{period[1]}__seasonalquantiles.nc')


data_hindcast = False
data_cordex = True

models = np.arange(1, 18)
info_per = {'historical':  [1979, 2005], 'rcp85_mid': [2034, 2060], 'rcp85_end': [2074, 2100]}

p = Path('/vhe/nasmetocean/wavewatch/cordex/climate_change')
dir_base_cordex = Path('/vhe/nasmetocean/wavewatch/cordex')
dir_hindcast = Path('/vhe/nasmetocean/wavewatch/wwiii/hindcast_10km_med/DICCA/grib')
dir_data = p / 'data' / f'wave_dataset'

# HINDCAST
if data_hindcast:
    ds = xr.open_mfdataset(dir_hindcast.glob('WW3_mediterr_*.grb2'), engine='cfgrib', parallel=True, preprocess=set_time)
    ds = ds.sel(time=slice(f'{info_per["historical"][0]}-01-01', f'{info_per["historical"][1]}-12-31'))
    ds = ds.rename_vars({'swh': 'hs', 'mwp': 'tm', 'perpw': 'tp', 'mwd': 'dirm'})
    ds = ds.where((ds.tp > 0) & (ds.tp < 20))
    ds = ds.sel(time=~ds.indexes['time'].duplicated())
    for var in ['hs', 'tm', 'tp']:
        stats('hindcast', ds, var, info_per["historical"])

    # mean direction
    for st in [circmean, circstd]:
        stn = st.__name__[4:]
        ds_d = xr.apply_ufunc(st, ds['dirm'].groupby(f'time.month'), dask='parallelized', input_core_dims=[["time"]],
                              kwargs=dict(high=0, low=360, axis=-1, nan_policy='omit'),
                              dask_gufunc_kwargs={'allow_rechunk': True})
        ds_d.to_netcdf(dir_data / f'hindcast_dirm_{info_per["historical"][0]}_{info_per["historical"][1]}__monthly{stn}.nc')

        ds_d = xr.apply_ufunc(st, ds['dirm'].groupby(f'time.season'), dask='parallelized', input_core_dims=[["time"]],
                              kwargs=dict(high=0, low=360, axis=-1, nan_policy='omit'),
                              dask_gufunc_kwargs={'allow_rechunk': True})
        ds_d.to_netcdf(dir_data / f'hindcast_dirm_{info_per["historical"][0]}_{info_per["historical"][1]}__seasonal{stn}.nc')


# CORDEX raw
if data_cordex:
    for mii, model in enumerate(models):
        info_model = info_models(model)
        print(info_model[2])

        suff = ''
        mod_name = f'{info_model[2]}_raw'
        vars = ['hs', 'tm', 'tp']

        for exp in ['historical', 'rcp85_mid', 'rcp85_end']:
            dir_cordex = dir_base_cordex / info_model[0] / 'wave' / f'{exp.split("_")[0]}{suff}'

            ds_cordex = xr.open_mfdataset(dir_cordex.glob(f'WW3_{info_model[-1]}_*.nc'), parallel=True)
            ds_cordex = ds_cordex.sel(time=~ds_cordex.indexes['time'].duplicated())
            ds_cordex['tp'] = 1 / ds_cordex['fp']
            ds_cordex = ds_cordex.where((ds_cordex.tp > 0) & (ds_cordex.tp < 20))
            ds_cordex = ds_cordex.rename_vars({'t0m1': 'tm', 'dir': 'dirm'})
            ds_cordex = ds_cordex.sel(time=slice(f'{info_per[exp][0]}-01-01', f'{info_per[exp][1]}-12-31'))

            for var in vars:
                stats(f'{mod_name}_{exp}', ds_cordex, var, info_per[exp])

            # mean direction
            for st in [circmean, circstd]:
                stn = st.__name__[4:]
                fn = dir_data / f'{mod_name}_{exp}_dirm_{info_per[exp][0]}_{info_per[exp][1]}__seasonal{stn}.nc'
                if not fn.is_file():
                    ds_d = xr.apply_ufunc(st, ds_cordex['dirm'].groupby(f'time.month'), dask='parallelized',
                                          input_core_dims=[["time"]],
                                          kwargs=dict(high=0, low=360, axis=-1, nan_policy='omit'),
                                          dask_gufunc_kwargs={'allow_rechunk': True})
                    ds_d.to_netcdf(dir_data / f'{mod_name}_{exp}_dirm_{info_per[exp][0]}_{info_per[exp][1]}'
                                              f'__monthly{stn}.nc')

                    ds_d = xr.apply_ufunc(st, ds_cordex['dirm'].groupby(f'time.season'), dask='parallelized',
                                          input_core_dims=[["time"]],
                                          kwargs=dict(high=0, low=360, axis=-1, nan_policy='omit'),
                                          dask_gufunc_kwargs={'allow_rechunk': True})
                    ds_d.to_netcdf(dir_data / f'{mod_name}_{exp}_dirm_{info_per[exp][0]}_{info_per[exp][1]}'
                                              f'__seasonal{stn}.nc')

