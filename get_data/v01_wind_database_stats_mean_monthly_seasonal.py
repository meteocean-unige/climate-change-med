import numpy as np
import xarray as xr
from pathlib import Path
from f_read import info_models, set_time
import xclim.core.calendar as xclimcl
###from scipy.stats import mean, std

def stats(exp, ds, var, period):
    fn_lst = dir_data / f'{exp}_{var}_{period[0]}_{period[1]}__meanseasonalmmin.nc'
    #fn_lst = dir_data / f'{exp}_{var}_{period[0]}_{period[1]}__seasonalquantiles.nc'    ### CONTROLLO CHE I FILE SIANO STATI GENERATI FACENDO UN CHECK SULL'ULTIMO FILE CHE DEVE ESSERE GENERATO
    if not fn_lst.is_file():
        ds_var_gr = ds[var].groupby('time.month')
        ds[var].resample(time='1M').max().groupby('time.month').mean().to_netcdf(dir_data / f'{exp}_{var}_{period[0]}_{period[1]}__meanmonthlymax.nc')
        ds[var].resample(time='1M').min().groupby('time.month').mean().to_netcdf(dir_data / f'{exp}_{var}_{period[0]}_{period[1]}__meanmonthlymin.nc')

        ds_var_gr = ds[var].groupby('time.season')
        ds[var].resample(time='1Q-NOV').max().groupby('time.season').mean().to_netcdf(dir_data / f'{exp}_{var}_{period[0]}_{period[1]}__meanseasonalmax.nc')
        ds[var].resample(time='1Q-NOV').min().groupby('time.season').mean().to_netcdf(dir_data / f'{exp}_{var}_{period[0]}_{period[1]}__meanseasonalmmin.nc')


data_hindcast = False
data_cordex = True

models = np.arange(1, 21) 
info_per = {'historical':  [1979, 2005], 'rcp85_mid': [2034, 2060], 'rcp85_end': [2074, 2100]}

p = Path('/vhe/nasmetocean/wavewatch/cordex/climate_change')
dir_base_cordex = Path('/vhe/nasmetocean/wavewatch/cordex')
dir_hindcast = Path('/vhe/nasmetocean/wavewatch/wrf/hindcast_med_eur11_rotpole')
dir_data = p / 'data' / f'wind_dataset'

# HINDCAST
if data_hindcast:
    ds = xr.open_mfdataset(dir_hindcast.glob('resampled6H_remapped_*.nc'), parallel=True, preprocess=set_time)
    ds = ds.sel(time=slice(f'{info_per["historical"][0]}-01-01', f'{info_per["historical"][1]}-12-31'))
    #ds = ds.rename_vars({'uas': 'uas', 'vas': 'vas'})   
    ds['wind'] = np.sqrt(ds['uas'] ** 2 + ds['vas'] ** 2)
    rho = 1.25
    ds['wind_energy'] = 1 / 2 * rho * np.sqrt(ds['uas'] ** 2 + ds['vas'] ** 2) ** 3 / 1000  # kW/m2
    #ds = ds.where((ds.tp > 0) & (ds.tp < 20))
    ds = ds.sel(time=~ds.indexes['time'].duplicated())
    for var in ['wind', 'wind_energy']:
        stats('hindcast', ds, var, info_per["historical"])

    fn=dir_data / f'hindcast_wind_{info_per["historical"][0]}_{info_per["historical"][1]}__quantiles.nc'
    if not fn.is_file():
        ds.wind.quantile([0.1, 0.5, 0.9, 0.95, 0.99],dim='time').to_netcdf(fn)

# CORDEX raw
if data_cordex:
    for mii, model in enumerate(models):
        for exp in ['historical', 'rcp85_mid', 'rcp85_end']:

            suff = ''
            info_model,_,_ = info_models(model,rcp=f'{exp.split("_")[0]}{suff}')
            print(f'{exp.split("_")[0]}{suff}')
            print(info_model)

            mod_name = f'{info_model[2]}_raw'
            vars = ['wind', 'wind_energy']

            dir_cordex_hist = dir_base_cordex / info_model[0] / 'wind' / f'{exp.split("_")[0]}{suff}'
            print(dir_cordex_hist)
            print(info_model[4])
            print(f'uas_{info_model[4]}_*.nc')
            #dir_cordex = dir_base_cordex / info_model[0] / 'wind' / f'{exp.split("_")[0]}'
            ds_cordex_hist = xr.open_mfdataset(dir_cordex_hist.glob(f'uas_{info_model[4]}_*.nc'), parallel=True)
            ds_vas = xr.open_mfdataset(dir_cordex_hist.glob(f'vas_{info_model[4]}_*.nc'), parallel=True)

            # Convert dates to default calendar
            if isinstance(ds_cordex_hist.indexes['time'], xr.CFTimeIndex) and model != 11 and  model != 7 and model !=11 and model != 17:
                print('################')
                print(model)
               	ds_cordex_hist['time'] = ds_cordex_hist.indexes['time'].to_datetimeindex()
                ds_vas['time'] = ds_vas.indexes['time'].to_datetimeindex()
            else:
                ds_cordex_hist = xclimcl.convert_calendar(ds_cordex_hist, 'default', align_on='date')
                ds_vas = xclimcl.convert_calendar(ds_vas, 'default', align_on='date')

            dir_cordex_hist = dir_base_cordex / info_model[0] / 'wind' / f'{exp.split("_")[0]}{suff}'    
            ds_cordex_hist = ds_cordex_hist.sel(time=~ds_cordex_hist.indexes['time'].duplicated())
            #ds_cordex['tp'] = 1 / ds_cordex['fp']
            #ds_cordex = ds_cordex.where((ds_cordex.tp > 0) & (ds_cordex.tp < 20))
            #ds_cordex = ds_cordex.rename_vars({'t0m1': 'tm', 'dir': 'dirm'})
            ds_cordex_hist['wind'] = np.sqrt(ds_cordex_hist['uas'] ** 2 + ds_vas['vas'] ** 2)
            rho = 1.25
            ds_cordex_hist['wind_energy'] = 1 / 2 * rho * np.sqrt(ds_cordex_hist['uas'] ** 2 + ds_vas['vas'] ** 2) ** 3 / 1000  # kW/m2
            ds_cordex_hist = ds_cordex_hist.sel(time=slice(f'{info_per[exp][0]}-01-01', f'{info_per[exp][1]}-12-31'))

            for var in vars:
                stats(f'{mod_name}_{exp}', ds_cordex_hist, var, info_per[exp])
