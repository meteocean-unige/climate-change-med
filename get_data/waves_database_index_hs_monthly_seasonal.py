from pathlib import Path
import xarray as xr
import numpy as np

from f_read import info_models, set_time
from f_aux import preprocess_set_index


def index_over_threshold(fn_monthly, ds, threshold):
    fn_mean = f'{fn_monthly.name.split("__")[0]}__mean{freq}_over_{thresh_lb}'

    # Mean value over threshold
    ds_overthresh = ds.where(ds > threshold)
    ds_overthresh_month = ds_overthresh.resample(time='1M').mean()
    ds_overthresh_season = ds_overthresh.resample(time='1Q-NOV').mean()

    ds_overthresh_month.groupby('time.month').mean().to_netcdf(fn_monthly.parent / f'{fn_mean}__monthlymean.nc')
    ds_overthresh_season.groupby('time.season').mean().to_netcdf(fn_monthly.parent / f'{fn_mean}__seasonalmean.nc')

    # Percentage of seastates over threshold
    n_overthresh_monthly = xr.where(ds > threshold, 1, 0).resample(time='1M').sum()
    perc_n_overthresh_monthly = (n_overthresh_monthly / ds.resample(time='1M').count()) * 100

    perc_n_overthresh_monthly.groupby('time.month').mean().to_netcdf(fn_monthly.parent / f'{fn_monthly.stem}_perc.nc')

    n_overthresh_seasonally = xr.where(ds > threshold, 1, 0).resample(time='1Q-NOV').sum()
    perc_n_overthresh_seasonally = (n_overthresh_seasonally / ds.resample(time='1Q-NOV').count()) * 100

    n_overthresh_seasonally.groupby('time.season').mean().to_netcdf(
        fn_monthly.parent / f'{"__".join(fn_monthly.name.split("__")[:2])}__seasonalmean.nc')
    perc_n_overthresh_seasonally.groupby('time.season').mean().to_netcdf(
        fn_monthly.parent / f'{"__".join(fn_monthly.name.split("__")[:2])}__seasonalmean_perc.nc')

    n_overthresh_monthly.groupby('time.month').mean().to_netcdf(fn_monthly)

    return


def index_over_threshold_duration(fn_monthly, ds, threshold):
    fn = f'{fn_monthly.name.split("__")[0]}__ndays_dmax_over_{thresh_lb}_2consecdays'

    ds_dailymax = ds.resample(time='1D').max()
    days_dailymax_overthresh = xr.where(ds_dailymax > threshold, 1, 0)
    ndays_dailymax_overthresh_2days = xr.where(days_dailymax_overthresh + days_dailymax_overthresh.shift(time=-1) == 2, 1, 0)

    # Percentage of seastates over threshold
    ndays_dailymax_overthresh_2days_monthly = ndays_dailymax_overthresh_2days.resample(time='1M').sum()
    perc_ndays_dailymax_overthresh_2days_monthly = (ndays_dailymax_overthresh_2days_monthly / ds_dailymax.resample(time='1M').count()) * 100

    ndays_dailymax_overthresh_2days_monthly.groupby('time.month').mean().to_netcdf(fn_monthly.parent / f'{fn}__monthlymean.nc')
    perc_ndays_dailymax_overthresh_2days_monthly.groupby('time.month').mean().to_netcdf(fn_monthly.parent / f'{fn}__monthlymean_perc.nc')

    ndays_dailymax_overthresh_2days_seasonally = ndays_dailymax_overthresh_2days.resample(time='1Q-NOV').sum()
    perc_ndays_dailymax_overthresh_2days_seasonally = (ndays_dailymax_overthresh_2days_seasonally / ds_dailymax.resample(time='1Q-NOV').count()) * 100

    ndays_dailymax_overthresh_2days_seasonally.groupby('time.season').mean().to_netcdf(fn_monthly.parent / f'{fn}__seasonalmean.nc')
    perc_ndays_dailymax_overthresh_2days_seasonally.groupby('time.season').mean().to_netcdf(fn_monthly.parent / f'{fn}__seasonalmean_perc.nc')

    return


def index_over_threshold_case(exp, period, fn, threshold, dir_data, index_duration=False):

    if exp == 'hindcast':
        ds = xr.open_mfdataset(dir_data.glob('WW3_mediterr_*.grb2'), engine='cfgrib', parallel=True,
                               preprocess=set_time)
        ds = ds.swh
        ds = ds.rename_vars({'swh': 'hs'})
    if exp == 'rcp85':
        if bias_adjusted:
            ds = xr.open_mfdataset(dir_data.glob(f'WW3_{info_model[-1]}_*_eqm_month_hs.nc'), parallel=True,
                                   preprocess=preprocess_set_index)

    ds = ds.sel(time=slice(f'{period[0]}-01-01', f'{period[1]}-12-31')).hs

    if index_duration:
        index_over_threshold_duration(fn, ds, threshold)
    else:
        index_over_threshold(fn, ds, threshold)

    return


ana_per = 'mid', 'end'
models = np.arange(1, 18)
freq = 'seastates'
bias_adjusted = True

data_hindcast = True
data_cordex = True

thresh = {'ss3-1p25': 1.25, 'ss4-2p5': 2.5, 'ss5-4': 4, 'p90': 0.9, 'p95': 0.95}

info_per = {'historical':  [1979, 2005], 'mid': [2034, 2060], 'end': [2074, 2100]}

p = Path('/vhe/nasmetocean/wavewatch/cordex/climate_change')
dir_base_cordex = Path('/vhe/nasmetocean/wavewatch/cordex')
dir_hindcast = Path('/vhe/nasmetocean/wavewatch/wwiii/hindcast_10km_med/DICCA')

dir_data = p / 'data' / f'wave_dataset'

for thresh_lb in thresh.keys():
    threshold = thresh[thresh_lb]
    index_duration = False
    if thresh_lb[0] == 'p':
        quantiles_hind = xr.open_dataarray(dir_data / 'hindcast_hs_1979_2005__quantiles.nc')
        threshold = quantiles_hind.sel(quantile=threshold)
        index_duration = True

    # HINDCAST
    if data_hindcast:
        exp = 'hindcast'

        fn_n_monthly = dir_data / f'{exp}_hs_{info_per["historical"][0]}_{info_per["historical"][1]}__n{freq}_over_' \
                                  f'{thresh_lb}__monthlymean.nc'
        if not fn_n_monthly.is_file():
            index_over_threshold_case(exp, info_per["historical"], fn_n_monthly, threshold, dir_hindcast)

        if index_duration:
            fn_n_monthly = dir_data / f'{exp}_hs_{info_per["historical"][0]}_{info_per["historical"][1]}__ndays_dmax_over_{thresh_lb}_2consecdays__monthlymean.nc'
            if not fn_n_monthly.is_file():
                index_over_threshold_case(exp, info_per["historical"], fn_n_monthly, threshold, dir_hindcast, index_duration=True)

    # FUTURE
    if data_cordex:
        exp = 'rcp85'
        for mii, model in enumerate(models):

            if exp == 'historical':
                ana_per = 'historical',
            elif exp == 'rcp85':
                ana_per = 'mid', 'end'

            info_model = info_models(model)
            dir_cordex_wave = dir_base_cordex / info_model[0] / 'wave'

            dir_cordex_files = dir_cordex_wave / f'{exp}__eqm_month'
            exp_lb = f'ba_eqm_month_{exp}'

            mod_name = info_model[2]

            for per in ana_per:
                fn_n_monthly = dir_data / f'{mod_name}_{exp_lb}_{per}_hs_{info_per[per][0]}_{info_per[per][1]}' \
                                          f'__n{freq}_over_{thresh_lb}__monthlymean.nc'
                if not fn_n_monthly.is_file():
                    index_over_threshold_case(exp, info_per[per], fn_n_monthly, threshold, dir_cordex_files)

                if index_duration:
                    fn_n_monthly = dir_data / f'{mod_name}_{exp_lb}_{per}_hs_{info_per[per][0]}_{info_per[per][1]}' \
                                              f'__ndays_dmax_over_{thresh_lb}_2consecdays__monthlymean.nc'
                    if not fn_n_monthly.is_file():
                        index_over_threshold_case(exp, info_per[per], fn_n_monthly, threshold, dir_cordex_files, index_duration=index_duration)
