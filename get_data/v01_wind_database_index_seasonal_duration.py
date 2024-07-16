from pathlib import Path
import xarray as xr
import numpy as np
import xclim.core.calendar as xclimcl

from f_read import info_models


def find_consecutive_events(data: np.ndarray, threshold:int=1, length:int=3, over=True):
    """Find runs of values in a 1D array above a given threshold."""
    if over:
        over_threshold = (data >= threshold)
    else:
        over_threshold = (data < threshold)
    # Counts "transitions" between 1/0 state in over_threshold
    cs=np.r_[[0], np.not_equal(over_threshold[:-1], over_threshold[1:])].cumsum()
    _, i, c = np.unique(cs, return_index=1, return_counts=1)
    # i and c are the index and count of each transition between runs of {length} or more.
    for index, count in zip(i,c):
        if over:
            if data[index] >= threshold and count >= length:
                yield (index, count)
        else:
            if data[index] < threshold and count >= length:
                yield (index, count)


def number_of_events(sample, threshold=1, length=3, over=True):
    """Compute how many 'runs' are in the time dimension?"""
    events = list(find_consecutive_events(sample, threshold=threshold, length=length, over=over))
    return len(events)


def mean_event_length(sample, threshold=1, length=3, over=True):
    """Compute how long is the average event of given length."""
    avg = 0
    for n, (_, c) in enumerate(find_consecutive_events(sample, threshold=threshold, length=length, over=over)):
        avg = ((avg * n) + c ) / (n+1)
    return avg


def index_over_threshold_duration(fn_number_events, ds, threshold, over, ndays):
    ds_dailymax = ds.resample(time='1D').max()
    ds_dailymax = ds_dailymax.chunk(chunks={'rlat':50, 'time':-1, 'rlon':50})

    ds_mean_event_len = xr.apply_ufunc(
        mean_event_length,
        ds_dailymax,
        vectorize=True,
        input_core_dims=[['time']],
        dask='parallelized',
        kwargs={'length': ndays, 'threshold': threshold, 'over': over}
    )
    fn = fn_number_events.parent / \
         f'{fn_number_events.stem.split("__")[0]}__mean_event_length__{fn_number_events.stem.split("__")[2]}.nc'
    ds_mean_event_len.to_netcdf(fn)

    ds_num_events = xr.apply_ufunc(
        number_of_events,
        ds_dailymax,
        vectorize=True,
        input_core_dims=[['time']],
        dask='parallelized',
        kwargs={'length': ndays, 'threshold': threshold, 'over': over}
    )
    ds_num_events.to_netcdf(fn_number_events)

    return


models = 4, np.arange(1, 22)
thresh_info = [3, 2, False], [3, 3, False], [3, 4, False], [25, 2, True], [25, 3, True], [25, 4, True]
info_per = {'historical':  [1979, 2005], 'rcp85_mid': [2034, 2060], 'rcp85_end': [2074, 2100]}

data_cordex = True

cineca = False
cineca_user = 'andrea'
if cineca:
    if cineca_user == 'andrea':
        p = Path('/g100_work/IscrC_WIDEMED/data')
    elif cineca_user == 'gio':
        p = Path('/g100_work/IscrC_SAFARI/aliraloa/data')
    elif cineca_user == 'comaris':
        p = Path('/g100_work/IscrB_COMARIS/fgiaroli')

    dir_base_cordex = p / 'cordex'
    dir_hindcast = p / 'hindcast'
    dir_data = p / 'wind_index_dataset_duration'
else:
    dir_hindcast = Path('/vhe/nasmetocean/wavewatch/wrf/hindcast_med_eur11_rotpole')
    dir_base_cordex = Path('/vhe/nasmetocean/wavewatch/cordex')

    p = Path('/vhe/nasmetocean/wavewatch/cordex/climate_change')
    dir_data = p / 'data' / 'wind_index_dataset_duration'

var = 'wind' # ['wind', 'wind_energy']

# CORDEX raw
if data_cordex:
    for mii, model in enumerate(models):
        for exp in ['historical', 'rcp85_mid', 'rcp85_end']:

            suff = ''
            info_model, _, _ = info_models(model,rcp=f'{exp.split("_")[0]}{suff}')
            print(f'{exp.split("_")[0]}{suff}')
            mod_name = f'{info_model[2]}_raw'

            dir_cordex_hist = dir_base_cordex / info_model[0] / 'wind' / f'{exp.split("_")[0]}{suff}'
            print(dir_cordex_hist)
            print(info_model[4])
            ds_cordex_hist = xr.open_mfdataset(dir_cordex_hist.glob(f'uas_{info_model[4]}_*.nc'), parallel=True)
            ds_vas = xr.open_mfdataset(dir_cordex_hist.glob(f'vas_{info_model[4]}_*.nc'), parallel=True)

            # Convert dates to default calendar
            if isinstance(ds_cordex_hist.indexes['time'], xr.CFTimeIndex) and not model in [1, 7, 17]:
                print('################')
                print(model)
                ds_cordex_hist['time'] = ds_cordex_hist.indexes['time'].to_datetimeindex()
                ds_vas['time'] = ds_vas.indexes['time'].to_datetimeindex()
            else:
                ds_cordex_hist = xclimcl.convert_calendar(ds_cordex_hist, 'default', align_on='date')
                ds_vas = xclimcl.convert_calendar(ds_vas, 'default', align_on='date')

            ds_cordex_hist['wind'] = np.sqrt(ds_cordex_hist['uas'] ** 2 + ds_vas['vas'] ** 2)
            ds_cordex_hist = ds_cordex_hist.sel(time=~ds_cordex_hist.indexes['time'].duplicated())

            ds_cordex_hist = ds_cordex_hist.sel(time=slice(f'{info_per[exp][0]}-01-01', f'{info_per[exp][1]}-12-31'))

            for threshold, ndays, over in thresh_info:

                if over:
                    overlb = 'over'
                else:
                    overlb = 'under'

                fn_number_events = dir_data / f'{mod_name}_{exp}_{var}_{info_per[exp][0]}_{info_per[exp][1]}' \
                                              f'__number_events__dmax_{overlb}_{threshold}_{ndays}consecdays.nc'

                if not fn_number_events.is_file():
                    index_over_threshold_duration(fn_number_events, ds_cordex_hist[var], threshold, over, ndays)