import xarray as xr
import pandas as pd


def preprocess_set_index(ds):
    ds = ds.sel(time=~ds.indexes['time'].duplicated())

    ds_index = pd.DatetimeIndex(ds.time.data)
    if ds_index.year[-1] != ds_index.year[0]:
        ds = ds.isel(time=slice(None, -1))

    if not pd.DatetimeIndex(ds.time.data).is_monotonic_increasing:
        ds = ds.sortby('time')

    return ds