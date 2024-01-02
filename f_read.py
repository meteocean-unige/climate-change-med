# coding: utf-8
import numpy as np
import pandas as pd
from pathlib import Path
import cftime
from _cordex.cfcalendar import to_datetimeindex
import xarray as xr
import re

GCM = {1: {'abr': 'MIROC5', 'full': 'MIROC-MIROC5'},
       2: {'abr': 'CanESM2', 'full': 'CCCma-CanESM2'},
       3: {'abr': 'CNRM-CM5', 'full': 'CNRM-CERFACS-CNRM-CM5'},
       4: {'abr': 'HadGEM2-ES', 'full': 'MOHC-HadGEM2-ES'},
       5: {'abr': 'EC-EARTH', 'full': 'ICHEC-EC-EARTH'},
       6: {'abr': 'MPI-ESM-LR', 'full': 'MPI-M-MPI-ESM-LR'},
       7: {'abr': 'IPSL-CM5A-MR', 'full': 'IPSL-IPSL-CM5A-MR'},
       8: {'abr': 'NorESM1-M', 'full': 'NCC-NorESM1-M'}}

institute_id = {1: 'CLMcom', 2: 'SMHI', 3: 'DMI', 4: 'CNRM', 5: 'ICTP', 6: 'CLMcom-ETH', 7:'KNMI'}
RCM = {1: 'CCLM4-8-17', 2: 'RCA4', 3: 'HIRHAM5', 4:'ALADIN63', 5: 'RegCM4-6', 6: 'COSMO-crCLIM-v1-1', 7:'RACMO22E'}
RCM_out = {1: 'CCLM4', 2: 'RCA4', 3: 'HIRHAM5', 4:'ALADIN63', 5: 'RegCM46', 6: 'COSMO-crCLIM1', 7:'RACMO22E'}

ensemble = {1: 'r1i1p1', 12:'r12i1p1', 3:'r3i1p1'}

# bias_dict = {3:'v1-IPSL-CDFT22-WFDEI-1979-2005', 5:'v1-IPSL-CDFT22-WFDEI-1979-2005',
#              6:'v1-IPSL-CDFT22-WFDEI-1979-2005', 7:'v1-IPSL-CDFT22-ERA-Interim-1979-2005',
#              8:'v1-IPSL-CDFT22-ERA-Interim-1979-2005'}
bias_dict = {3:'v1-IPSL-CDFT22-WFDEI-1979-2005', 5:'v1-IPSL-CDFT22-WFDEI-1979-2005',
             6:'v1-IPSL-CDFT22-WFDEI-1979-2005', 7:'v1-IPSL-CDFT22-WFDEI-1979-2005'}

"""EURO-CORDEX models DICCA

    WW3:
   1: CLMcom_CanESM2
   2: CLMcom_MIROC5
   3: SMHI_MPI-ESM-LR
   4: SMHI_NorESM1-M
   5: SMHI_CNRM-CM5
   6: SMHI_IPSL-CM5A-MR
   7: SMHI_HadGEM2-ES
   8: SMHI_EC-EARTH
   9: DMI_EC-EARTH
   10: DMI_NorESM1-M
   11: DMI_HadGEM2-ES
   12: DMI_MPI-ESM-LR
   13: DMI_CNRM-CM5
   14: DMI_IPSL-CM5A-MR
   15: CLMcom-ETH_EC-EARTH
   16: CLMcom-ETH_NorESM1-M
   17: CLMcom-ETH_HadGEM2-ES
   18: RACMO22E-CNRM-CM5
   19: RACMO22E-EC-EARTH
   20: RACMO22E-IPSL-CM5A-MR
   21: RACMO22E-HadGEM2-ES
   SCHISM:
   22: ALADIN63_CNRM-CM5
   23: ALADIN63_HadGEM2-ES
   24: ALADIN63_NorESM1-M
  """

models_dicca = {1: {'institute_id': 1, 'GCM': 2, 'ensemble': 1, 'project': 'CORDEX-Reklies', 'version': '1'},
                2: {'institute_id': 1, 'GCM': 1, 'ensemble': 1, 'project': 'CORDEX-Reklies', 'version': '1'},
                3: {'institute_id': 2, 'GCM': 6, 'ensemble': 1, 'project': 'CORDEX', 'version': '1a'},
                4: {'institute_id': 2, 'GCM': 8, 'ensemble': 1, 'project': 'CORDEX', 'version': '1'},
                5: {'institute_id': 2, 'GCM': 3, 'ensemble': 1, 'project': 'CORDEX', 'version': '1'},
                6: {'institute_id': 2, 'GCM': 7, 'ensemble': 1, 'project': 'CORDEX', 'version': '1'},
                7: {'institute_id': 2, 'GCM': 4, 'ensemble': 1, 'project': 'CORDEX', 'version': '1'},
                8: {'institute_id': 2, 'GCM': 5, 'ensemble': 1, 'project': 'CORDEX', 'version': '1'},
                9: {'institute_id': 3, 'GCM': 5, 'ensemble': 1, 'project': 'CORDEX', 'version': '1'},
                10: {'institute_id': 3, 'GCM': 8, 'ensemble': 1, 'project': 'CORDEX', 'version': '3'},
                11: {'institute_id': 3, 'GCM': 4, 'ensemble': 1, 'project': 'CORDEX', 'version': '2'},
                12: {'institute_id': 3, 'GCM': 6, 'ensemble': 1, 'project': 'CORDEX', 'version': '1'},
                13: {'institute_id': 3, 'GCM': 3, 'ensemble': 1, 'project': 'CORDEX', 'version': '2'},
                14: {'institute_id': 3, 'GCM': 7, 'ensemble': 1, 'project': 'CORDEX', 'version': '1'},
                # 15: {'institute_id': 5, 'GCM': 4, 'ensemble': 1, 'project': 'CORDEX', 'version': '1'},
                # 16: {'institute_id': 5, 'GCM': 6, 'ensemble': 1, 'project': 'CORDEX', 'version': '1'},
                # 17: {'institute_id': 5, 'GCM': 8, 'ensemble': 1, 'project': 'CORDEX', 'version': '1'},
                15: {'institute_id': 6, 'GCM': 5, 'ensemble': 1, 'project': 'CORDEX', 'version': '1'},
                16: {'institute_id': 6, 'GCM': 8, 'ensemble': 1, 'project': 'CORDEX', 'version': '1'},
                17: {'institute_id': 6, 'GCM': 4, 'ensemble': 1, 'project': 'CORDEX', 'version': '1'},
                # 17: {'institute_id': 3, 'GCM': 5, 'ensemble': 3, 'project': 'CORDEX', 'version': '1'},
                # 20: {'institute_id': 4, 'GCM': 3, 'ensemble': 1, 'project': 'CORDEX', 'version': '2'}
                18: {'institute_id': 7, 'GCM': 3, 'ensemble': 1, 'project': 'CORDEX', 'version': '2'},
                19: {'institute_id': 7, 'GCM': 5, 'ensemble': 1, 'project': 'CORDEX', 'version': '1'},
                20: {'institute_id': 7, 'GCM': 7, 'ensemble': 1, 'project': 'CORDEX', 'version': '1'},
                21: {'institute_id': 7, 'GCM': 4, 'ensemble': 1, 'project': 'CORDEX', 'version': '2'},
                22: {'institute_id': 4, 'GCM': 3, 'ensemble': 1, 'project': 'CORDEX', 'version': '2'},
                }

points_spectra = {"003506": {'coords': [5.6873, 40.71], 'area': 'western',
                             'DJF': [0.024, [292.5, 360]], 'MAM': [0.0, [0, 0]], 'JJA': [0.0, [0, 0]], 'SON': [0.0, [0, 0]]},
                  "000396": {'coords': [8.8707, 43.86], 'area': 'ligurian'},
                  "013651": {'coords': [17.784, 35.76], 'area': 'central'},
                  "000246": {'coords': [13.964, 44.31], 'area': 'adriatic'},
                  "012365": {'coords': [-4.4993, 36.21], 'area': 'alboran'},
                  "006729": {'coords': [19.0574, 38.91], 'area': 'ionian'},
                  "004257": {'coords': [24.7874, 40.26], 'area': 'aegean'},
                  "006694": {'coords': [13.964, 38.91], 'area': 'tyrrhenian'},
                  "022123": {'coords': [19.0574, 31.26], 'area': 'sidra'},
                  "018735": {'coords': [30.5174, 33.51], 'area': 'levantine'},
                  "013786": {'coords': [34.974, 35.76], 'area': 'eastern'}}


urls_info_north_atlantic = {'EC_EARTH': {'historical': {'path': 'EC_EARTH', 'fn': 'EC_EARTH', 'ens': 'r2i1p1'},
                                         'rcp85': {'path': 'EC_EARTH', 'fn': 'EC_EARTH', 'ens': 'r2i1p1'}, },
                            'HadGEM2_ES': {'historical': {'path': 'HadGEM2_ES', 'fn': 'HadGEM2-ES', 'ens': 'r2i1p1'},
                                           'rcp85': {'path': 'HadGEM2_ES', 'fn': 'HadGEM2-ES', 'ens': 'r1i1p1'}}}


def info_cordex_down(model):

    info_model = models_dicca[model]

    info_model_down = [institute_id[info_model['institute_id']],
                       GCM[info_model['GCM']]['full'],
                       ensemble[info_model['ensemble']],
                       info_model['project'],
                       info_model['version']
                       ]

    return info_model_down


def info_models(model, freq='6hr', rcp='rcp85', bias_adjust=False):

    info_model = models_dicca[model]

    if bias_adjust:
        bias_val = bias_dict[model]
    else:
        bias_val = None

    ensemble_mod = ensemble[info_model['ensemble']]
    if model == 8 and bias_val is not None:
        ensemble_mod = ensemble[12]

    models_cordex = name_cordex_exp(institute_id[info_model['institute_id']], GCM[info_model['GCM']], ensemble_mod,
                                    [RCM[info_model['institute_id']], RCM_out[info_model['institute_id']]],
                                    info_model['version'], freq, rcp=rcp, bias_adj=bias_val)

    return models_cordex


def name_cordex_exp(institute_id, GCM, ensemble, RCMs, downs, time_fq, rcp, bias_adj=None):

    RCMi = RCMs[0]
    RCM_outi = RCMs[1]

    name_rcm = 'EUR-11_{}-{}_{}-{}_v{}_{}'.format(institute_id, GCM['full'], ensemble, RCMi, downs, time_fq)
    name_wave_data = '{}-{}'.format(institute_id, GCM['full'])
    name_out = '{}-{}'.format(RCM_outi, GCM['abr'])
    name_cordex = 'EUR-11_{}-{}_{}_{}_{}-{}_v{}_{}'.format(GCM['abr'].split('-')[0], GCM['full'], rcp, ensemble,
                                                           institute_id, RCMi, downs, time_fq)
    name_cordex_2 = 'EUR-11_{}_{}_{}_{}-{}_v{}_{}'.format(GCM['full'], rcp, ensemble, institute_id, RCMi, downs,
                                                          time_fq)
    name_cordex_ww3 = 'EUR-11_{}_{}_{}-{}_v{}_{}'.format(GCM['full'], ensemble, institute_id, RCMi, downs, time_fq)

    if bias_adj is not None:
        name_rcm = 'EUR-11_{}-{}_{}-{}_{}_{}'.format(institute_id, GCM['full'], ensemble, RCMi, bias_adj, time_fq)
        name_cordex = 'EUR-11_{}-{}_{}_{}_{}-{}_{}_{}'.format(GCM['abr'].split('-')[0], GCM['full'], rcp, ensemble,
                                                                  institute_id, RCMi, bias_adj, time_fq)
        name_cordex_2 = 'EUR-11_{}_{}_{}_{}-{}_{}_{}'.format(GCM['full'], rcp, ensemble, institute_id, RCMi,
                                                                bias_adj, time_fq)

    return name_rcm, name_wave_data, name_out, name_cordex, name_cordex_2, name_cordex_ww3


def info_models_vars(model, var):

    institute, driving_model, ensemblei, projecti, versioni = info_cordex_down(model)
    project = [projecti]
    version = [versioni]
    ensemble = [ensemblei]

    if var == 'pr':
        var_name = ['Precipitation']
        var_label = ['pr']
        freq = ['3hr']
        if bias_dict.get(model):
            var_name.append('Bias-Adjusted Precipitation')
            var_label.append('prAdjust')
            freq.append('day')
            project.append('CORDEX-Adjust')
            version.append('v1')
            if model == 8:
                ensemble.append('r12i1p1')
            else:
                ensemble.append(ensemblei)
    elif var == 'tas':
        var_name = ['Near-Surface Air Temperature']
        var_label = ['tas']
        freq = ['3hr']
        if bias_dict.get(model):
            var_name.append('Bias-Adjusted Near-Surface Air Temperature')
            var_label.append('tasAdjust')
            freq.append('3hr')
            project.append('CORDEX-Adjust')
            version.append('v1')
            if model == 8:
                ensemble.append('r12i1p1')
            else:
                ensemble.append(ensemblei)
    elif var == 'wind_uv':
        var_name = ['Eastward Near-Surface Wind', 'Northward Near-Surface Wind']
        var_label = ['uas', 'vas']
        freq = ['6hr', '6hr']
        project.append(projecti)
        version.append(versioni)
        ensemble.append(ensemblei)
    elif var == 'wind':
        var_name = ['Eastward Near-Surface Wind', 'Northward Near-Surface Wind', 'Near-Surface Wind Speed']
        var_label = ['uas', 'vas', 'sfcWind']
        freq = ['6hr', '6hr', '3hr']
        project.append(projecti)
        project.append(projecti)
        version.append(versioni)
        version.append(versioni)
        ensemble.append(ensemblei)
        ensemble.append(ensemblei)
        if bias_dict.get(model):
            var_name.append('Bias-Adjust Near-Surface Wind Speed')
            var_label.append('sfcWindAdjust')
            freq.append('3hr')
            project.append('CORDEX-Adjust')
            version.append('v1')
            if model == 8:
                ensemble.append('r12i1p1')
            else:
                ensemble.append(ensemblei)
    elif var == 'psl':
        var_name = ['Sea Level Pressure']
        var_label = ['psl']
        freq = ['3hr']
    elif var == 'rad':
        var_name = ['Surface Downwelling Shortwave Radiation']
        var_label = ['rsds']
        freq = ['3hr']
        if bias_dict.get(model):
            var_name.append('Bias-Adjusted Surface Downwelling Shortwave Radiation')
            var_label.append('rsdsAdjust')
            freq.append('3hr')
            project.append('CORDEX-Adjust')
            version.append('v1')
            if model == 8:
                ensemble.append('r12i1p1')
            else:
                ensemble.append(ensemblei)
    elif var == 'humr':
        var_name = ['Near-Surface Relative Humidity']
        var_label = ['hurs']
        freq = ['3hr']
    elif var == 'orog':
        var_name = ['Surface Altitude']
        var_label = ['orog']
        freq = ['fx']
        ensemble = ['r0i0p0']

    return institute, driving_model, ensemble, var_name, var_label, freq, project, version


def read_ww3hindcast_reg_grib(dir_hindcast):

    ds_hindcast = xr.open_mfdataset(dir_hindcast.glob('WW3_mediterr_*.grb2'), engine='cfgrib', parallel=True,
                                    preprocess=set_time)
    ds_hindcast = ds_hindcast.rename_vars(
        {'swh': 'hs', 'mwp': 'tm', 'perpw': 'tp', 'mwd': 'dirm', 'dirpw': 'dp', 'u': 'uw',
         'v': 'vw'})
    ds_hindcast = ds_hindcast.where((ds_hindcast.tp > 0) & (ds_hindcast.tp < 20))
    ds_hindcast = ds_hindcast.sel(time=~ds_hindcast.indexes['time'].duplicated()).drop('surface')

    return ds_hindcast


def read_ww3hindcast_reg_rotpole_6h(dir_hindcast=Path('/vhe/nasmetocean/wavewatch/wrf/hindcast_med_eur11_rotpole')):

    ds_hindcast = xr.open_mfdataset(dir_hindcast.glob('resampled6H_remapped_*.nc'), parallel=True,
                                    preprocess=set_datetimeindex_nc)
    ds_hindcast = ds_hindcast.rename_vars({'uas': 'uw', 'vas': 'vw'})

    return ds_hindcast


def read_ww3hindcast_point(dir_data, name_file, rename_vars=False):

    dicca = pd.read_csv(dir_data/ name_file, delim_whitespace=True, parse_dates=[[0, 1, 2, 3]],
                        index_col=0, names=['YY', 'mm', 'DD', 'time', 'hs', 'tm', 'tp', 'dirm', 'dp', 'spr', 'h',
                                              'lm', 'lp', 'uw', 'vw'])
    dicca.loc[dicca['tp'] < 0, 'tp'] = np.NaN
    dicca.loc[dicca.tp > 20, 'tp'] = np.NaN
    if rename_vars:
        dicca.rename(columns={'hs': 'Hs', 'tm': 'Tm', 'tp': 'Tp', 'dirm': 'Dirm'}, inplace=True)
    dicca.dropna(inplace=True)

    return dicca


def read_ww3cordex_point_old(dir_data, name_file, rename_vars=False):

    data_rcp85_ini = pd.read_csv(dir_data/ name_file, delim_whitespace=True, parse_dates=[[0, 1, 2, 3]], index_col=0)
    data_rcp85_ini = data_rcp85_ini[data_rcp85_ini.index.year < 2101]

    data_rcp85_ini['pw'] = 1026 * 9.81 ** 2 * data_rcp85_ini['hs'] ** 2 * data_rcp85_ini['t0m1'] / (64 * np.pi)
    data_rcp85_ini['tp'] = 1. / data_rcp85_ini['fp']

    data_rcp85_ini.rename(columns={'t0m1': 'tm01'}, inplace=True)
    if rename_vars:
        data_rcp85_ini.rename(columns={'hs': 'Hm0', 'tm01': 'Tm01', 'tp': 'Tp', 'dir': 'DirM'}, inplace=True)
    data_rcp85_ini.replace(np.inf, np.NaN, inplace=True)
    data_rcp85_ini.dropna(inplace=True)

    return data_rcp85_ini


def read_ww3cordex_point(dir_data, name_file):
    data_rcp85_ini = pd.read_csv(dir_data/ name_file, index_col=0, parse_dates=True)

    data_rcp85_ini['tp'] = 1. / data_rcp85_ini['fp']
    data_rcp85_ini.rename(columns={'t0m1': 'tm', 'dir': 'dirm'}, inplace=True)

    data_rcp85_ini.replace(np.inf, np.NaN, inplace=True)
    data_rcp85_ini.dropna(inplace=True)

    return data_rcp85_ini


def set_datetimeindex(data_nc, data_nc_df):

    if isinstance(data_nc_df.index, xr.CFTimeIndex) and \
            (data_nc_df.index.date_type == cftime.Datetime360Day):
        dtindex, valid_pos = to_datetimeindex(data_nc.time)
        data_nc_df = data_nc_df.iloc[valid_pos, :]
        data_nc_df.index = dtindex
    elif isinstance(data_nc_df.index, xr.CFTimeIndex):
        data_nc_df.index = data_nc_df.index.to_datetimeindex()

    return data_nc_df


def set_datetimeindex_nc(ds):

    ds_dtindex, valid_pos = to_datetimeindex(ds.time)
    ds = ds.isel(time=valid_pos)
    ds['time'] = ds_dtindex

    return ds


def output_name_prefix(f, prefix):
    return f"{f.resolve().parent}/{prefix}_{f.stem}{f.suffix}"


def get_rcoords(dataset, lat_bnds, lon_bnds):
    rlat_bnds, rlon_bnds = list(), list()
    for lat_bnd, lon_bnd in zip(lat_bnds, lon_bnds):
        rlat_bnd, rlon_bnd = get_rcoords_point(dataset, lat_bnd, lon_bnd)
        rlat_bnds.append(rlat_bnd)
        rlon_bnds.append(rlon_bnd)

    return np.asarray(rlat_bnds), np.asarray(rlon_bnds)


def get_rcoords_point(dataset, lat_point, lon_point):
    latmin = np.abs(dataset.lat - lat_point)
    rlat_point = latmin.where(latmin == latmin.min(), drop=True).squeeze().rlat.data

    lonmin = np.abs(dataset.lon - lon_point)
    rlon_point = lonmin.where(lonmin == lonmin.min(), drop=True).squeeze().rlon.data

    return rlat_point, rlon_point


def set_time(ds):
    index = ds.time + ds.step
    ds = ds.assign_coords(time=index)
    ds = ds.swap_dims({'step': 'time'})
    ds = ds.drop_vars('step')
    if pd.to_datetime(ds.time.data[0]).is_month_start and pd.to_datetime(ds.time.data[-1]).is_month_start:
        ds = ds.isel(time=slice(None, -1))

    return ds


def get_cordex_ds(model, run='rcp85', dir_base_cordex=Path('/vhe/nasmetocean/wavewatch/cordex')):
    info_model = info_models(model)
    dir_cordex_rcp = dir_base_cordex / info_model[0] / 'wave' / run

    ds_cordex_rcp = xr.open_mfdataset(dir_cordex_rcp.glob(f'WW3_{info_model[-1]}_*.nc'), parallel=True)
    ds_cordex_rcp = ds_cordex_rcp.sel(time=~ds_cordex_rcp.indexes['time'].duplicated())
    ds_cordex_rcp['tp'] = 1 / ds_cordex_rcp['fp']
    ds_cordex_rcp = ds_cordex_rcp.where((ds_cordex_rcp.tp > 0) & (ds_cordex_rcp.tp < 20))

    """
    Energy
    """
    rho = 1025
    g = 9.81
    ds_cordex_rcp['wave_power'] = ((rho * g ** 2) / (64 * np.pi)) * \
                                  ds_cordex_rcp['hs'] ** 2 * ds_cordex_rcp['tp'] / 1000  # kW/m

    rho_air = 1.25
    ds_cordex_rcp['wind_energy'] = 1 / 2 * rho_air * \
                                   xr.ufuncs.sqrt(
                                       ds_cordex_rcp['uwnd'] ** 2 + ds_cordex_rcp['vwnd'] ** 2) ** 3 / 1000  # kW/m2

    return info_model[2], ds_cordex_rcp
