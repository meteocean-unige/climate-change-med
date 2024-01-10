from pathlib import Path
import xarray as xr
import numpy as np


dir_base_cordex = Path('/vhe/nasmetocean/wavewatch/cordex')


dir_gcm = {1: ['EUR-11_CLMcom-CCCma-CanESM2_r1i1p1-CCLM4-8-17_v1_6hr',
               'EUR-11_CCCma-CanESM2_', '_r1i1p1_CLMcom-CCLM4-8-17_v1_6hr'],
           2: ['EUR-11_CLMcom-MIROC-MIROC5_r1i1p1-CCLM4-8-17_v1_6hr',
               'EUR-11_MIROC-MIROC5_', '_r1i1p1_CLMcom-CCLM4-8-17_v1_6hr'],
           3: ['EUR-11_SMHI-MPI-M-MPI-ESM-LR_r1i1p1-RCA4_v1a_6hr',
               'EUR-11_MPI-M-MPI-ESM-LR_', '_r1i1p1_SMHI-RCA4_v1a_6hr'],
           4: ['EUR-11_SMHI-NCC-NorESM1-M_r1i1p1-RCA4_v1_6hr',
               'EUR-11_NCC-NorESM1-M_', '_r1i1p1_SMHI-RCA4_v1_6hr'],
           5: ['EUR-11_SMHI-CNRM-CERFACS-CNRM-CM5_r1i1p1-RCA4_v1_6hr',
               'EUR-11_CNRM-CERFACS-CNRM-CM5_', '_r1i1p1_SMHI-RCA4_v1_6hr'],
           6: ['EUR-11_SMHI-IPSL-IPSL-CM5A-MR_r1i1p1-RCA4_v1_6hr',
               'EUR-11_IPSL-IPSL-CM5A-MR_', '_r1i1p1_SMHI-RCA4_v1_6hr'],
           7: ['EUR-11_SMHI-MOHC-HadGEM2-ES_r1i1p1-RCA4_v1_6hr',
               'EUR-11_MOHC-HadGEM2-ES_', '_r1i1p1_SMHI-RCA4_v1_6hr'],
           8: ['EUR-11_SMHI-ICHEC-EC-EARTH_r1i1p1-RCA4_v1_6hr',
               'EUR-11_ICHEC-EC-EARTH_', '_r1i1p1_SMHI-RCA4_v1_6hr'],
           9: ['EUR-11_DMI-ICHEC-EC-EARTH_r1i1p1-HIRHAM5_v1_6hr',
               'EUR-11_ICHEC-EC-EARTH_', '_r1i1p1_DMI-HIRHAM5_v1_6hr'],
           10: ['EUR-11_DMI-NCC-NorESM1-M_r1i1p1-HIRHAM5_v3_6hr',
                'EUR-11_NCC-NorESM1-M_', '_r1i1p1_DMI-HIRHAM5_v3_6hr'],
           11: ['EUR-11_DMI-MOHC-HadGEM2-ES_r1i1p1-HIRHAM5_v2_6hr',
                'EUR-11_MOHC-HadGEM2-ES_', '_r1i1p1_DMI-HIRHAM5_v2_6hr'],
           12: ['EUR-11_DMI-MPI-M-MPI-ESM-LR_r1i1p1-HIRHAM5_v1_6hr',
                'EUR-11_MPI-M-MPI-ESM-LR_', '_r1i1p1_DMI-HIRHAM5_v1_6hr'],
           13: ['EUR-11_DMI-CNRM-CERFACS-CNRM-CM5_r1i1p1-HIRHAM5_v2_6hr',
                'EUR-11_CNRM-CERFACS-CNRM-CM5_', '_r1i1p1_DMI-HIRHAM5_v2_6hr'],
           14: ['EUR-11_DMI-IPSL-IPSL-CM5A-MR_r1i1p1-HIRHAM5_v1_6hr',
                'EUR-11_IPSL-IPSL-CM5A-MR_', '_r1i1p1_DMI-HIRHAM5_v1_6hr'],
           15: ['EUR-11_CLMcom-ETH-ICHEC-EC-EARTH_r1i1p1-COSMO-crCLIM-v1-1_v1_6hr',
                'EUR-11_ICHEC-EC-EARTH_', '_r1i1p1_CLMcom-ETH-COSMO-crCLIM-v1-1_v1_6hr'],
           16: ['EUR-11_CLMcom-ETH-NCC-NorESM1-M_r1i1p1-COSMO-crCLIM-v1-1_v1_6hr',
                'EUR-11_NCC-NorESM1-M_', '_r1i1p1_CLMcom-ETH-COSMO-crCLIM-v1-1_v1_6hr'],
           17: ['EUR-11_CLMcom-ETH-MOHC-HadGEM2-ES_r1i1p1-COSMO-crCLIM-v1-1_v1_6hr',
                'EUR-11_MOHC-HadGEM2-ES_', '_r1i1p1_CLMcom-ETH-COSMO-crCLIM-v1-1_v1_6hr'],
           18: ['EUR-11_KNMI-CNRM-CERFACS-CNRM-CM5_r1i1p1-RACMO22E_v2_6hr',
                'EUR-11_CNRM-CERFACS-CNRM-CM5_', '_r1i1p1_KNMI-RACMO22E_v2_6hr'],
           19: ['EUR-11_KNMI-IPSL-IPSL-CM5A-MR_r1i1p1-RACMO22E_v1_6hr',
                'EUR-11_IPSL-IPSL-CM5A-MR_', '_r1i1p1_KNMI-RACMO22E_v1_6hr'],
           20: ['EUR-11_KNMI-MOHC-HadGEM2-ES_r1i1p1-RACMO22E_v2_6hr',
                'EUR-11_MOHC-HadGEM2-ES_', '_r1i1p1_KNMI-RACMO22E_v2_6hr']
          }

info_per = {'historical':  [1979, 2005], 'rcp85_mid': [2034, 2060], 'rcp85_end': [2074, 2100]}


def get_ds_cordex(model, case='historical'):
    dir_data = dir_base_cordex / dir_gcm[model][0] / 'wind' / case
    fn = f'{dir_gcm[model][1]}{case}{dir_gcm[model][2]}_*.nc'
    
    ds = xr.open_mfdataset(dir_data.glob(f'uas_{fn}'), parallel=True)
    ds['vas'] = xr.open_mfdataset(dir_data.glob(f'vas_{fn}'), parallel=True).vas
    
    ds = ds.sel(rlon=slice(-10, 2), rlat=slice(-15, -2))

    ds['uw'] = np.sqrt(ds['uas'] ** 2 + ds['vas'] ** 2)
    rho = 1.25
    ds['wind_energy'] = 1 / 2 * rho * np.sqrt(ds['uas'] ** 2 + ds['vas'] ** 2) ** 3 / 1000  # kW/m2
    ds = ds.drop(['uas', 'vas'])#, 'lon_vertices', 'lat_vertices'])

    if case == 'historical':
        period = info_per[case]
        ds_out = ds.sel(time=slice(f'{period[0]}-01-01', f'{period[1]}-12-31'))
    if case == 'rcp85':
        period = info_per[f'{case}_mid']
        ds_mid = ds.sel(time=slice(f'{period[0]}-01-01', f'{period[1]}-12-31'))
        period = info_per[f'{case}_end']
        ds_end = ds.sel(time=slice(f'{period[0]}-01-01', f'{period[1]}-12-31'))
        ds_out = ds_mid, ds_end

    return ds_out
    
