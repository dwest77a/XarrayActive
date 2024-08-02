import xarray as xr

loc = '/'.join(__file__.split('/')[:-2])


# Chunks only works if you have this as an installed backend engine.
ds = xr.open_dataset(
    '/badc/cmip6/data/CMIP6/ScenarioMIP/CNRM-CERFACS/CNRM-ESM2-1/ssp119/r1i1p1f2/3hr/huss/gr/v20190328/huss_3hr_CNRM-ESM2-1_ssp119_r1i1p1f2_gr_209501010300-210101010000.nc',
    engine='Active',
    chunks={'time':100},
)

ds = xr.open_dataset('/home/users/dwest77/Documents/CFAPyX/testfiles/raincube/example0_0_0.nc', group='/rain1/', engine='Active',chunks={'time':2})


p = ds['p'].sel(time=slice(1,3),latitude=slice(50,54), longitude=slice(0,9))
#p = ds['huss'].isel(time=slice(1,3),lat=slice(50,54), lon=slice(0,9))
pq = p.mean(dim='time')




x=1
pq.plot()