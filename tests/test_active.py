# All routines for testing CFA general methods.
import xarray as xr
import numpy as np

def test_active():

    path_to_active = f'tests/rain_test.nc'

    try:
        ds = xr.open_dataset(
            path_to_active, 
            engine='Active',
            active_options={})#{'chunks':{'time':2}})
    except Exception as err:
        assert isinstance(err, NotImplementedError)

    ds = xr.open_dataset(
            path_to_active, 
            engine='Active',
            active_options={'chunks':{'time':2}})

    assert 'p' in ds
    assert ds['p'].shape == (20, 180, 360)

    p_sel = ds['p']#.isel(time=slice(0,3),latitude=slice(140,145), longitude=slice(90,100))

    #assert p_sel.shape == (3, 5, 10)

    p_value = p_sel.mean()

    assert p_value.shape == ()
    assert (p_value.to_numpy() - 0.53279) < 0.01

def test_active_recursive():

    path_to_active = f'tests/rain_test.nc'

    try:
        ds = xr.open_dataset(
            path_to_active, 
            engine='Active',
            active_options={})#{'chunks':{'time':2}})
    except Exception as err:
        assert isinstance(err, NotImplementedError)

    ds = xr.open_dataset(
            path_to_active, 
            engine='Active',
            active_options={'chunks':{'time':2}})

    assert 'p' in ds
    assert ds['p'].shape == (20, 180, 360)

    p_sel = ds['p'].isel(time=slice(0,3),latitude=slice(140,145), longitude=slice(90,100))

    assert p_sel.shape == (3, 5, 10)

    p_mean = p_sel.mean(dim='time')

    assert p_mean.shape == (5, 10)
    assert (p_mean[0][0].to_numpy() - 0.683402) < 0.01

def test_active_methods():

    path_to_active = f'tests/rain_test.nc'

    try:
        ds = xr.open_dataset(
            path_to_active, 
            engine='Active',
            active_options={})#{'chunks':{'time':2}})
    except Exception as err:
        assert isinstance(err, NotImplementedError)

    ds = xr.open_dataset(
            path_to_active, 
            engine='Active',
            active_options={'chunks':{'time':2}})

    assert 'p' in ds
    assert ds['p'].shape == (20, 180, 360)

    p_sel = ds['p'].isel(time=slice(0,3),latitude=slice(140,145), longitude=slice(90,100))

    assert p_sel.shape == (3, 5, 10)

    p_value = p_sel.isel().max()
    assert p_value.shape == ()
    assert (p_value.to_numpy() - 0.9978273) < 0.01

    p_value = p_sel.isel().min()
    assert p_value.shape == ()
    assert (p_value.to_numpy() - 0.0014456) < 0.01

    p_value = p_sel.isel().sum()
    assert p_value.shape == ()
    assert (p_value.to_numpy() - 76.7931739) < 0.01

if __name__ == '__main__':
    test_active()
    test_active_recursive()
    test_active_methods()