# XarrayActive
For use with the Xarray module as an additional backend. See the module[PyActiveStorage](https://github.com/NCAS-CMS/PyActiveStorage) for more details.

## Installation

```
pip install xarray==2024.6.0
pip install XarrayActive==2024.9.0
```

## Usage

```
import xarray as xr

ds = xr.open_dataset('any_file.nc', engine='Active')
# Plot data

```