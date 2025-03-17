# XarrayActive

![Static Badge](https://img.shields.io/badge/Xarray%20Engine%20Component-1E4B23)
[![PyPI version](https://badge.fury.io/py/XarrayActive.svg)](https://pypi.python.org/pypi/XarrayActive/)

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
