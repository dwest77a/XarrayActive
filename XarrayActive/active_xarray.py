__author__    = "Daniel Westwood"
__contact__   = "daniel.westwood@stfc.ac.uk"
__copyright__ = "Copyright 2023 United Kingdom Research and Innovation"

import numpy as np

from xarray.core.dataset import Dataset
from xarray.core.dataarray import DataArray

from .active_dask import DaskActiveArray
from xarray.core import duck_array_ops

class ActiveDataArray(DataArray):
    # No additional properties
    __slots__ = ()

    # Override Xarray DataArray standard functions in favour of Active enabled ones.
    def mean(self, *args,**kwargs):
        return self._active_op(dataarray_active_mean, *args, **kwargs)
    
    def max(self, *args,**kwargs):
        return self._active_op(dataarray_active_max, *args, **kwargs)
    
    def min(self, *args,**kwargs):
        return self._active_op(dataarray_active_min, *args, **kwargs)
    
    def sum(self, *args,**kwargs):
        return self._active_op(dataarray_active_sum, *args, **kwargs)
    
    def _active_op(
        self,
        op = None,
        dim = None,
        *,
        skipna: bool | None = None,
        keep_attrs: bool | None = None,
        **kwargs,
    ):
        """
        Reduce this DataArray's data by applying an ``active`` operation along some dimension(s).

        Parameters
        ----------
        dim : str, Iterable of Hashable, "..." or None, default: None
            Name of dimension[s] along which to apply the operation`. For e.g. ``dim="x"``
            or ``dim=["x", "y"]``. If "..." or None, will reduce over all dimensions.
        skipna : bool or None, optional
            If True, skip missing values (as marked by NaN). By default, only
            skips missing values for float dtypes; other dtypes either do not
            have a sentinel missing value (int) or ``skipna=True`` has not been
            implemented (object, datetime64 or timedelta64).
        keep_attrs : bool or None, optional
            If True, ``attrs`` will be copied from the original
            object to the new one.  If False, the new object will be
            returned without attributes.
        **kwargs : Any
            Additional keyword arguments passed on to the appropriate array
            function for calculating the operation on this object's data.
            These could include dask-specific kwargs like ``split_every``.

        Returns
        -------
        reduced : DataArray
            New DataArray with reduction applied to its data and the
            indicated dimension(s) removed

        """
        return self.reduce(
            op,
            dim=dim,
            skipna=skipna,
            keep_attrs=keep_attrs,
            **kwargs)
        
class ActiveDataset(Dataset):

    # No additional properties
    __slots__ = ()

    def _construct_dataarray(self, name):
        """Construct a DataArray by indexing this dataset"""

        darr = super()._construct_dataarray(name)

        is_active_variable = True

        # Convert variable to DaskActiveArray if not already defined as that type.
        # CFAPyX - FragmentArrayWrapper returns a DaskActiveArray upon indexing.
        variable = darr.variable
        # If the active parts have been lost at this point.
        if not isinstance(variable.data, DaskActiveArray) and is_active_variable:
            variable.data = DaskActiveArray(
                variable.data.dask, 
                variable.data.name,
                variable.data.chunks,
                meta=variable.data
            )

        coords   = {k: v for k, v in zip(darr.coords.keys(), darr.coords.values())}
        name     = darr.name

        # Not ideal to break into the DataArray class but seems to be unavoidable (for now)
        indexes  = darr._indexes

        return ActiveDataArray(
            variable,
            coords,
            name=name,
            indexes=indexes,
            fastpath=True
        )
    
## DataArray methods to apply to the DaskActiveArray
def dataarray_active_mean(array, *args, **kwargs):
    return dataarray_active_method(array, 'mean', *args, **kwargs)

def dataarray_active_max(array, *args, **kwargs):
    return dataarray_active_method(array, 'max', *args, **kwargs)

def dataarray_active_min(array, *args, **kwargs):
    return dataarray_active_method(array, 'min', *args, **kwargs)

def dataarray_active_sum(array, *args, **kwargs):
    return dataarray_active_method(array, 'sum', *args, **kwargs)

def dataarray_active_method(array: DaskActiveArray, method: str, axis=None, skipna=None, **kwargs):
    """
    Function provided to dask reduction, activates the ``active`` methods of the ``DaskActiveArray``.

    :param array:       (obj) A DaskActiveArray object which has additional methods enabling Active operations.

    :param axis:        (int) The axis over which to perform the active_mean operation.

    :param skipna:      (bool) Skip NaN values when calculating the mean.

    :returns:       The result from performing the ``DaskActiveArray.active_mean`` method, which gives a new
                    ``DaskActiveArray`` object.
    """
    from xarray.core import duck_array_ops
    arr_methods = {
        'mean': array.active_mean,
        'max': array.active_max,
        'min': array.active_min,
        'sum': array.active_sum
    }

    # On failure of the Active method, can use Duck methods instead - normal behaviour.
    duck_methods = {
        'mean': duck_array_ops.mean,
        'max': duck_array_ops.max,
        'min': duck_array_ops.min,
        'sum': duck_array_ops.sum
    }

    from xarray.core import duck_array_ops
    try:
        return arr_methods[method](axis, skipna=skipna, **kwargs)
    except AttributeError:
        print("ActiveWarning: Unable to compute active mean - array has already been loaded.")
        print("NetCDF file size may prohibit lazy loading and thus Active methods.")
        return duck_methods[method](array, axis=axis, skipna=skipna, **kwargs)
