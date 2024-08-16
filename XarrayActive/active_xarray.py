__author__    = "Daniel Westwood"
__contact__   = "daniel.westwood@stfc.ac.uk"
__copyright__ = "Copyright 2023 United Kingdom Research and Innovation"

import numpy as np

from xarray.core.dataset import Dataset
from xarray.core.dataarray import DataArray

from .active_dask import DaskActiveArray

class ActiveDataArray(DataArray):
    # No additional properties
    __slots__ = ()

    def mean(
        self,
        *,
        dim = None,
        skipna = None,
        keep_attrs = None,
        **kwargs,
    ):
        """
        Reduce this DataArray's data by applying ``mean`` along some dimension(s).

        Parameters
        ----------
        dim : str, Iterable of Hashable, "..." or None, default: None
            Name of dimension[s] along which to apply ``mean``. For e.g. ``dim="x"``
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
            function for calculating ``mean`` on this object's data.
            These could include dask-specific kwargs like ``split_every``.

        Returns
        -------
        reduced : DataArray
            New DataArray with ``mean`` applied to its data and the
            indicated dimension(s) removed

        """
        return self.reduce(
            dataarray_active_mean, # from duck_array_ops.mean
            dim=dim,
            skipna=skipna,
            keep_attrs=keep_attrs,
            **kwargs,
        )
    
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
    
def dataarray_active_mean(array: DaskActiveArray, axis=None, skipna=None, **kwargs):
    """
    Function provided to dask reduction, activates the ``active_mean`` method of the ``DaskActiveArray``.

    :param array:       (obj) A DaskActiveArray object which has additional methods enabling Active operations.

    :param axis:        (int) The axis over which to perform the active_mean operation.

    :param skipna:      (bool) Skip NaN values when calculating the mean.

    :returns:       The result from performing the ``DaskActiveArray.active_mean`` method, which gives a new
                    ``DaskActiveArray`` object.
    """
    from xarray.core import duck_array_ops
    try:
        return array.active_mean(axis, skipna=skipna, **kwargs)
    except AttributeError:
        print("ActiveWarning: Unable to compute active mean - array has already been loaded.")
        print("NetCDF file size may prohibit lazy loading and thus Active methods.")
        return duck_array_ops.mean(array, axis=axis, skipna=skipna, **kwargs)
