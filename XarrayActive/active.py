__author__    = "Daniel Westwood"
__contact__   = "daniel.westwood@stfc.ac.uk"
__copyright__ = "Copyright 2023 United Kingdom Research and Innovation"

# To be added to PyActiveStorage as xarray_active.py

import dask.array as da
from dask.array.reductions import mean_agg
import numpy as np

from xarray.core.dataset import Dataset
from xarray.core.dataarray import DataArray

def block_active_mean(arr, *args, **kwargs):
    if hasattr(arr,'active_mean'):
        return arr.active_mean(*args, **kwargs)
    else:
        # Here's where barebones Xarray might fall over - may need a non-CFA custom class.
        raise NotImplementedError

class DaskActiveArray(da.Array):

    description = "Dask Array Wrapper enabling the use of Active Storage."

    @property
    def is_active(self):
        return True

    def copy(self):
        """
        Create a new DaskActiveArray instance with all the same parameters as the current instance.
        """
        return DaskActiveArray(self.dask, self.name, self.chunks, meta=self)
    
    def __getitem__(self, index):
        """
        Perform indexing for this ActiveArray. May need to overwrite further if it turns out
        the indexing is performed **after** the dask `getter` method (i.e if retrieval and indexing
        are separate items on the dask graph). If this is the case, will need another `from_delayed`
        and `concatenation` method as used in ``active_mean``.
        """
        arr = super().__getitem__(index)
        return DaskActiveArray(arr.dask, arr.name, arr.chunks, meta=arr)

    def active_mean(self, axis=None, skipna=None):
        """
        Perform ``dask delayed`` active mean for each ``dask block`` which corresponds to a single ``chunk``.
        Combines the results of the dask delayed ``active_mean`` operations on each block into a single dask Array,
        which is then mapped to a new DaskActiveArray object.

        :param axis:        (int) The index of the axis on which to perform the active mean.

        :param skipna:      (bool) Skip NaN values when calculating the mean.

        :returns:       A new ``DaskActiveArray`` object which has been reduced along the specified axes using
                        the concatenations of active_means from each chunk.
        """

        newarr = da.reduction(
            self,
            block_active_mean,
            mean_agg,
            axis=axis,
            dtype=self.dtype
        )

        return DaskActiveArray(newarr.dask, newarr.name, newarr.chunks, meta=newarr)

class ActiveDataArray(DataArray):
    # No additional properties
    __slots__ = ()

    def mean(
        self,
        dim,
        *,
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

from xarray.backends import StoreBackendEntrypoint, BackendEntrypoint
from xarray.backends.common import AbstractDataStore
from xarray.core.dataset import Dataset
from xarray import conventions

from xarray.backends import ( 
    NetCDF4DataStore
)

def open_active_dataset(
        filename_or_obj,
        drop_variables=None,
        mask_and_scale=None,
        decode_times=None,
        concat_characters=None,
        decode_coords=None,
        use_cftime=None,
        decode_timedelta=None,
        group=None,
        ):
    """
    Top-level function which opens a NetCDF dataset using XarrayActive classes, overriding
    normal Xarray routines. Creates a ``NetCDF4DataStore`` (for now) 
    from the ``filename_or_obj`` provided, then passes this to a StoreBackendEntrypoint
    to create an Xarray Dataset. 

    :returns:       An ActiveDataset object composed of ActiveDataArray objects representing the different
                    NetCDF variables and dimensions. Non-active 
    """

    # Load the normal datastore from the provided file (object not supported).
    store = NetCDF4DataStore.open(filename_or_obj, group=group)

    # Xarray makes use of StoreBackendEntrypoints to provide the Dataset 'ds'
    store_entrypoint = ActiveStoreBackendEntrypoint()
    ds = store_entrypoint.open_dataset(
        store,
        mask_and_scale=mask_and_scale,
        decode_times=decode_times,
        concat_characters=concat_characters,
        decode_coords=decode_coords,
        drop_variables=drop_variables,
        use_cftime=use_cftime,
        decode_timedelta=decode_timedelta,
    )

    return ds

class ActiveBackendEntrypoint(BackendEntrypoint):

    description = "Open NetCDF4 files with Active storage in mind - engine entrypoint"
    url = "https://cedadev.github.io/XarrayActive/"

    def open_dataset(
            self,
            filename_or_obj,
            *,
            drop_variables=None,
            mask_and_scale=None,
            decode_times=None,
            concat_characters=None,
            decode_coords=None,
            use_cftime=None,
            decode_timedelta=None,
            group=None,
            # backend specific keyword arguments
            # do not use 'chunks' or 'cache' here
        ):
        """
        Returns a complete xarray representation of a NetCDF dataset which has the infrastructure 
        to enable Active methods.
        """

        return open_active_dataset(
            filename_or_obj, 
            drop_variables=drop_variables,
            mask_and_scale=mask_and_scale,
            decode_times=decode_times,
            concat_characters=concat_characters,
            decode_coords=decode_coords,
            use_cftime=use_cftime,
            decode_timedelta=decode_timedelta,
            group=group)

class ActiveStoreBackendEntrypoint(StoreBackendEntrypoint):
    description = "Open Active-enabled dataset"

    def open_dataset(
        self,
        store,
        *,
        mask_and_scale=True,
        decode_times=True,
        concat_characters=True,
        decode_coords=True,
        drop_variables=None,
        use_cftime=None,
        decode_timedelta=None,
    ) -> Dataset:
        """
        Takes store of type AbstractDataStore and creates an ActiveDataset instance.

        :returns:           An ActiveDataset instance composed of ActiveDataArray instances representing the different
                            NetCDF variables and dimensions.
        """
        assert isinstance(store, AbstractDataStore)

        # Same as NetCDF4 operations, just with the CFA Datastore
        vars, attrs = store.load()
        encoding    = store.get_encoding()

        # Ensures variables/attributes comply with CF conventions.
        vars, attrs, coord_names = conventions.decode_cf_variables(
            vars,
            attrs,
            mask_and_scale=mask_and_scale,
            decode_times=decode_times,
            concat_characters=concat_characters,
            decode_coords=decode_coords,
            drop_variables=drop_variables,
            use_cftime=use_cftime,
            decode_timedelta=decode_timedelta,
        )

        ds = ActiveDataset(vars, attrs=attrs)
        ds = ds.set_coords(coord_names.intersection(vars))
        ds.set_close(store.close)
        ds.encoding = encoding

        return ds