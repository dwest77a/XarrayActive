from xarray.backends import StoreBackendEntrypoint, BackendEntrypoint
from xarray.backends.common import AbstractDataStore
from xarray.core.dataset import Dataset
from xarray import conventions

from xarray.backends import ( 
    NetCDF4DataStore
)

from .active_xarray import ActiveDataset
from .datastore import ActiveDataStore

def open_active_dataset(
        filename_or_obj,
        drop_variables=None,
        mask_and_scale=None,
        decode_times=None,
        concat_characters=None,
        decode_coords=None,
        use_cftime=None,
        decode_timedelta=None,
        active_options={},
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
    store = ActiveDataStore.open(filename_or_obj, group=group)

    store.active_options = active_options

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
            active_options={},
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
            active_options=active_options,
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