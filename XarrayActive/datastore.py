from xarray.backends import NetCDF4DataStore
from xarray.backends.common import (
    BackendArray,
    robust_getitem
)
from xarray.coding.variables import pop_to
from xarray.coding.strings import create_vlen_dtype

from xarray.core import indexing
from xarray.core.variable import Variable

from dask.utils import SerializableLock
from dask.array.core import getter
from dask.base import tokenize

from contextlib import suppress
import functools
import operator

import numpy as np

from .active_dask import DaskActiveArray

class ActiveDataStore(NetCDF4DataStore):
    def open_store_variable(self, name: str, var):
        import netCDF4

        dimensions = var.dimensions
        attributes = {k: var.getncattr(k) for k in var.ncattrs()}
        data       = indexing.LazilyIndexedArray(ActiveSubarrayWrapper(name, self))
        encoding   = {}

        if isinstance(var.datatype, netCDF4.EnumType):
            encoding["dtype"] = np.dtype(
                data.dtype,
                metadata={
                    "enum": var.datatype.enum_dict,
                    "enum_name": var.datatype.name,
                },
            )
        else:
            encoding["dtype"] = var.dtype

        if data.dtype.kind == "S" and "_FillValue" in attributes:
            attributes["_FillValue"] = np.bytes_(attributes["_FillValue"])

        # netCDF4 specific encoding; save _FillValue for later
        filters = var.filters()
        if filters is not None:
            encoding.update(filters)
        chunking = var.chunking()
        if chunking is not None:
            if chunking == "contiguous":
                encoding["contiguous"] = True
                encoding["chunksizes"] = None
            else:
                encoding["contiguous"] = False
                encoding["chunksizes"] = tuple(chunking)
                encoding["preferred_chunks"] = dict(zip(var.dimensions, chunking))
        # TODO: figure out how to round-trip "endian-ness" without raising
        # warnings from netCDF4
        # encoding['endian'] = var.endian()
        pop_to(attributes, encoding, "least_significant_digit")
        # save source so __repr__ can detect if it's local or not
        encoding["source"] = self._filename
        encoding["original_shape"] = data.shape

        return Variable(dimensions, data, attributes, encoding)
    
class ActiveSubarrayWrapper(BackendArray, SuperLazyArrayLike):

    def __init__(self, variable_name, datastore, chunks=None, extent=None):
        self.datastore     = datastore
        self.variable_name = variable_name

        self._chunks = chunks
        self._extent = extent
        self._lock = SerializableLock()

        self._variable = self._get_variable()
        self.shape = self._variable.shape
        self.ndim  = len(self.shape)

        dtype = self._variable.dtype
        if dtype is str:
            # use object dtype (with additional vlen string metadata) because that's
            # the only way in numpy to represent variable length strings and to
            # check vlen string dtype in further steps
            # it also prevents automatic string concatenation via
            # conventions.decode_cf_variable
            dtype = create_vlen_dtype(str)
        self.dtype = dtype

        self.__array_function__ = self.get_array
                
    def _get_variable(self, needs_lock=True):
        ds = self.datastore._acquire(needs_lock)
        variable = ds.variables[self.variable_name]
        variable.set_auto_maskandscale(False)
        # only added in netCDF4-python v1.2.8
        with suppress(AttributeError):
            variable.set_auto_chartostring(False)
        return variable

    def __array__(self):

        if not self._chunks:
            # get_array should just get the whole array if that's what we're trying to do.
            # indexing should just be added to the instance of this class, and then the
            # built-in mean from _ActiveFragment should take care of things.
            return self._variable
        

        # for every dask chunk return a smaller object with the right extent.
        # Create a chunk_shape tuple from chunks and _variable (figure out which chunk and which axis, divide etc.)
        # Define a subarray for each chunk, with appropriate index.

        chunks = None # Need to find out what this needs to be.

        name = (f"{self.__class__.__name__}-{tokenize(self)}",)
        dsk = {}
        for pos in positions:
            
            subarray = ArrayPartition(
                filename,
                address,
                dtype=,
                shape=,
                position=pos,
            )

            key = f"{subarray.__class__.__name__}-{tokenize(subarray)}"
            dsk[key] = subarray
            dsk[name + f_index] = (
                getter, # Dask default should be enough with the new indexing routine.
                key,
                extent,
                False,
                getattr(subarray,"_lock",False)
            )

        return DaskActiveArray(dsk, name, chunks=chunks, dtype=self.dtype)


    def _getitem(self, key):
        if self.datastore.is_remote:  # pragma: no cover
            getitem = functools.partial(robust_getitem, catch=RuntimeError)
        else:
            getitem = operator.getitem

        try:
            with self.datastore.lock:
                original_array = self.get_array(needs_lock=False)
                array = getitem(original_array, key)
        except IndexError:
            # Catch IndexError in netCDF4 and return a more informative
            # error message.  This is most often called when an unsorted
            # indexer is used before the data is loaded from disk.
            msg = (
                "The indexing operation you are attempting to perform "
                "is not valid on netCDF4.Variable object. Try loading "
                "your data into memory first by calling .load()."
            )
            raise IndexError(msg)
        return array
