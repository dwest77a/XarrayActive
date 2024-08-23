from xarray.backends import NetCDF4DataStore
from xarray.core.utils import FrozenDict
from xarray.coding.variables import pop_to

from xarray.core import indexing
from xarray.core.variable import Variable

from contextlib import suppress

import numpy as np

from .active_chunk import (
    ActiveOptionsContainer,
)

from .wrappers import ActiveArrayWrapper

class ActiveDataStore(NetCDF4DataStore, ActiveOptionsContainer):

    def get_variables(self):
        """
        Override normal store behaviour to allow opening some variables 'actively'
        """
        return FrozenDict(
            (k, self.open_variable(k, v)) for k, v in self.ds.variables.items()
        )
    
    def open_variable(self, name: str, var):
        """
        Allow opening some variables 'actively', if they are not a dimension (where
        you'll want the whole array anyway) and where the active chunks are specified
        - required by XarrayActive.
        """
        if name in self.ds.dimensions or not self._active_chunks:
            return self.open_store_variable(name, var)
        else:
            return self.open_active_variable(name, var)

    def open_active_variable(self, name: str, var):
        """
        Utilise the ActiveArrayWrapper builder to obtain the data
        Lazily for this variable so active methods can be applied later.
        """
        import netCDF4

        dimensions = var.dimensions

        units = ''
        if hasattr(var, 'units'):
            units = getattr(var, 'units')

        attributes = {k: var.getncattr(k) for k in var.ncattrs()}
        data       = indexing.LazilyIndexedArray(
            ActiveArrayWrapper(
                self._filename,
                var,
                var.shape,
                units,
                var.dtype,
                named_dims=dimensions,
                active_options=self.active_options
            )
        )
        
        # Everything after this point is normal store behaviour
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
