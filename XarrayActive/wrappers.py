from arraypartition import (
    ArrayPartition, 
    ArrayLike,
    get_chunk_space,
    get_chunk_shape,
    get_chunk_positions,
    get_chunk_extent,
    get_dask_chunks,
    combine_slices
)
from .active_chunk import (
    ActiveOptionsContainer
)

from .active_dask import DaskActiveArray

from dask.array.core import getter
from dask.base import tokenize

from itertools import product

class ActivePartition(ArrayPartition):
    """
    Container for future ActivePartition behaviour, may not be required unless
    additional behaviour is required.
    """
    def copy(self, extent=None):

        kwargs = self.get_kwargs()
        if extent:
            kwargs['extent'] = combine_slices(self.shape, list(self.get_extent()), extent)
        ap = ActivePartition(
            self.filename,
            self.address,
            **kwargs
        )
        return ap

class ActiveArrayWrapper(ArrayLike, ActiveOptionsContainer):
    """
    ActiveArrayWrapper behaves like an Array that can be indexed or referenced to 
    return a Dask-like array object. This class is essentially a constructor for the 
    partitions that feed into the returned Dask-like array into Xarray.
    """
    def __init__(
            self, 
            filename,
            var, 
            shape,
            units=None,
            dtype=None,
            named_dims=None,
            active_options={},
        ):

        self._variable   = var

        self.filename    = filename
        self.name        = var.name
        self.active_options = active_options

        self.named_dims = named_dims

        super().__init__(shape, units=units, dtype=dtype)

        # Further work required to get this to work - 23/08/24

        #self._active_chunks = normalize_partition_chunks(
        #    self._active_chunks,
        #    self.shape,
        #    self.dtype,
        #    self.named_dims)

        self.chunk_shape = get_chunk_shape(
            self._active_chunks,
            self.shape,
            self.named_dims,
            chunk_limits=self._chunk_limits
        )

        self.chunk_space = get_chunk_space(
            self.chunk_shape,
            self.shape
        )

        self.__array_function__ = self.__array__
                
    def __getitem__(self, selection):
        """
        Non-lazy retrieval of the dask array when this object is indexed.
        """
        arr = self.__array__()
        return arr[selection]

    def __array__(self, *args, **kwargs):

        if not self._active_chunks:
            # get_array should just get the whole array if that's what we're trying to do.
            # indexing should just be added to the instance of this class, and then the
            # built-in mean from _ActiveFragment should take care of things.
            return self._variable
        else:

            # For every dask chunk return a smaller object with the right extent.
            # Create a chunk_shape tuple from chunks and _variable (figure out which chunk and which axis, divide etc.)
            # Define a subarray for each chunk, with appropriate index.

            array_name = (f"{self.__class__.__name__}-{tokenize(self)}",)
            dsk = {}
            positions = get_chunk_positions(self.chunk_space)
            request   = get_chunk_extent(positions[0], self.shape, self.chunk_space)

            global_extent = {}

            for position in positions:
                position = tuple(position)
            
                extent   = get_chunk_extent(position, self.shape, self.chunk_space)
                cformat  = None
                global_extent[position] = extent
                
                chunk = ActivePartition(
                    self.filename,
                    self.name,
                    dtype=self.dtype,
                    units=self.units,
                    shape=self.chunk_shape,
                    position=position,
                    extent=extent,
                    format=cformat
                )

                c_identifier = f"{chunk.__class__.__name__}-{tokenize(chunk)}"
                dsk[c_identifier] = chunk
                dsk[array_name + position] = (
                    getter, # Dask default should be enough with the new indexing routine.
                    c_identifier,
                    request,
                    False,
                    getattr(chunk,"_lock",False)
                )

            dask_chunks = get_dask_chunks(
                self.shape,
                self.chunk_space,
                extent=global_extent,
                dtype=self.dtype,
                explicit_shapes=None
            )

            return DaskActiveArray(dsk, array_name[0], chunks=dask_chunks, dtype=self.dtype)
