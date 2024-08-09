from .partition import ArrayPartition, ArrayLike
from .active_chunk import (
    ActiveChunk, 
    ActiveOptionsContainer, 
    _determine_chunk_space
)

from .active_dask import DaskActiveArray

from dask.array.core import getter
from dask.base import tokenize

from itertools import product

class ActiveChunkWrapper(ArrayPartition):
    """
    Combines ActiveChunk - active methods, and ArrayPartition - array methods
    into a single ChunkWrapper class. 
    """
    def copy(self):
        return ActiveChunkWrapper(
            self.filename,
            self.address,
            **self.get_kwargs()
        )

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
            units,
            dtype,
            named_dims,
            active_options={},
        ):

        self._variable   = var

        self.filename    = filename
        self.name        = var.name
        self.active_options = active_options

        self.chunk_space = _determine_chunk_space(
            self._active_chunks, 
            shape, 
            named_dims,
            chunk_limits=self._chunk_limits)

        super().__init__(shape, units=units, dtype=dtype)
                
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

            # for every dask chunk return a smaller object with the right extent.
            # Create a chunk_shape tuple from chunks and _variable (figure out which chunk and which axis, divide etc.)
            # Define a subarray for each chunk, with appropriate index.

            array_name = (f"{self.__class__.__name__}-{tokenize(self)}",)
            dsk = {}
            for position in self.get_chunk_positions():
                position = tuple(position)
            
                extent   = self.get_chunk_extent(position)
                cformat  = None
                
                chunk = ActiveChunkWrapper(
                    self.filename,
                    self.name,
                    dtype=self.dtype,
                    units=self.units,
                    shape=self.shape,
                    position=position,
                    extent=extent,
                    format=cformat
                )

                c_identifier = f"{chunk.__class__.__name__}-{tokenize(chunk)}"
                dsk[c_identifier] = chunk
                dsk[array_name + position] = (
                    getter, # Dask default should be enough with the new indexing routine.
                    c_identifier,
                    chunk.get_extent(),
                    False,
                    getattr(chunk,"_lock",False)
                )

            return DaskActiveArray(dsk, array_name[0], chunks=self.get_dask_chunks(), dtype=self.dtype)

    def get_chunk_positions(self):
        origin = [0 for i  in range(self.ndim)]

        positions = [
            coord for coord in product(
                *[range(r[0], r[1]) for r in zip(origin, self.chunk_space)]
            )
        ]

        return positions
    
    def get_chunk_extent(self, position):
        extent = []
        for dim in self.ndim:
            pos_index   = position[dim]
            shape_size = self.shape[dim]
            space_size = self.chunk_space[dim]

            conversion = shape_size/space_size

            ext = slice(
                pos_index*conversion, (pos_index+1)*conversion
            )
            extent.append(ext)
        return extent

    def get_dask_chunks(self, explicit_shapes=None):
        """
        Define the `chunks` array passed to Dask when creating a Dask Array. This is an array of fragment sizes 
        per dimension for each of the relevant dimensions. Copied from cf-python version 3.14.0 onwards.

        Explicit shapes copied from cf-python but not implemented in the wider class.

        :returns:       A tuple of the chunk sizes along each dimension.
        """
                
        from numbers import Number
        from dask.array.core import normalize_chunks

        extent = self.get_chunk_extent([0 for dim in self.ndim])
        ndim   = len(self.shape)
        csizes_per_dim, chunked_dim_indices = [],[]

        for dim, n_chunks in enumerate(self.chunk_space):
            if n_chunks != 1:

                csizes = []
                index = [0] * ndim
                for n in range(n_chunks):
                    index[dim] = n
                    ext = extent[tuple(index)][dim]
                    chunk_size = ext.stop - ext.start
                    csizes.append(chunk_size)

                csizes_per_dim.append(tuple(csizes))
                chunked_dim_indices.append(dim)
            else:
                # This aggregated dimension is spanned by exactly one
                # fragment. Store None, for now, in the expectation
                # that it will get overwritten.
                csizes_per_dim.append(None)

        ## Handle explicit shapes for the fragments.

        if isinstance(explicit_shapes, (str, Number)) or explicit_shapes is None:
            csizes_per_dim = [
                cs if i in chunked_dim_indices else explicit_shapes for i, cs in enumerate(csizes_per_dim)
            ]
        elif isinstance(explicit_shapes, dict):
            csizes_per_dim = [
                csizes_per_dim[i] if i in chunked_dim_indices else explicit_shapes.get(i, "auto")
                for i, cs in enumerate(csizes_per_dim)
            ]
        else:
            # explicit_shapes is a sequence
            if len(explicit_shapes) != ndim:
                raise ValueError(
                    f"Wrong number of 'explicit_shapes' elements in {explicit_shapes}: "
                    f"Got {len(explicit_shapes)}, expected {ndim}"
                )

            csizes_per_dim = [
                cs if i in chunked_dim_indices else explicit_shapes[i] for i, cs in enumerate(csizes_per_dim)
            ]

        return normalize_chunks(csizes_per_dim, shape=self.shape, dtype=self.dtype)

