import numpy as np
from itertools import product


class ActiveOptionsContainer:
    """
    Container for ActiveOptions properties.
    """
    @property
    def active_options(self):
        """
        Property of the datastore that relates private option variables to the standard 
        ``active_options`` parameter.
        """
        return {
            'chunks': self._active_chunks,
            'chunk_limits': self._chunk_limits,
        }
    
    @active_options.setter
    def active_options(self, value):
        self._set_active_options(**value)

    def _set_active_options(self, chunks={}, chunk_limits=True):

        if chunks == {}:
            raise NotImplementedError(
                'Default chunking is not implemented, please provide a chunk scheme '
                ' - active_options = {"chunks": {}}'
            )

        self._active_chunks = chunks
        self._chunk_limits = chunk_limits



# Holds all Active routines.
class ActiveChunk:

    description = "Container class for Active routines performed on each chunk. All active-per-chunk content can be found here."
    
    def _post_process_data(self, data):
        # Perform any post-processing steps on the data here
        return data

    def _standard_mean(self, axes=None, skipna=None, **kwargs):
        """
        Standard Mean routine matches the normal routine for dask, required at this
        stage if Active mean not available.
        """
        size = 1
        for i in axes:
            size *= self.shape[i]

        arr = np.array(self)
        if skipna:
            total = np.nanmean(arr, axis=axes, **kwargs) *size
        else:
            total = np.mean(arr, axis=axes, **kwargs) *size
        return {'n': self._numel(axes=axes), 'total': total}
    
    def _numel(self, axes=None):
        if not axes:
            return self.size
        
        size = 1
        for i in axes:
            size *= self.shape[i]
        newshape = list(self.shape)
        for ax in axes:
            newshape[ax] = 1

        return np.full(newshape, size)

    def active_mean(self, axis=None, skipna=None, **kwargs):
        """
        Use PyActiveStorage package functionality to perform mean of this Fragment.

        :param axis:        (int) The axes over which to perform the active_mean operation.

        :param skipna:      (bool) Skip NaN values when calculating the mean.

        :returns:       A ``duck array`` (numpy-like) with the reduced array or scalar value, 
                        as specified by the axes parameter.
        """
        try:
            from activestorage.active import Active
        except ImportError:
            # Unable to import Active package. Default to using normal mean.
            print("ActiveWarning: Unable to import active module - defaulting to standard method.")
            return self._standard_mean(axes=axis, skipna=skipna, **kwargs)
            
        active = Active(self.filename, self.address)
        active.method = "mean"
        extent = tuple(self.get_extent())
        data   = active[extent]

        if axis == None:
            axis = tuple([i for i in range(self.ndim)])

        n = self._numel(axes=axis)

        if len(axis) == self.ndim:

            t = self._post_process_data(data) * n

            r = {
                'n': n,
                'total': t
            }
            return r

        # Experimental Recursive requesting to get each 1D column along the axes being requested.
        range_recursives = []
        for dim in range(self.ndim):
            if dim not in axis:
                range_recursives.append(range(extent[dim].start, extent[dim].stop))
            else:
                range_recursives.append(extent[dim])
        results = np.array(self._get_elements(active, range_recursives, hyperslab=[]))

        t = self._post_process_data(results) * n
        return {
            'n': n,
            'total': t
        }

    def _get_elements(self, active, recursives, hyperslab=[]):
        dimarray = []
        if not len(recursives) > 0:

            # Perform active slicing and meaning here.
            return active[tuple(hyperslab)].flatten()[0]
        
        current = recursives[0]

        if type(current) == slice:
            newslab = hyperslab + [current]
            dimarray.append(self._get_elements(active, recursives[1:], hyperslab=newslab))

        else:
            for i in current:
                newslab = hyperslab + [slice(i, i+1)]
                dimarray.append(self._get_elements(active, recursives[1:], hyperslab=newslab))

        return dimarray

def _determine_chunk_space(chunks, shape, dims, chunk_limits=True):
    
    if not chunks:
        return None

    chunk_space = [1 for i in range(len(shape))]

    max_chunks = np.prod(shape)
    if chunk_limits:
        max_chunks = int(max_chunks/ 2e6)

    for x, d in enumerate(dims):
        if d not in chunks:
            continue

        chunks_in_dim = chunks[d]
        if chunks_in_dim > max_chunks:
            chunks_in_dim = max_chunks

        chunk_space[x] = int(shape[x]/chunks[d])

    return tuple(chunk_space)
