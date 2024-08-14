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

    def _set_active_options(self, chunks=None, chunk_limits=True):
        self._active_chunks = chunks
        self._chunk_limits = chunk_limits


# Holds all CFA-specific Active routines.
class ActiveChunk:

    description = "Container class for Active routines performed on each chunk. All active-per-chunk content can be found here."
    
    def _post_process_data(self, data):
        # Perform any post-processing steps on the data here
        return data

    def _standard_mean(self, axis=None, skipna=None, **kwargs):
        """
        Standard Mean routine matches the normal routine for dask, required at this
        stage if Active mean not available.
        """
        size = 1
        for i in axis:
            size *= self.shape[i]

        arr = np.array(self)
        if skipna:
            total = np.nanmean(arr, axis=axis, **kwargs) *size
        else:
            total = np.mean(arr, axis=axis, **kwargs) *size
        return {'n': self._numel(arr, axis=axis), 'total': total}

    def _numel(self, axis=None):
        if not axis:
            return self.size
        
        size = 1
        for i in axis:
            size *= self.shape[i]
        newshape = list(self.shape)
        newshape[axis] = 1

        return np.full(newshape, size)

    def active_mean(self, axis=None, skipna=None, **kwargs):
        """
        Use PyActiveStorage package functionality to perform mean of this Fragment.

        :param axis:        (int) The axis over which to perform the active_mean operation.

        :param skipna:      (bool) Skip NaN values when calculating the mean.

        :returns:       A ``duck array`` (numpy-like) with the reduced array or scalar value, 
                        as specified by the axis parameter.
        """
        try:
            from activestorage.active import Active
        except ImportError:
            # Unable to import Active package. Default to using normal mean.
            print("ActiveWarning: Unable to import active module - defaulting to standard method.")
            return self._standard_mean(axis=axis, skipna=skipna, **kwargs)
            
        active = Active(self.filename, self.address)
        active.method = "mean"
        extent = self.get_extent()

        if not axis is None:
            return {
                'n': self._numel(axis=axis),
                'total': self._post_process_data(active[extent])
            }

        # Experimental Recursive requesting to get each 1D column along the axis being requested.
        range_recursives = []
        for dim in range(self.ndim):
            if dim != axis:
                range_recursives.append(range(extent[dim].start, extent[dim].stop+1))
            else:
                range_recursives.append(extent[dim])
        results = np.array(self._get_elements(active, range_recursives, hyperslab=[]))

        return {
            'n': self._numel(axis=axis),
            'total': self._post_process_data(results)
        }

    def _get_elements(self, active, recursives, hyperslab=[]):
        dimarray = []
        current = recursives[0]
        if not len(recursives) > 1:

            # Perform active slicing and meaning here.
            return active[hyperslab]

        if type(current) == slice:
            newslab = hyperslab + [current]
            dimarray.append(self._get_elements(active, recursives[1:], hyperslab=newslab))

        else:
            for i in current:
                newslab = hyperslab + [slice(i, i+1)]
                dimarray.append(self._get_elements(active, recursives[1:], hyperslab=newslab))

        return dimarray

def _determine_chunk_space(chunks, shape, dims, chunk_limits=True):
    
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
