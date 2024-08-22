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


        if chunks == {} and False: # Remove for testing
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

    def _standard_sum(self, axes=None, skipna=None, **kwargs):
        """
        Standard Mean routine matches the normal routine for dask, required at this
        stage if Active mean not available.
        """

        arr = np.array(self)
        if skipna:
            total = np.nansum(arr, axis=axes, **kwargs)
        else:
            total = np.sum(arr, axis=axes, **kwargs)
        return total
    
    def _standard_max(self, axes=None, skipna=None, **kwargs):
        return np.max(self, axis=axes)
    
    def _standard_min(self, axes=None, skipna=None, **kwargs):
        return np.min(self, axis=axes)

    def _numel(self, method, axes=None):
        if not axes:
            return self.size
        
        size = 1
        for i in axes:
            size *= self.shape[i]
        newshape = list(self.shape)
        for ax in axes:
            newshape[ax] = 1

        return np.full(newshape, size)

    def active_method(self, method, axis=None, skipna=None, **kwargs):
        """
        Use PyActiveStorage package functionality to perform mean of this Fragment.

        :param axis:        (int) The axes over which to perform the active_mean operation.

        :param skipna:      (bool) Skip NaN values when calculating the mean.

        :returns:       A ``duck array`` (numpy-like) with the reduced array or scalar value, 
                        as specified by the axes parameter.
        """

        standard_methods = {
            'mean': self._standard_sum,
            'sum' : self._standard_sum,
            'max' : self._standard_max,
            'min' : self._standard_min
        }
        ret = None
        n = self._numel(method, axes=axis)

        try:
            from activestorage.active import Active
        except ImportError:
            # Unable to import Active package. Default to using normal mean.
            print("ActiveWarning: Unable to import active module - defaulting to standard method.")
            ret = {
                'n': n,
                'total': standard_methods[method](axes=axis, skipna=skipna, **kwargs)
            }

        if not ret:
            
            # Create Active client
            active = Active(self.filename, self.address)
            active.method = method

            # Fetch extent for this chunk instance.
            extent = tuple(self.get_extent())

            # Properly format the 'axis' kwarg.
            if axis == None:
                axis = tuple([i for i in range(self.ndim)])

            # Determine reduction parameter for combining chunk results for dask.
            n = self._numel(method, axes=axis)

            if len(axis) == self.ndim:
                data   = active[extent]
                t = self._post_process_data(data) * n

                ret = {
                    'n': n,
                    'total': t
                }

        if not ret:
            # Experimental Recursive requesting to get each 1D column along the axes being requested.
            range_recursives = []
            for dim in range(self.ndim):
                if dim not in axis:
                    range_recursives.append(range(extent[dim].start, extent[dim].stop))
                else:
                    range_recursives.append(extent[dim])
            results = np.array(self._get_elements(active, range_recursives, hyperslab=[]))

            t = self._post_process_data(results) * n
            ret = {
                'n': n,
                'total': t
            }

        if method == 'mean':
            return ret
        else:
            return ret['total']/ret['n']

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
