import numpy as np


# Holds all CFA-specific Active routines.
class ActiveChunk:

    description = "Container class for Active routines performed on each chunk. All active-per-chunk content can be found here."

    def __init__(self, *args, **kwargs):
        raise NotImplementedError
    
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
