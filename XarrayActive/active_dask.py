import dask.array as da
from dask.array.reductions import mean_agg, mean_combine
from dask.utils import deepmap
from dask.array.core import _concatenate2
import numpy as np

def partition_mean(arr, *args, **kwargs):
    return partition_method(arr, 'mean', *args, **kwargs)
    
def partition_max(arr, *args, **kwargs):
    return partition_method(arr, 'max', *args, **kwargs)
    
def partition_min(arr, *args, **kwargs):
    return partition_method(arr, 'min', *args, **kwargs)
    
def partition_sum(arr, *args, **kwargs):
    return partition_method(arr, 'sum', *args, **kwargs)
    
def partition_method(arr, method, *args, **kwargs):
    if hasattr(arr,'active_method'):
        return arr.active_method(method,*args, **kwargs)
    else:
        # Here's where barebones Xarray might fall over - may need a non-CFA custom class.
        x=input('stopped')
        raise NotImplementedError

def general_combine(pairs, axis=None):
    if not isinstance(pairs, list):
        pairs = [pairs]
    return _concatenate2(pairs, axes=axis)

def max_agg(pairs, axis=None, **kwargs):
    return general_combine(pairs, axis=axis).max(axis=axis, **kwargs)

def min_agg(pairs, axis=None, **kwargs):
    return general_combine(pairs, axis=axis).min(axis=axis, **kwargs)

def sum_agg(pairs, axis=None, **kwargs):
    return general_combine(pairs, axis=axis).sum(axis=axis, **kwargs)

class DaskActiveArray(da.Array):

    description = "Dask Array Wrapper enabling the use of Active Storage."

    @property
    def is_active(self):
        return True

    def copy(self):
        """
        Create a new DaskActiveArray instance with all the same parameters as the current instance.
        """
        copy_arr = DaskActiveArray(self.dask, self.name, self.chunks, meta=self)
        return copy_arr
    
    #def __getitem__(self, index):
    #    """
    #    Perform indexing for this ActiveArray. May need to overwrite further if it turns out
    #    the indexing is performed **after** the dask `getter` method (i.e if retrieval and indexing
    #    are separate items on the dask graph). If this is the case, will need another `from_delayed`
    #    and `concatenation` method as used in ``active_mean``.
    #    """
    #    copy = self.copy()

    #    return copy._getitem(index)

    #def _getitem(self, index):
    #    return super().__getitem__(index)

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
            self.copy(),
            partition_mean,
            mean_agg,
            combine=mean_combine,
            axis=axis,
            dtype=self.dtype,
        )

        return newarr

    def active_max(self, axis=None, skipna=None):
        """
        Perform ``dask delayed`` active mean for each ``dask block`` which corresponds to a single ``chunk``.
        Combines the results of the dask delayed ``active_max`` operations on each block into a single dask Array,
        which is then mapped to a new DaskActiveArray object.

        :param axis:        (int) The index of the axis on which to perform the active max.

        :param skipna:      (bool) Skip NaN values when calculating the max.

        :returns:       A new ``DaskActiveArray`` object which has been reduced along the specified axes using
                        the concatenations of active_means from each chunk.
        """

        newarr = da.reduction(
            self.copy(),
            partition_max,
            max_agg,
            combine=max_agg,
            axis=axis,
            dtype=self.dtype,
        )

        return newarr
    
    def active_min(self, axis=None, skipna=None):
        """
        Perform ``dask delayed`` active mean for each ``dask block`` which corresponds to a single ``chunk``.
        Combines the results of the dask delayed ``active_min`` operations on each block into a single dask Array,
        which is then mapped to a new DaskActiveArray object.

        :param axis:        (int) The index of the axis on which to perform the active min.

        :param skipna:      (bool) Skip NaN values when calculating the min.

        :returns:       A new ``DaskActiveArray`` object which has been reduced along the specified axes using
                        the concatenations of active_means from each chunk.
        """

        newarr = da.reduction(
            self.copy(),
            partition_min,
            min_agg,
            combine=min_agg,
            axis=axis,
            dtype=self.dtype,
        )

        return newarr
    
    def active_sum(self, axis=None, skipna=None):
        """
        Perform ``dask delayed`` active mean for each ``dask block`` which corresponds to a single ``chunk``.
        Combines the results of the dask delayed ``active_sum`` operations on each block into a single dask Array,
        which is then mapped to a new DaskActiveArray object.

        :param axis:        (int) The index of the axis on which to perform the active sum.

        :param skipna:      (bool) Skip NaN values when calculating the sum.

        :returns:       A new ``DaskActiveArray`` object which has been reduced along the specified axes using
                        the concatenations of active_means from each chunk.
        """

        newarr = da.reduction(
            self.copy(),
            partition_sum,
            sum_agg,
            combine=sum_agg,
            axis=axis,
            dtype=self.dtype,
        )

        return newarr