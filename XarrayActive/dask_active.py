import dask.array as da
from dask.array.reductions import mean_agg


def block_active_mean(arr, *args, **kwargs):
    if hasattr(arr,'active_mean'):
        return arr.active_mean(*args, **kwargs)
    else:
        # Here's where barebones Xarray might fall over - may need a non-CFA custom class.
        raise NotImplementedError

class DaskActiveArray(da.Array):

    description = "Dask Array Wrapper enabling the use of Active Storage."

    @property
    def is_active(self):
        return True

    def copy(self):
        """
        Create a new DaskActiveArray instance with all the same parameters as the current instance.
        """
        return DaskActiveArray(self.dask, self.name, self.chunks, meta=self)
    
    def __getitem__(self, index):
        """
        Perform indexing for this ActiveArray. May need to overwrite further if it turns out
        the indexing is performed **after** the dask `getter` method (i.e if retrieval and indexing
        are separate items on the dask graph). If this is the case, will need another `from_delayed`
        and `concatenation` method as used in ``active_mean``.
        """
        arr = super().__getitem__(index)
        return DaskActiveArray(arr.dask, arr.name, arr.chunks, meta=arr)

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
            self,
            block_active_mean,
            mean_agg,
            axis=axis,
            dtype=self.dtype
        )

        return DaskActiveArray(newarr.dask, newarr.name, newarr.chunks, meta=newarr)
