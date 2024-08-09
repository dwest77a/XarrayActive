from XarrayActive.partition import ArrayPartition, ArrayLike

class ActiveNetCDF4Wrapper(ArrayLike):

    def __init__(
            self, 
            variable_name, 
            datastore, 
            chunks=None, 
            extent=None
        ):
        self.datastore     = datastore
        self.variable_name = variable_name

        self._chunks = chunks
        self._extent = extent
        self._lock = SerializableLock()

        self._variable = self.get_variable()
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

        self.__array_function__ = self.get_array

        super().__init__(shape, units=units, dtype=dtype)
                
    def get_variable(self, needs_lock=True):
        ds = self.datastore._acquire(needs_lock)
        variable = ds.variables[self.variable_name]
        variable.set_auto_maskandscale(False)
        # only added in netCDF4-python v1.2.8
        with suppress(AttributeError):
            variable.set_auto_chartostring(False)
        return variable

    def get_array(self):

        if not self._chunks:
            # get_array should just get the whole array if that's what we're trying to do.
            # indexing should just be added to the instance of this class, and then the
            # built-in mean from _ActiveFragment should take care of things.
            return self._variable
        else:

            # for every dask chunk return a smaller object with the right extent.
            # Create a chunk_shape tuple from chunks and _variable (figure out which chunk and which axis, divide etc.)
            # Define a subarray for each chunk, with appropriate index.

            f_indices = None # from chunks
            chunks = None # Need to find out what this needs to be.

            name = (f"{self.__class__.__name__}-{tokenize(self)}",)
            dsk = {}
            for f_index in f_indices:

                position = None
                extent   = None
                cformat  = None
                
                subarray = ArrayPartition(
                    self.filename,
                    self.variable_name,
                    dtype=self.dtype,
                    units=self.units,
                    shape=self.shape,
                    position=position,
                    extent=extent,
                    format=cformat
                )

                key = f"{subarray.__class__.__name__}-{tokenize(subarray)}"
                dsk[key] = subarray
                dsk[name + f_index] = (
                    getter, # Dask default should be enough with the new indexing routine.
                    key,
                    f_indices,
                    False,
                    getattr(subarray,"_lock",False)
                )

            return DaskActiveArray(dsk, name, chunks=chunks, dtype=self.dtype)


    def __getitem__(self, index):
        self._extent = self._combine_slices(index)
        return self

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