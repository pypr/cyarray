<%
import platform
type_info = [
    ('int', 'IntArray', 'NPY_INT'),
    ('unsigned int', 'UIntArray', 'NPY_UINT'),
    ('long', 'LongArray', 'NPY_LONG'),
    ('float', 'FloatArray', 'NPY_FLOAT'),
    ('double', 'DoubleArray', 'NPY_DOUBLE'),
]
%># This file (carray.pxd) has been generated automatically.
# DO NOT modify this file
# To make changes modify the source templates (carray.pxd.mako) and regenerate
# distutils: language=c++
# cython: embedsignature=True, language_level=3
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
"""
Implementation of resizeable arrays of different types in Cython.

All arrays provide for the following operations:

 - access by indexing.
 - access through get/set function.
 - appending values at the end of the array.
 - reserving space for future appends.
 - access to internal data through a numpy array.


Each array also provides an interface to its data through a numpy array.
This is done through the ``get_npy_array`` function. The returned numpy
array can be used just like any other numpy array but for the following
restrictions:

 - the array may not be resized.
 - references of this array should not be kept.
 - slices of this array may not be made.

The numpy array may however be copied and used in any manner.
"""

# For malloc etc.
from libc.stdlib cimport *

from libc.stdint cimport uintptr_t

cimport numpy as np
np.import_array()

import numpy as np

# logging imports
import logging
logger = logging.getLogger()

# 'importing' some Numpy C-api functions.
cdef extern from "numpy/arrayobject.h":
    cdef void  _import_array()

    ctypedef struct PyArrayObject:
        char  *data
        np.npy_intp *dimensions

    cdef enum NPY_TYPES:
        NPY_INT,
        NPY_UINT,
        NPY_LONG,
        NPY_FLOAT,
        NPY_DOUBLE

    np.ndarray PyArray_SimpleNewFromData(int, np.npy_intp*, int, void*)

    void *PyArray_DATA(np.ndarray arr) noexcept nogil

    np.npy_intp *PyArray_DIMS(np.ndarray arr) noexcept nogil


# memcpy
cdef extern from "stdlib.h":
     void *memcpy(void *dst, void *src, long n) noexcept nogil

# Cython>= 3.0 now disallows replacing the guts of a numpy array.
# Workaround: https://github.com/rainwoodman/pandas/blob/05d3fe2402e4563124e7060837ded7513ab5bca7/pandas/_libs/reduction.pyx#L27 # noqa: E501
cdef extern from *:
    """
    static void PyArray_SET_DATA(PyArrayObject *arr, char * data) {
        ((PyArrayObject_fields *)arr)->data = data;
    }
    """
    void PyArray_SET_DATA(np.ndarray arr, char * data) noexcept nogil

# numpy module initialization call
_import_array()

cdef inline long aligned(long n, int item_size) noexcept nogil:
    """Align `n` items each having size (in bytes) `item_size` to
    64 bytes and return the appropriate number of items that would
    be aligned to 64 bytes.
    """
    if n*item_size%64 == 0:
        return n
    else:
        if 64%item_size == 0:
            return (n*item_size//64 + 1)*64//item_size
        else:
            return (n*item_size//64 + 1)*64

cpdef long py_aligned(long n, int item_size):
    """Align `n` items each having size (in bytes) `item_size` to
    64 bits and return the appropriate number of items that would
    be aligned to 64 bytes.
    """
    return aligned(n, item_size)

cdef void* _aligned_malloc(size_t bytes) noexcept nogil:
    """Allocates block of memory starting on a cache line.

    Algorithm from:
    http://www.drdobbs.com/parallel/understanding-and-avoiding-memory-issues/212400410
    """
    cdef size_t cache_size = 64
    cdef char* base = <char*>malloc(cache_size + bytes)

    # Round pointer up to next line
    cdef char* result = <char*>(<uintptr_t>(base+cache_size)&-(cache_size))

    # Record where block actually starts.
    (<char**>result)[-1] = base

    return <void*>result

cdef void* _aligned_realloc(void *existing, size_t bytes, size_t old_size) noexcept nogil:
    """Allocates block of memory starting on a cache line.

    """
    cdef void* result = _aligned_malloc(bytes)
    cdef size_t copy_size = min(bytes, old_size)

    # Copy everything from the old to the new and free the old.
    memcpy(<void*>result, <void*>existing, copy_size)
    aligned_free(<void*>existing)

    return result

cdef void* _deref_base(void* ptr) noexcept nogil:
    cdef size_t cache_size = 64
    # Recover where block actually starts
    cdef char* base = (<char**>ptr)[-1]
    if <void*>(<uintptr_t>(base+cache_size)&-(cache_size)) != ptr:
        with gil:
            raise MemoryError("Passed pointer is not aligned.")
    return <void*>base

cdef void* aligned_malloc(size_t bytes) noexcept nogil:
    return _aligned_malloc(bytes)

cdef void* aligned_realloc(void* p, size_t bytes, size_t old_size) noexcept nogil:
    return _aligned_realloc(p, bytes, old_size)

cdef void aligned_free(void* p) noexcept nogil:
    """Free block allocated by alligned_malloc.
    """
    free(<void*>_deref_base(p))


cdef class BaseArray:
    """Base class for managed C-arrays.
    """

    #### Cython interface  #################################################

    cdef void c_align_array(self, LongArray new_indices, int stride=1) noexcept nogil:
        """Rearrange the array contents according to the new indices.
        """
        pass

    cdef void c_reserve(self, long size) noexcept nogil:
        pass

    cdef void c_reset(self) noexcept nogil:
        cdef np.npy_intp *dims
        self.length = 0
        dims = PyArray_DIMS(self._npy_array)
        dims[0] = self.length

    cdef void c_resize(self, long size) noexcept nogil:
        pass

    cdef void c_squeeze(self) noexcept nogil:
        pass

    #### Python interface  #################################################

    cpdef str get_c_type(self):
        """Return the c data type of this array.
        """
        raise NotImplementedError('BaseArray::get_c_type')

    cpdef reserve(self, long size):
        """Resizes the internal data to required size.
        """
        raise NotImplementedError('BaseArray::reserve')

    cpdef resize(self, long size):
        """Resizes the array to the new size.
        """
        raise NotImplementedError('BaseArray::resize')

    cpdef np.ndarray get_npy_array(self):
        """Returns a numpy array of the data: do not keep its reference.
        """
        return self._npy_array

    cpdef set_data(self, np.ndarray nparr):
        """Set data from the given numpy array.

        If the numpy array is a reference to the numpy array maintained
        internally by this class, nothing is done.
        Otherwise, if the size of nparr matches this array, values are
        copied into the array maintained.

        """
        cdef PyArrayObject* sarr = <PyArrayObject*>nparr
        cdef PyArrayObject* darr = <PyArrayObject*>self._npy_array
        cdef np.npy_intp *s_dims, *d_dims
        cdef void *s_data, *d_data
    
        s_data = PyArray_DATA(nparr)
        d_data = PyArray_DATA(self._npy_array)

        s_dims = PyArray_DIMS(nparr)
        d_dims = PyArray_DIMS(self._npy_array)

        if s_data == d_data:
            return
        elif s_dims[0] <= d_dims[0]:
            self._npy_array[:s_dims[0]] = nparr
        else:
            raise ValueError('array size mismatch')

    cpdef squeeze(self):
        """Release any unused memory.
        """
        raise NotImplementedError('BaseArray::squeeze')

    cpdef remove(self, np.ndarray index_list, bint input_sorted=0, int stride=1):
        """Remove the particles with indices in index_list.
        """
        raise NotImplementedError('BaseArray::remove')

    cpdef extend(self, np.ndarray in_array):
        """Extend the array with data from in_array.
        """
        raise NotImplementedError('BaseArray::extend')

    cpdef align_array(self, LongArray new_indices, int stride=1):
        """Rearrange the array contents according to the new indices.
        """
        if new_indices.length != self.length//stride:
            raise ValueError('Unequal array lengths')
        self.c_align_array(new_indices, stride)

    cpdef reset(self):
        """Reset the length of the array to 0.
        """
        raise NotImplementedError('BaseArray::reset')

    cpdef copy_values(self, LongArray indices, BaseArray dest, int stride=1, int start=0):
        """Copy values of indexed particles from self to dest.
        """
        raise NotImplementedError('BaseArray::copy_values')

    cpdef copy_subset(self, BaseArray source,
                      long start_index=-1, long end_index=-1, int stride=1):
        """Copy subset of values from source to self.
        """
        raise NotImplementedError('BaseArray::copy_subset')

    cpdef update_min_max(self):
        """Update the min and max values of the array.
        """
        raise NotImplementedError('BaseArray::update_min_max')

    def __len__(self):
        return self.length

    def __iter__(self):
        """ Support the iteration protocol"""
        return BaseArrayIter(self)


cdef class BaseArrayIter:
    """ Iteration object to support iteration over BaseArray. """
    def __init__(self, BaseArray arr):
        self.arr = arr
        self.i = -1

    def __next__(self):
        self.i = self.i+1
        if self.i < self.arr.length:
            return self.arr[self.i]
        else:
            raise StopIteration

    def __iter__(self):
        return self


% for ARRAY_TYPE, CLASSNAME, NUMPY_TYPENAME in type_info:
# ###########################################################################
# `${CLASSNAME}` class.
# ###########################################################################
cdef class ${CLASSNAME}(BaseArray):
    """Represents an array of `${ARRAY_TYPE}s`

    Mallocs a memory buffer of size (n*sizeof(${ARRAY_TYPE})) and sets up
    the numpy array.  The memory is aligned to 64 byte boundaries.

    Parameters
    ----------

    n : long
        Length of the array.

    Attributes
    ----------
    data: pointer
        Pointer to an integer array.
    length: long
        Size of the array itself.
    alloc: long
        Size of the data buffer allocated.

    Examples
    --------

    >>> x = ${CLASSNAME}()
    >>> x.resize(5)
    >>> x.set_data(np.arange(5))
    >>> x[0]
    0

    >>> x = ${CLASSNAME}(5)
    >>> xnp = x.get_npy_array()
    >>> xnp[:] = np.arange(5)
    >>> x[0], x[4]
    (0.0, 4.0)

    """

    #cdef public long length, alloc
    #cdef ${ARRAY_TYPE} *data
    #cdef np.ndarray _npy_array

    def __cinit__(self, long n=0):
        """Constructor for the class.
        """
        self.length = n
        self._parent = None
        self._old_data = NULL
        if n == 0:
            n = 16
        self.alloc = n
        self.data = <${ARRAY_TYPE}*>aligned_malloc(n*sizeof(${ARRAY_TYPE}))

        self._setup_npy_array()

    def __dealloc__(self):
        """Frees the array.
        """
        if self._old_data == NULL:
            aligned_free(<void*>self.data)
        else:
            aligned_free(<void*>self._old_data)

    def __getitem__(self, long idx):
        """Get item at position idx.
        """
        return self.data[idx]

    def __setitem__(self, long idx, ${ARRAY_TYPE} value):
        """Set location idx to value.
        """
        self.data[idx] = value

    cpdef long index(self, ${ARRAY_TYPE} value):
        """Returns the index at which value is in self, else -1.
        """
        cdef long i
        for i in range(self.length):
            if self.data[i] == value:
                return i
        return -1

    def __contains__(self, ${ARRAY_TYPE} value):
        """Returns True if value is in self.
        """
        return (self.index(value) >= 0)

    def __reduce__(self):
        """Implemented to facilitate pickling.
        """
        d = {}
        d['data'] = self.get_npy_array()

        return (${CLASSNAME}, (), d)

    def __setstate__(self, d):
        """Load the carray from the dictionary d.
        """
        cdef np.ndarray arr = d['data']
        self.resize(arr.size)
        self.set_data(arr)

    cdef _setup_npy_array(self):
        """Create the numpy array.
        """
        cdef int nd = 1
        cdef np.npy_intp dims = self.length

        self._npy_array = PyArray_SimpleNewFromData(
            nd, &dims, ${NUMPY_TYPENAME}, self.data
        )

    ##### Cython protocol ######################################

    cdef void c_align_array(self, LongArray new_indices, int stride=1) noexcept nogil:
        """Rearrange the array contents according to the new indices.
        """

        cdef long i, j, new_index
        cdef long length = self.length
        cdef long n_bytes
        cdef ${ARRAY_TYPE} *temp

        n_bytes = sizeof(${ARRAY_TYPE})*length
        temp = <${ARRAY_TYPE}*>aligned_malloc(n_bytes)

        memcpy(<void*>temp, <void*>self.data, n_bytes)

        # copy the data from the resized portion to the actual positions.
        if stride == 1:
            for i in range(length):
                new_index = new_indices.data[i]
                if i != new_index:
                    self.data[i] = temp[new_index]
        else:
            for i in range(length//stride):
                new_index = new_indices.data[i]
                if i != new_index:
                    for j in range(stride):
                        self.data[i*stride + j] = temp[new_index*stride + j]

        aligned_free(<void*>temp)

    cdef void c_append(self, ${ARRAY_TYPE} value) noexcept nogil:
        cdef long l = self.length
        cdef np.npy_intp *arr_dims

        if l >= self.alloc:
            self.c_reserve(l*2)
        self.data[l] = value
        self.length += 1

        # update the numpy arrays length
        arr_dims = PyArray_DIMS(self._npy_array)
        arr_dims[0] = self.length

    cdef void c_reserve(self, long size) noexcept nogil:
        cdef void* data = NULL
        if size > self.alloc:
            data = <${ARRAY_TYPE}*>aligned_realloc(
                self.data, size*sizeof(${ARRAY_TYPE}),
                self.alloc*sizeof(${ARRAY_TYPE})
            )

            if data == NULL:
                aligned_free(<void*>self.data)
                with gil:
                    raise MemoryError

            self.data = <${ARRAY_TYPE}*>data
            self.alloc = size
            PyArray_SET_DATA(self._npy_array, <char *>self.data)

    cdef void c_reset(self) noexcept nogil:
        BaseArray.c_reset(self)
        if self._old_data != NULL:
            self.data = self._old_data
            self._old_data = NULL
            PyArray_SET_DATA(self._npy_array, <char *>self.data)

    cdef void c_resize(self, long size) noexcept nogil:
        cdef np.npy_intp *arr_dims

        # reserve memory
        self.c_reserve(size)

        # update the lengths
        self.length = size
        arr_dims = PyArray_DIMS(self._npy_array)
        arr_dims[0] = self.length

    cdef void c_set_view(self, ${ARRAY_TYPE} *array, long length) noexcept nogil:
        """Create a view of a given raw data pointer with given length.
        """
        cdef np.npy_intp *arr_dims

        if self._old_data == NULL:
            self._old_data = self.data

        self.data = array
        self.length = length
        PyArray_SET_DATA(self._npy_array, <char *>self.data)
        arr_dims = PyArray_DIMS(self._npy_array)
        arr_dims[0] = self.length

    cdef void c_squeeze(self) noexcept nogil:
        cdef PyArrayObject* arr = <PyArrayObject*>self._npy_array
        cdef void* data = NULL
        cdef size_t size = max(self.length, 16)
        data = <${ARRAY_TYPE}*>aligned_realloc(
            self.data, size*sizeof(${ARRAY_TYPE}),
            self.alloc*sizeof(${ARRAY_TYPE})
        )

        if data == NULL:
            # free original data
            aligned_free(<void*>self.data)
            with gil:
                raise MemoryError

        self.data = <${ARRAY_TYPE}*>data
        self.alloc = size
        PyArray_SET_DATA(self._npy_array, <char *>self.data)

    ##### Python protocol ######################################

    cpdef str get_c_type(self):
        """Return the c data type for this array as a string.
        """
        return '${ARRAY_TYPE}'

    cdef ${ARRAY_TYPE}* get_data_ptr(self):
        """Return the internal data pointer.
        """
        return self.data

    cpdef ${ARRAY_TYPE} get(self, long idx):
        """Gets value stored at position `idx`.
        """
        return self.data[idx]

    cpdef set(self, long idx, ${ARRAY_TYPE} value):
        """Sets location `idx` to `value`.
        """
        self.data[idx] = value

    cpdef append(self, ${ARRAY_TYPE} value):
        """Appends `value` to the end of the array.
        """
        self.c_append(value)

    cpdef reserve(self, long size):
        """Resizes the internal data to ``size*sizeof(${ARRAY_TYPE})`` bytes.
        """
        self.c_reserve(size)

    cpdef reset(self):
        """Reset the length of the array to 0.
        """
        self.c_reset()
        if self._old_data != NULL:
            self._parent = None

    cpdef resize(self, long size):
        """Resizes internal data to ``size*sizeof(${ARRAY_TYPE})`` bytes
        and sets the length to the new size.

        """
        if self._old_data != NULL:
            raise RuntimeError('Cannot reize array which is a view.')

        self.c_resize(size)

    cpdef set_view(self, ${CLASSNAME} parent, long start, long end):
        """Create a view of a given a `parent` array from start to end.

        Note that this excludes the end index.

        Parameters
        ----------

        parent : ${CLASSNAME}
            The parent array of which this is a view.
        start : long
            The starting index to start the view from.
        end : long
            The ending index to end the view at, excludes the end
            itself.

        """
        cdef np.npy_intp *arr_dims

        if self._parent is None:
            self._old_data = self.data
        self._parent = parent
        self.data = parent.data + start
        self.length = end - start
        PyArray_SET_DATA(self._npy_array, <char *>self.data)
        arr_dims = PyArray_DIMS(self._npy_array)
        arr_dims[0] = self.length

    cpdef squeeze(self):
        """Release any unused memory.
        """
        if self._old_data != NULL:
            raise RuntimeError('Cannot squeeze array which is a view.')

        self.c_squeeze()

    cpdef remove(self, np.ndarray index_list, bint input_sorted=0, int stride=1):
        """Remove the particles with indices in index_list.

        Parameters
        ----------

        index_list : ndarray
            a list of indices which should be removed.
        input_sorted : bool
            indicates if the input is sorted in ascending order.  if
            not, the array will be sorted internally.
        stride : int
            indicates the stride size for the indices.

        Notes
        -----

         If the input indices are not sorted, sort them in ascending order.
         Starting with the last element in the index list, start replacing the
         element at the said index with the last element in the data and update
         the length of the array.

         If stride is 3, then the indices are multiplied by 3 and chunks of 3
         elements are removed.

        """
        if self._old_data != NULL:
            raise RuntimeError('Cannot remove elements from view array.')
        cdef long i
        cdef int j
        cdef long inlength = index_list.size
        cdef np.ndarray sorted_indices
        cdef long id
        cdef np.npy_intp *arr_dims

        if inlength > self.length:
            return

        if input_sorted != 1:
            sorted_indices = np.sort(index_list)
        else:
            sorted_indices = index_list

        if stride == 1:
            for i in range(inlength):
                id = sorted_indices[inlength-(i+1)]
                if id < self.length:
                    self.data[id] = self.data[self.length-1]
                    self.length = self.length - 1
        else:
            for i in range(inlength):
                id = sorted_indices[inlength-(i+1)]*stride
                if id < self.length:
                    for j in range(stride):
                        self.data[id + j] = self.data[self.length - stride + j]
                    self.length = self.length - stride
        arr_dims = PyArray_DIMS(self._npy_array)
        arr_dims[0] = self.length

    cpdef extend(self, np.ndarray in_array):
        """Extend the array with data from in_array.

        Parameters
        ----------

        in_array : ndarray
            a numpy array with data to be added to the current array.

        Notes
        -----

         - accessing the in_array using the indexing operation seems to be
           costly. Look at the annotated cython html file.

        """
        if self._old_data != NULL:
            raise RuntimeError('Cannot extend array which is a view.')
        cdef long len = in_array.size
        cdef long i
        for i in range(len):
            self.append(in_array[i])

    cpdef copy_values(self, LongArray indices, BaseArray dest, int stride=1, int start=0):
        """Copies values of indices in indices from self to `dest`.

        No size check if performed, we assume the dest to of proper size
        i.e. atleast as long as indices.
        """
        cdef ${CLASSNAME} dest_array = <${CLASSNAME}>dest
        cdef long i, num_values
        cdef int j
        num_values = indices.length

        if stride == 1:
            for i in range(num_values):
                dest_array.data[start + i] = self.data[indices.data[i]]
        else:
            for i in range(num_values):
                for j in range(stride):
                    dest_array.data[start + i*stride + j] = self.data[indices.data[i]*stride + j]

    cpdef copy_subset(self, BaseArray source, long start_index=-1,
                      long end_index=-1, int stride=1):
        """Copy a subset of values from src to self.

        Parameters
        ----------

        start_index : long
            the first index in self that corresponds to the 0th
            index in source

        end_index : long
            the first index in self from start_index that is not copied

        stride : int
            the stride along indices, basically copy over chunks of
            the given stride.
        """
        cdef long si, ei, s_length, d_length, i, j
        cdef int k
        cdef ${CLASSNAME} src = <${CLASSNAME}>source
        s_length = src.length
        d_length = self.length

        if end_index < 0:
            if start_index < 0:
                if s_length != d_length:
                    msg = 'Source length should be same as dest length'
                    logger.error(msg)
                    raise ValueError, msg
                si = 0
                ei = self.length
            else:
                # meaning we copy from the specified start index to the end of
                # self. make sure the sizes are consistent.
                si = start_index
                ei = d_length

                if start_index > (d_length-1):
                    msg = 'start_index beyond array length'
                    logger.error(msg)
                    raise ValueError, msg

                if (ei - si) > s_length:
                    msg = 'Not enough values in source'
                    logger.error(msg)
                    raise ValueError, msg
        else:
            if start_index < 0:
                msg = 'start_index : %d, end_index : %d'%(start_index,
                                                          end_index)
                logger.error(msg)
                raise ValueError, msg
            else:
                if (start_index > (d_length-1) or end_index > d_length or
                    start_index > end_index):
                    msg = 'start_index : %d, end_index : %d'%(start_index,
                                                              end_index)
                    logger.error(msg)
                    raise ValueError, msg

                si = start_index
                ei = end_index

        # we have valid start and end indices now. can start copying now.
        if stride == 1:
            j = 0
            for i in range(si, ei):
                self.data[i] = src.data[j]
                j += 1
        else:
            j = 0
            for i in range(si, ei):
                for k in range(stride):
                    self.data[i*stride + k] = src.data[j*stride + k]
                j += 1

    cpdef update_min_max(self):
        """Updates the min and max values of the array.
        """
        cdef long i = 0
        cdef ${ARRAY_TYPE} min_val, max_val

        if self.length == 0:
            self.minimum = <${ARRAY_TYPE}>0
            self.maximum = <${ARRAY_TYPE}>0
            return

        min_val = self.data[0]
        max_val = self.data[0]

        for i in range(self.length):
            if min_val > self.data[i]:
                min_val = self.data[i]
            if max_val < self.data[i]:
                max_val = self.data[i]

        self.minimum = min_val
        self.maximum = max_val

% endfor
