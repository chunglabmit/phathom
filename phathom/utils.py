import contextlib
import os
import pickle
import numpy as np
from itertools import product
import sys
if sys.platform.startswith("linux"):
    is_linux = True
    import tempfile
else:
    is_linux = False
    import mmap

# import pyina.launchers
# from pyina.ez_map import ez_map

# TODO: Convert utils.py module to use pathlib module


def make_dir(path):
    """Makes a new directory at the provided path only if it doesn't already exist.

    Parameters
    ----------
    path : str
        The path of the directory to make

    """
    if not os.path.exists(path):
        os.makedirs(path)
    return os.path.abspath(path)


def files_in_dir(path):
    """Searches a path for all files

    Parameters
    ----------
    path : str
        The directory path to check for files

    Returns
    -------
    list
        list of all files and subdirectories in the input path (excluding . and ..)

    """
    return sorted(os.listdir(path))


def tifs_in_dir(path):
    """Searches input path for tif files

    Parameters
    ----------
    path : str
        path of the directory to check for tif images

    Returns
    -------
    tif_paths : list
        list of paths to tiffs in path
    tif_filenames : list
        list of tiff filenames (with the extension) in path

    """
    abspath = os.path.abspath(path)
    files = files_in_dir(abspath)
    tif_paths = []
    tif_filenames = []
    for f in files:
        if f.endswith('.tif') or f.endswith('.tiff'):
            tif_paths.append(os.path.join(abspath, f))
            tif_filenames.append(f)
    return tif_paths, tif_filenames


def load_metadata(path):
    """Loads a metadata.pkl file within provided path

    Parameters
    ----------
    path : str
        path of a directory containing a 'metadata.pkl' file

    Returns
    -------
    dict
        dictionary containing the stored metadata

    """
    return pickle_load(os.path.join(path, 'metadata.pkl'))


def pickle_save(path, data):
    """Pickles data and saves it to provided path

    Parameters
    ----------
    path : str
        path of the pickle file to create / overwrite
    data : dict
        dictionary with data to be pickled

    """
    with open(path, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


def pickle_load(path):
    """Un-pickles a file provided at the input path

    Parameters
    ----------
    path : str
        path of the pickle file to read

    Returns
    -------
    dict
        data that was stored in the input pickle file

    """
    with open(path, 'rb') as f:
        return pickle.load(f)


def chunk_dims(img_shape, chunk_shape):
    """Calculate the number of chunks needed for a given image shape

    Parameters
    ----------
    img_shape : tuple
        whole image shape
    chunk_shape : tuple
        individual chunk shape

    Returns
    -------
    nb_chunks : tuple
        a tuple containing the number of chunks in each dimension

    """
    return tuple(int(np.ceil(i/c)) for i, c in zip(img_shape, chunk_shape))


def chunk_coordinates(shape, chunks):
    """Calculate the global coordaintes for each chunk's starting position

    Parameters
    ----------
    shape : tuple
        shape of the image to chunk
    chunks : tuple
        shape of each chunk

    Returns
    -------
    list
        the starting indices of each chunk

    """
    nb_chunks = chunk_dims(shape, chunks)
    start = []
    for indices in product(*tuple(range(n) for n in nb_chunks)):
        start.append(tuple(i*c for i, c in zip(indices, chunks)))
    return start


# mapper = None
#
#
# def parallel_map(fn, args):
#     """Map a function over an argument list, returning one result per arg
#
#     Parameters
#     ----------
#     fn : callable
#         the function to execute
#     args : list
#         a list of the single argument to send through the function per invocation
#
#     Returns
#     -------
#     list
#         a list of results
#
#     Notes
#     -----
#     The mapper is configured by two environment variables:
#
#     PHATHOM_MAPPER - this is the name of one of the mapper classes. Typical
#                      choices are MpiPool or MpiScatter for OpenMPI and
#                      SlurmPool or SlurmScatter for SLURM. By default, it
#                      uses the serial mapper which runs on a single thread.
#
#     PHATHOM_NODES - this is the number of nodes that should be used in
#                     parallel.
#
#     By default, a serial mapper is returned if there is no mapper.
#
#     Examples
#     --------
#     myresults = parallel_map(my_function, my_inputs)
#
#     """
#     global mapper
#
#     if mapper is None:
#         if "PHATHOM_MAPPER" in os.environ:
#             mapper_name = os.environ["PHATHOM_MAPPER"]
#             mapper_class = getattr(pyina.launchers, mapper_name)
#             if "PHATHOM_NODES" in os.environ:
#                 nodes = os.environ["PHATHOM_NODES"]
#                 mapper = mapper_class(nodes)
#             else:
#                 mapper = mapper_class()
#         else:
#             mapper = pyina.launchers.SerialMapper()
#
#     return mapper.map(fn, args)

class SharedMemory:
    """A class to share memory between processes

    Instantiate this class in the parent process and use in all processes.

    For all but Linux, we use the mmap module to get a buffer for Numpy
    to access through numpy.frombuffer. But in Linux, we use /dev/shm which
    has no file backing it and does not need to deal with maintaining a
    consistent view of itself on a disk.

    Typical use:

    shm = SharedMemory((100, 100, 100), np.float32)

    def do_something():

        with shm.txn() as a:

            a[...] = ...

    with multiprocessing.Pool() as pool:

        pool.apply_async(do_something, args)

    """

    if is_linux:
        def __init__(self, shape, dtype):
            """Initializer

            :param shape: the shape of the array

            :param dtype: the data type of the array
            """
            self.tempfile = tempfile.NamedTemporaryFile(
                prefix="proc_%d_" % os.getpid(),
                suffix=".shm",
                dir="/dev/shm",
                delete=True)
            self.pathname = self.tempfile.name
            self.shape = shape
            self.dtype = np.dtype(dtype)

        @contextlib.contextmanager
        def txn(self):
            """ A contextual wrapper of the shared memory

            :return: a view of the shared memory which has the shape and
            dtype given at construction
            """
            memory = np.memmap(self.pathname,
                               shape=self.shape,
                               dtype=self.dtype)
            yield memory
            del memory

        def __getstate__(self):
            return self.pathname, self.shape, self.dtype

        def __setstate__(self, args):
            self.pathname, self.shape, self.dtype = args

    else:
        def __init__(self, shape, dtype):
            """Initializer

            :param shape: the shape of the array

            :param dtype: the data type of the array
            """
            length = np.prod(shape) * dtype.itemsize
            self.mmap = mmap.mmap(-1, length)
            self.shape = shape
            self.dtype = dtype

        def txn(self):
            """ A contextual wrapper of the shared memory

            :return: a view of the shared memory which has the shape and
            dtype given at construction
            """
            memory = np.frombuffer(self.mmap, self.shape, self.dtype)
            yield memory
            del memory