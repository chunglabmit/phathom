import contextlib
import os
import pickle
import numpy as np
from itertools import product
import multiprocessing
import tqdm
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
    start_coords : ndarray
        the starting indices of each chunk

    """
    nb_chunks = chunk_dims(shape, chunks)
    start = []
    for indices in product(*tuple(range(n) for n in nb_chunks)):
        start.append(tuple(i*c for i, c in zip(indices, chunks)))
    return np.asarray(start)


def box_slice_idx(start, stop):
    """Creates an index tuple for a bounding box from `start` to `stop` using slices

    Parameters
    ----------
    start : array-like
        index of box start
    stop : array-like
        index of box stop (index not included in result)

    Returns
    -------
    idx : tuple
        index tuple for bounding box

    """
    return tuple(np.s_[a:b] for a, b in zip(start, stop))


def extract_box(arr, start, stop):
    """Indexes `arr` from `start` to `stop`

    Parameters
    ----------
    arr : array-like or SharedMemory
        input array to index
    start : array-like
        starting index of the slice
    stop : array-like
        ending index of the slice. The element at this index is not included.

    Returns
    -------
    box : ndarray
        resulting box from `arr`

    """
    idx = box_slice_idx(start, stop)
    if isinstance(arr, SharedMemory):
        with arr.txn() as a:
            box = a[idx]
    else:
        box = arr[idx]
    return box


def insert_box(arr, start, stop, data):
    """Indexes `arr` from `start` to `stop` and inserts `data`

    Parameters
    ----------
    arr : array-like
        input array to index
    start : array-like
        starting index of the slice
    stop : array-like
        ending index of the slice. The element at this index is not included.
    data : array-like
        sub-array to insert into `arr`

    Returns
    -------
    box : ndarray
        resulting box from `arr`

    """
    idx = box_slice_idx(start, stop)
    if isinstance(arr, SharedMemory):
        with arr.txn() as a:
            a[idx] = data
    else:
        arr[idx] = data
    return arr


def pmap_chunks(f, arr, chunks=None, nb_workers=None, use_imap=False):
    """Maps a function over an array in parallel using chunks

    The function `f` should take a reference to the array, a starting index, and the chunk size.
    Since each subprocess is handling it's own indexing, any overlapping should be baked into `f`.
    Caution: `arr` may get copied if not using memmap. Use with SharedMemory or Zarr array to avoid copies.

    Parameters
    ----------
    f : callable
        function with signature f(arr, start_coord, chunks). May need to use partial to define other args.
    arr : array-like
        an N-dimensional input array
    chunks : tuple, optional
        the shape of chunks to use. Default tries to access arr.chunks and falls back to arr.shape
    nb_workers : int, optional
        number of parallel processes to apply f with. Default, cpu_count
    use_imap : bool, optional
        whether or not to use imap instead os starmap in order to get an iterator for tqdm.
        Note that this requires input tuple unpacking manually inside of `f`.

    Returns
    -------
    result : list
        list of results for each chunk

    """
    if chunks is None:
        try:
            chunks = arr.chunks
        except AttributeError:
            chunks = arr.shape

    if nb_workers is None:
        nb_workers = multiprocessing.cpu_count()

    start_coords = chunk_coordinates(arr.shape, chunks)

    args_list = []
    for i, start_coord in enumerate(start_coords):
        args = (arr, start_coord, chunks)
        args_list.append(args)

    with multiprocessing.Pool(processes=nb_workers) as pool:
        if use_imap:
            results = list(tqdm.tqdm(pool.imap(f, args_list), total=len(args_list)))
        else:
            results = list(pool.starmap(f, args_list))

    return results


def extract_ghosted_chunk(arr, start_coord, chunks, overlap):
    end_coord = np.minimum(arr.shape, start_coord + np.asarray(chunks))
    start_coord_ghosted = np.maximum(np.zeros(arr.ndim, 'int'),
                                     np.array([s - overlap for s in start_coord]))
    stop_coord_ghosted = np.minimum(arr.shape,
                                    np.array([e + overlap for e in end_coord]))
    ghosted_chunk = extract_box(arr, start_coord_ghosted, stop_coord_ghosted)
    return ghosted_chunk, start_coord_ghosted, stop_coord_ghosted


def filter_points_in_box(coords, start, stop):
    interior_z = np.logical_and(coords[:, 0] >= start[0], coords[:, 0] < stop[0])
    interior_y = np.logical_and(coords[:, 1] >= start[1], coords[:, 1] < stop[1])
    interior_x = np.logical_and(coords[:, 2] >= start[2], coords[:, 2] < stop[2])
    interior = np.logical_and(np.logical_and(interior_z, interior_y), interior_x)
    return coords[np.where(interior)]


def filter_ghosted_points(start_ghosted, start_coord, centers_local, chunks, overlap):
    # filter nuclei on edges
    if start_ghosted[0] < start_coord[0]:
        interior_z = np.logical_and(centers_local[:, 0] >= overlap, centers_local[:, 0] < chunks[0] + overlap)
    else:
        interior_z = (centers_local[:, 0] < chunks[0])
    if start_ghosted[1] < start_coord[1]:
        interior_y = np.logical_and(centers_local[:, 1] >= overlap, centers_local[:, 1] < chunks[1] + overlap)
    else:
        interior_y = (centers_local[:, 1] < chunks[1])
    if start_ghosted[2] < start_coord[2]:
        interior_x = np.logical_and(centers_local[:, 2] >= overlap, centers_local[:, 2] < chunks[2] + overlap)
    else:
        interior_x = (centers_local[:, 2] < chunks[2])
    interior = np.logical_and(np.logical_and(interior_z, interior_y), interior_x)
    return centers_local[np.where(interior)]



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
