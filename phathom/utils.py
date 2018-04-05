import os
import pickle
import numpy as np
from itertools import product
import pyina.launchers
from pyina.ez_map import ez_map

# TODO: Convert utils.py module to use pathlib module


def make_dir(path):
    """
    :param path: path of the directory to create
    :return: absolute path of the directory
    Makes a new directory at the provided path only if it doesn't already exist.
    """
    if not os.path.exists(path):
        os.makedirs(path)
    return os.path.abspath(path)


def files_in_dir(path):
    """ Searches a path for all files

    :param path: path of the directory to check for files
    :return: list of all files in the input path
    """
    return sorted(os.listdir(path))


def tifs_in_dir(path):
    """ Searches input path for tif files

    :param path: path of the directory to check for tif images
    :return: two lists containing the paths and filenames of all tifs found in provided path
    """
    abspath = os.path.abspath(path)
    files = files_in_dir(abspath)
    tif_paths = []
    tif_filenames = []
    for f in files:
        if f.endswith('.tif'):
            tif_paths.append(os.path.join(abspath, f))
            tif_filenames.append(f)
    return tif_paths, tif_filenames


def load_metadata(path):
    """ Loads a metadata.pkl file within provided path

    :param path: path of a directory containing a 'metadata.pkl' file
    :return: dictionary containing the stored metadata
    """
    return pickle_load(os.path.join(path, 'metadata.pkl'))


def pickle_save(path, data):
    """ Pickles data and saves it to provided path

    :param path: path of the pickle file to create / overwrite
    :param data: data to be pickled
    """
    with open(path, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


def pickle_load(path):
    """ Un-pickles a file provided at the input path

    :param path: path of the pickle file to read
    :return: data that was stored in the input pickle file
    """
    with open(path, 'rb') as f:
        return pickle.load(f)


def chunk_dims(img_shape, chunk_shape):
    """ Calculate the number of chunks needed for a given image shape
    :param: tuple containing whole image shape
    :param: tuple containing individual chunk shape
    :return: a tuple containing the number of chunks in each dimension
    """
    return tuple(int(np.ceil(i/c)) for i, c in zip(img_shape, chunk_shape))


def chunk_coordinates(shape, chunks):
    """ Calculate the global coordaintes for each chunk's starting position

    :param shape: shape of the image to chunk
    :param chunks: shape of each chunk
    :return: a list containing the starting indices of each chunk
    """
    nb_chunks = chunk_dims(shape, chunks)
    start = []
    for indices in product(*tuple(range(n) for n in nb_chunks)):
        start.append(tuple(i*c for i, c in zip(indices, chunks)))
    return start

mapper = None

def parallel_map(fn, args):
    """Map a function over an argument list, returning one result per arg

    :param fn: the function to execute
    :param args: a list of the single argument to send through the function
                 per invocation
    :returns: a list of results.

    Usage:
        myresults = parallel_map(my_function, my_inputs)

    Configuration:
        The mapper is configured by two environment variables:

        PHATHOM_LAUNCHER - this is the name of one of the launchers, e.g.
                           "mpirun" for OpenMPI or similar or "srun" for
                           SLURM's srun.
                           The full list can be obtained by running
                [ _.split(".")[1][:-9] for _ in pyina.launchers.all_launchers()]

        PHATHOM_NODES - this is the number of nodes that should be used in
                        parallel.

    By default, a serial mapper is returned if there is no mapper.
    """
    global mapper

    if mapper is None:
        if "PHATHOM_LAUNCHER" in os.environ:
            launcher_name = os.environ["PHATHOM_LAUNCHER"] + "_launcher"
            launcher = getattr(pyina.launchers, launcher_name)
            if "PHATHOM_NODES" in os.environ:
                nodes = int(os.environ["PHATHOM_NODES"])
                mapper = pyina.launchers.ParallelMapper(
                    nodes, launcher=launcher)
            else:
                mapper = pyina.launchers.ParallelMapper(launcher=launcher)
        else:
            mapper = pyina.launchers.SerialMapper()

    return mapper.map(fn, args)