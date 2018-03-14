import os
import pickle

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
    """
    :param path: path of the directory to check for files
    :return: list of all files in the input path
    Searches a path for all files
    """
    return os.listdir(path)


def tifs_in_dir(path):
    """
    :param path: path of the directory to check for tif images
    :return: two lists containing the paths and filenames of all tifs found in provided path
    Searches input path for tif files
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
    """
    :param path: path of a directory containing a 'metadata.pkl' file
    :return: dictionary containing the stored metadata
    Loads a metadata.pkl file within provided path
    """
    return pickle_load(os.path.join(path, 'metadata.pkl'))


def pickle_save(path, data):
    """
    :param path: path of the pickle file to create / overwrite
    :param data: data to be pickled
    Pickles data and saves it to provided path
    """
    with open(path, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


def pickle_load(path):
    """
    :param path: path of the pickle file to read
    :return: data that was stored in the input pickle file
    Un-pickles a file provided at the input path
    """
    with open(path, 'rb') as f:
        return pickle.load(f)
