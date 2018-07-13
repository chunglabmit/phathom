import skimage.external.tifffile as tifffile
import multiprocessing
import numpy as np


def imread(path):
    """ Reads Tiff file into a numpy array in memory.

    :param path: path to tif image to open
    :return: numpy ndarray with image data
    """
    return tifffile.imread(files=path).astype(np.uint16)  # ADDED TYPE CAST


def imsave(path, data, compress=1):
    """ Saves numpy array as a TIF image.

    :param path: path to tif image to create / overwrite
    :param data: numpy ndarray with image data
    :param compress: int (0-9) indicating the degree of lossless compression
    """
    tifffile.imsave(file=path, data=data, compress=compress)


def imread_parallel(paths, nb_workers):
    """ Reads Tiff files into a numpy array in memory.

    :param paths: A list of tiff paths to read (order is preserved)
    :param nb_workers: An int indicating how many parallel processes to use
    """
    img = imread(paths[0]).astype(np.uint16)  # ADDED TYPE CAST
    with multiprocessing.Pool(nb_workers) as pool:
        data = np.array(pool.map(imread, paths), img.dtype)
    return data
