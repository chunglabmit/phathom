import skimage.external.tifffile as tifffile
import multiprocessing
import numpy as np


def imread(path):
    """ Reads TIF file into a numpy array in memory.

    :param path: path to tif image to open
    :return: numpy ndarray with image data
    """
    return tifffile.imread(files=path)


def imsave(path, data, compress=0):
    """ Saves numpy array as a TIF image.

    :param path: path to tif image to create / overwrite
    :param data: numpy ndarray with image data
    :param compress: int (0-9) indicating the degree of lossless compression
    """
    tifffile.imsave(file=path, data=data, compress=compress)


def read_tifs(tif_paths):
    # TODO: fixed hard-coded nb_workers
    img = imread(tif_paths[0])
    shape = (len(tif_paths), *img.shape)
    data = np.zeros(shape, dtype=img.dtype)
    with multiprocessing.Pool(16) as pool:
        data = np.array(pool.map(imread, tif_paths))
    return data
