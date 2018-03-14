from . import utils
import skimage.external.tifffile as tifffile
import zarr



def imread(path):
    """
    :param path: path to tif image to open
    :return: numpy ndarray with image data
    Reads TIF file into a numpy array in memory.
    """
    return tifffile.imread(files=path)


def imsave(path, data):
    """
    :param path: path to tif image to create / overwrite
    :param data: numpy ndarray with image data
    Saves numpy array as a TIF image.
    """
    tifffile.imsave(file=path, data=data)


def tifs_to_zarr(tif_paths, zarr_path, chunks):
    """
    :param tif_paths: list of paths to 2D tif images
    :param zarr_path: path of the zarr array to be created / overwritten
    :param chunks: tuple of ints specifying the chunk shape
    :return: reference to persistent (on-disk) zarr array
    Loads 2D TIF images into a (chunked) zarr array one-by-one
    """
    img = imread(tif_paths[0])
    shape = (len(tif_paths), *img.shape)
    dtype = img.dtype
    z_arr = zarr.open(zarr_path, mode='w', shape=shape, chunks=chunks, dtype=dtype)
    for z, tif_path in enumerate(tif_paths):
        z_arr[z] = imread(tif_path)
    return z_arr


def tif_to_zarr(tif_path, zarr_path, chunks):
    """
    :param tif_path: path to a 3D tif image
    :param zarr_path: path of the zarr array to be created / overwritten
    :param chunks: tuple of ints specifying the chunk shape
    :return: reference to persistent (on-disk) zarr array
    Load a 3D tif image into a persistent zarr array. Does not load
    the entire tif image into memory.
    """
    with tifffile.TiffFile(tif_path) as tif:
        ij_metadata = tif.imagej_metadata
        first_slice = tif.asarray(key=0)

        nb_slices = ij_metadata['slices']
        shape = (nb_slices, *first_slice.shape)
        dtype = first_slice.dtype

        z_arr = zarr.open(zarr_path, mode='w', shape=shape, chunks=chunks, dtype=dtype)

        for i in range(nb_slices):
            img = tif.asarray(key=i)
            z_arr[i] = img
