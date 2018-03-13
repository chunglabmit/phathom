import tifffile
from . import utils
import zarr


def imread(path):
    """
    Reads TIF file into a numpy array in memory.
    :param path:
    :return:
    """
    return tifffile.imread(files=path)


def imsave(path, data):
    """
    Saves numpy array as a TIF image.
    """
    tifffile.imsave(file=path, data=data)


def tifs_to_zarr(tif_paths, zarr_path, chunks):
    """
    Loads 2D TIF images into a (chunked) Zarr array one-by-one
    """
    img = imread(tif_paths[0])
    shape = (len(tif_paths), *img.shape)
    dtype = img.dtype
    z_arr = zarr.open(zarr_path, mode='w', shape=shape, chunks=chunks, dtype=dtype)
    for z, tif_path in enumerate(tif_paths):
        z_arr[z] = imread(tif_path)
    return z_arr


def main():
    tif_paths, tif_filenames = utils.tifs_in_dir('data')
    print(len(tif_paths))
    tifs_to_zarr(tif_paths, 'data/data.zarr', (64, 128, 128))


if __name__ == '__main__':
    main()
