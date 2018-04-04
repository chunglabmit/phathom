"""
Conversion module
=================

Main functionality:

tiffs_to_zarr - save directory with 2D tifs into a chunked zarr array
tiff_to_zarr - save a 3D tif as a chunked zarr array
ims_to_tiffs - save the z-slices in a terastitcher Imaris file to tiffs
ims_to_zarr - save a terastitcher Imaris file into a chunked zarr array
"""
from . import utils
import skimage.external.tifffile as tifffile
import zarr
import numpy as np
import multiprocessing
import h5py
import os


def imread(path):
    """ Reads TIF file into a numpy array in memory.

    :param path: path to tif image to open
    :return: numpy ndarray with image data
    """
    return tifffile.imread(files=path)


def imsave(path, data):
    """ Saves numpy array as a TIF image.

    :param path: path to tif image to create / overwrite
    :param data: numpy ndarray with image data
    """
    tifffile.imsave(file=path, data=data)


def write_chunk(data, z_arr, start):
    stop = tuple(s+c for s,c in zip(start, data.shape))
    assert start[0] < stop[0]
    assert start[1] < stop[1]
    assert start[2] < stop[2]
    z_arr[start[0]:stop[0], start[1]:stop[1], start[2]:stop[2]] = data


def read_tifs(tif_paths):
    img = imread(tif_paths[0])
    shape = (len(tif_paths), *img.shape)
    data = np.zeros(shape, dtype=img.dtype)
    with multiprocessing.Pool(16) as pool:
        data = np.array(pool.map(imread, tif_paths))
    return data


def tifs_to_zarr_chunks(z_arr, tif_paths, start_list):
    data = read_tifs(tif_paths) # read tiffs in parallel

    args = []
    for start in start_list:
        stop = tuple(min(e, s+c) for e,s,c in zip(z_arr.shape, start, z_arr.chunks))
        chunk = data[:, start[1]:stop[1], start[2]:stop[2]]
        args.append((chunk, z_arr, start))

    with multiprocessing.Pool(16) as pool:
        pool.starmap(write_chunk, args) # write tiffs in parallel


def tifs_to_zarr(tif_dir, zarr_path, chunks, in_memory=False):
    """ Loads 2D TIF images into a (chunked) zarr array

    :param tif_dir: path to directory with 2D tifs to save as zarr array
    :param zarr_path: path of the zarr array to be created / overwritten
    :param chunks: tuple of ints specifying the chunk shape
    :return: reference to persistent (on-disk) zarr array
    """
    tif_paths, _ = utils.tifs_in_dir(tif_dir)
    img = imread(tif_paths[0])
    shape = (len(tif_paths), *img.shape)
    dtype = img.dtype
    # TODO: Swtich to producer-consumer model for tif reading -> zarr writing
    if in_memory:
        data = np.zeros(shape, dtype=dtype)
        for z, tif_path in enumerate(tif_paths):
            data[z] = imread(tif_path)
        zarr.save_array(zarr_path, data, chunks=chunks)
        return data
    else:
        z_arr = zarr.open(zarr_path, mode='w', shape=shape, chunks=chunks, dtype=dtype)
        nb_chunks = utils.chunk_dims(shape, chunks)
        start_list = utils.chunk_coordinates(shape, chunks)
        xy_chunks = nb_chunks[1] * nb_chunks[2]

        for z_chunk in range(nb_chunks[0]):
            z0 = z_chunk * chunks[0]
            z1 = min(shape[0], z0 + chunks[0])
            k0 = z_chunk * xy_chunks
            k1 = (z_chunk+1) * xy_chunks
            tifs_to_zarr_chunks(z_arr, tif_paths[z0:z1], start_list[k0:k1])


def tif_to_zarr(tif_path, zarr_path, chunks, in_memory=False):
    """ Load a 3D tif image into a persistent zarr array.

    :param tif_path: path to a 3D tif image
    :param zarr_path: path of the zarr array to be created / overwritten
    :param chunks: tuple of ints specifying the chunk shape
    :param in_memory: boolean to load whole image into memory
    :return: reference to persistent (on-disk) zarr array
    """
    if in_memory:
        data = imread(tif_path)
        zarr.save_array(zarr_path, data, chunks=chunks)
    else:
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


def slice_to_tiff(arr, idx, output_dir):
    img = arr[idx]
    filename = '{0:04d}.tif'.format(idx)
    path = os.path.join(output_dir, filename)
    imsave(path, img)


def zarr_to_tifs(zarr_path, output_dir, nb_workers=1, in_memory=False):
    """ Convert zarr array into tiffs

    :param zarr_path: path to input zarr array
    :param output_dir: path to output directory
    :param nb_workers: number of parallel processes for writing
    :param in_memory: boolean indicating if the zarr array should be loaded into memory
    :return:
    """
    utils.make_dir(output_dir)
    if in_memory:
        arr = zarr.load(zarr_path)
        for z in arr.shape[0]:
            slice_to_tiff(arr, z, output_dir)
    else:
        z_arr = zarr.open(zarr_path, mode='r')
        with multiprocessing.Pool(nb_workers) as pool:
            args = []
            for z in range(z_arr.shape[0]):
                args.append((z_arr, z, output_dir))
            pool.starmap(slice_to_tiff, args)


def ims_slice_to_tiff(z, ims_file, output_path):
    """ Save a slice from an Imaris file from terastitcher as a tiff.
    Only works with a single timepoint and channel.

    :param z: index of the z slice
    :param ims_file: path to Imaris file
    :param output_path: path to write the slice to (including .tif)
    """
    with h5py.File(ims_file, "r") as fd:
        # This group structure is default from Terastitcher
        img = fd['DataSet/ResolutionLevel 0/TimePoint 0/Channel 0/Data'][z]
    imsave(output_path, img)


def ims_to_tiffs(ims_file, output_dir, nb_workers):
    """ Save an Imaris file to individual tiffs

    :param ims_file: path to Imaris file
    :param output_dir: path to output directory
    :param nb_workers: number of threads for reading and writing
    """
    utils.make_dir(output_dir)

    with h5py.File(ims_file, "r") as ims_f:
        dset = ims_f['DataSet/ResolutionLevel 0/TimePoint 0/Channel 0/Data']
        shape = dset.shape
        dtype = dset.dtype

    with multiprocessing.Pool(nb_workers) as pool:
        args = []
        for z in range(shape[0]):
            filename = '{0:04d}.tif'.format(z)
            output_path = os.path.join(output_dir, filename)
            args.append((z, ims_file, output_path))
        # Save slices in parallel
        pool.starmap(ims_slice_to_tiff, args)


def ims_chunk_to_zarr(ims_path, start, z_arr):
    """ Save a slice from an Imaris file from terastitcher as a tiff.
    Only works with a single timepoint and channel.

    :param ims_path: path to Imaris file
    :param idx: a tuple of starting indices of the chunk to write
    :param z_arr: reference to on-disk zarr array
    """
    shape = z_arr.shape
    chunks = z_arr.chunks

    stop = tuple(min(i+j, limit-1) for i,j,limit in zip(start, chunks, shape))
    z1, y1, x1 = start
    z2, y2, x2 = stop

    with h5py.File(ims_path, "r") as fd:
        dset = fd['DataSet/ResolutionLevel 0/TimePoint 0/Channel 0/Data']
        z_arr[z1:z2, y1:y2, x1:x2] = dset[z1:z2, y1:y2, x1:x2]


def ims_to_zarr(ims_path, zarr_path, chunks, nb_workers):
    """ Save an Imaris file as a chunked zarr array

    :param ims_path: path to input Imaris file
    :param zarr_path: path to output zarr array
    :param chunks: shape of individual chunks in zarr array
    :param nb_workers: number of processes that are reading / writing
    """

    with h5py.File(ims_path, "r") as ims_f:
        dset = ims_f['DataSet/ResolutionLevel 0/TimePoint 0/Channel 0/Data']
        shape = dset.shape
        dtype = dset.dtype

    z_arr = zarr.open(zarr_path, mode='w', shape=shape, chunks=chunks, dtype=dtype)

    with multiprocessing.Pool(nb_workers) as pool:

        args = []
        for chunk_dim in utils.chunk_dims(shape, chunks):
            start_list = []
            for idx, size in zip(chunk_dim, chunks):
                start_list.append(idx*size)
            start = tuple(start_list)
            args.append((ims_path, start, z_arr))

        # Save chunks in parallel
        pool.starmap(ims_chunk_to_zarr, args)


def main():
    tif_dir = 'D:/Justin/coregistration/fixed/C0/zslices'
    zarr_path = 'D:/Justin/coregistration/fixed/C0/fixed.zarr'
    chunks = (256, 512, 512)
    # tif_path = 'D:/Justin/coregistration/moving/C0/roi1_registered_tifs/'
    nb_workers = 44

    # tifs_to_zarr(tif_dir, zarr_path, chunks)

    zarr_path = 'D:/Justin/coregistration/moving/C0/roi1_registered3.zarr'
    tif_path = 'D:/Justin/coregistration/moving/C0/roi1_registered3_tifs/'
    # zarr_to_tifs(zarr_path, tif_path, nb_workers)




if __name__ == '__main__':
    main()
