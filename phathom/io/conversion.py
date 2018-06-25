"""Conversion

This module provides functions for converting between large image formats.
Conversion between the following file formats is supported:
- Tiff
- Zarr
- HDF5 / Imaris

Main functionality:

tiffs_to_zarr - save directory with 2D tifs into a chunked zarr array
tiff_to_zarr - save a 3D tif as a chunked zarr array
ims_to_tiffs - save the z-slices in a terastitcher Imaris file to tiffs
ims_to_zarr - save a terastitcher Imaris file into a chunked zarr array
downsample_zarr - downsample a chunked zarr array in parallel
rechunk_zarr - copy a zarr array with new parameters
"""
from phathom import utils
from phathom import io
import skimage.external.tifffile as tifffile
import zarr
from numcodecs import Blosc
import numpy as np
import multiprocessing
import h5py
import os
from itertools import product
from sys import stdout
import tqdm


def tifs_to_zarr_chunks(z_arr, tif_paths, start_list, nb_workers):
    data = io.tiff.imread_parallel(tif_paths, nb_workers)  # read tiffs in parallel

    args = []
    for start in start_list:
        stop = tuple(min(e, s+c) for e,s,c in zip(z_arr.shape, start, z_arr.chunks))
        chunk = data[:, start[1]:stop[1], start[2]:stop[2]]
        args.append((chunk, z_arr, start))

    with multiprocessing.Pool(nb_workers) as pool:
        pool.starmap(io.zarr.write_subarray, args)  # write tiffs in parallel


def tifs_to_zarr(tif_dir, zarr_path, chunks, in_memory=False, nb_workers=1):
    """ Loads 2D TIF images into a (chunked) zarr array

    :param tif_dir: path to directory with 2D tifs to save as zarr array
    :param zarr_path: path of the zarr array to be created / overwritten
    :param chunks: tuple of ints specifying the chunk shape
    :param in_memory: bool indicating whether or not to load the tifs into memory
    :param nb_workers: int of workers for parallel io
    :return: reference to persistent (on-disk) zarr array
    """
    compressor = Blosc(cname='zstd', clevel=1, shuffle=Blosc.BITSHUFFLE)
    tif_paths, _ = utils.tifs_in_dir(tif_dir)
    img = io.tiff.imread(tif_paths[0])
    shape = (len(tif_paths), *img.shape)
    dtype = img.dtype
    # TODO: Swtich to producer-consumer model for tif reading -> zarr writing
    if in_memory:
        data = np.zeros(shape, dtype=dtype)
        for z, tif_path in enumerate(tif_paths):
            data[z] = io.tiff.imread(tif_path)
        zarr.save_array(zarr_path, data, chunks=chunks, compression=compressor)
        return data
    else:
        z_arr = zarr.open(zarr_path, mode='w', shape=shape, chunks=chunks, dtype=dtype, compression=compressor)
        nb_chunks = utils.chunk_dims(shape, chunks)
        start_list = utils.chunk_coordinates(shape, chunks)
        xy_chunks = nb_chunks[1] * nb_chunks[2]

        for z_chunk in range(nb_chunks[0]):
            z0 = z_chunk * chunks[0]
            z1 = min(shape[0], z0 + chunks[0])
            k0 = z_chunk * xy_chunks
            k1 = (z_chunk+1) * xy_chunks
            tifs_to_zarr_chunks(z_arr, tif_paths[z0:z1], start_list[k0:k1], nb_workers)


def tif_to_zarr(tif_path, zarr_path, chunks, in_memory=False, **kwargs):
    """ Load a 3D tif image into a persistent zarr array.

    :param tif_path: path to a 3D tif image
    :param zarr_path: path of the zarr array to be created / overwritten
    :param chunks: tuple of ints specifying the chunk shape
    :param in_memory: boolean to load whole image into memory
    :return: reference to persistent (on-disk) zarr array
    """
    if in_memory:
        data = io.tiff.imread(tif_path)
        zarr.save_array(zarr_path, data, chunks=chunks, **kwargs)
    else:
        with tifffile.TiffFile(tif_path) as tif:
            ij_metadata = tif.imagej_metadata
            first_slice = tif.asarray(key=0)

            nb_slices = ij_metadata['slices']
            shape = (nb_slices, *first_slice.shape)
            dtype = first_slice.dtype

            z_arr = zarr.open(zarr_path, mode='w', shape=shape, chunks=chunks, dtype=dtype, **kwargs)

            for i in range(nb_slices):
                img = tif.asarray(key=i)
                z_arr[i] = img


def slice_to_tiff(arr, idx, output_dir, compress):
    """ Save a slice of a zarr array to disk

    :param arr: Input zarr array
    :param idx: z-slice index to save
    :param output_dir: output directory
    :param compress: degree of lossless tiff compresion
    :return:
    """
    img = arr[idx]
    filename = '{0:04d}.tif'.format(idx)
    path = os.path.join(output_dir, filename)
    io.tiff.imsave(path, img, compress=compress)


def zarr_to_tifs(zarr_path, output_dir, nb_workers=1, compress=0, in_memory=False):
    """ Convert zarr array into tiffs

    :param zarr_path: path to input zarr array
    :param output_dir: path to output directory
    :param nb_workers: number of parallel processes for writing
    :param compress: degree of lossless tiff compression
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
                args.append((z_arr, z, output_dir, compress))
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
    io.tiff.imsave(output_path, img)


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
    tif_dir = '/media/share2/Justin/organoid/syto16_remount'
    zarr_path = '/media/share2/Justin/organoid/syto16_remount.zarr'
    chunks = (256, 512, 512)
    # tif_path = 'D:/Justin/coregistration/moving/C0/roi1_registered_tifs/'
    nb_workers = 4

    tifs_to_zarr(tif_dir, zarr_path, chunks, nb_workers=12)

    # zarr_path = '/media/jswaney/Drive/Justin/coregistration/whole_brain/nonrigid.zarr'
    # tif_path = '/media/jswaney/Drive/Justin/coregistration/whole_brain/nonrigid.tifs'
    # zarr_to_tifs(zarr_path, tif_path, nb_workers=12)

    # source_zarr_path = 'D:/Justin/coregistration/fixed/C0/fixed.zarr'
    # dest_zarr_path = 'D:/Justin/coregistration/fixed/C0/fixed_int.zarr'
    # from numcodecs import Blosc

    # downsample_zarr(zarr.open(zarr_path), (8, 8, 8), downsample_path, nb_workers=12, compression=Zstd(level=1))
    #
    # tif_path = 'D:/Justin/coregistration/moving/C0/roi1_downsampled8'
    # zarr_to_tifs(downsample_path, tif_path, nb_workers=8)

    # source_zarr_path = 'D:/Justin/coregistration/fixed/C0/fixed.zarr'
    # dest_zarr_path = 'D:/Justin/coregistration/fixed/C0/fixed_int.zarr'
    #
    # source = zarr.open(source_zarr_path, mode='r')
    # dest = zarr.open(dest_zarr_path,
    #                  mode='w',
    #                  shape=source.shape,
    #                  chunks=source.chunks,
    #                  dtype='uint16',
    #                  compression=Blosc(cname='zstd', clevel=1, shuffle=Blosc.BITSHUFFLE))
    #
    # import time
    # t0 = time.time()
    # rechunk_zarr(source, dest, 44)
    # print('Elapsed time: {sec:.1f} seconds'.format(sec=time.time()-t0))


if __name__ == '__main__':
    main()
