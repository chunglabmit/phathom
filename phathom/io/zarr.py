import multiprocessing
from itertools import product
import numpy as np
import zarr
from numcodecs import Blosc
from skimage.transform import downscale_local_mean
from skimage.measure import block_reduce
from skimage.util import pad
import tqdm
from phathom import utils


def open(zarr_path):
    return zarr.open(zarr_path)


def new_zarr(path, shape, chunks, dtype, **kwargs):
    compressor = Blosc(cname='zstd', clevel=1, shuffle=Blosc.BITSHUFFLE)
    z_arr_out = zarr.open(path,
                          mode='w',
                          shape=shape,
                          chunks=chunks,
                          dtype=dtype,
                          compressor=compressor,
                          **kwargs)
    return z_arr_out



def write_subarray(data, z_arr, start):
    """ Write a subarray into a zarr array.

    :param data: An array of data to write
    :param z_arr: A reference to a persistent zarr array (with write access)
    :param start: Array-like containing the 3 indices of the starting coordinate
    """
    stop = tuple(s+c for s, c in zip(start, data.shape))
    assert start[0] < stop[0]
    assert start[1] < stop[1]
    assert start[2] < stop[2]
    z_arr[start[0]:stop[0], start[1]:stop[1], start[2]:stop[2]] = data


# output_img = zarr.open(registered_zarr_path,
#                        mode='w',
#                        shape=fixed_img.shape,
#                        chunks=fixed_img.chunks,
#                        dtype=fixed_img.dtype,
#                        compressor=Blosc(cname='zstd', clevel=1, shuffle=Blosc.BITSHUFFLE))


def downsample_chunk(args):
    """ Downsample a zarr array chunk and write it to another zarr array as a smaller chunk

    :param z_arr_in: input zarr array
    :param z_arr_out: ouput zarr array
    :param chunk_idx: array-like of chunk indices
    :param factors: array-like of float downsampling factors
    :return:
    """""
    z_arr_in, z_arr_out, chunk_idx, factors = args

    in_start = np.array([i*c for i, c in zip(chunk_idx, z_arr_in.chunks)])
    in_stop = np.array([(i+1)*c if (i+1)*c < s else s for i, c, s in zip(chunk_idx, z_arr_in.chunks, z_arr_in.shape)])

    data = z_arr_in[in_start[0]:in_stop[0], in_start[1]:in_stop[1], in_start[2]:in_stop[2]]

    # down_data = downscale_local_mean(image=data, factors=factors)
    down_data = block_reduce(data, factors, np.max, 0)

    if down_data != z_arr_out.chunks:
        pad_width = [(0, c-s) for s, c in zip(down_data.shape, z_arr_out.chunks)]
        down_data = np.pad(down_data, pad_width, 'constant')

    out_start = np.array([i*c for i, c in zip(chunk_idx, z_arr_out.chunks)])
    out_stop = np.array([(i+1)*c if (i+1)*c < s else s for i, c, s in zip(chunk_idx, z_arr_out.chunks, z_arr_out.shape)])

    z_arr_out[out_start[0]:out_stop[0], out_start[1]:out_stop[1], out_start[2]:out_stop[2]] = down_data


def downsample_zarr(z_arr_in, factors, output_path, nb_workers=1, **kwargs):
    """ Downsample a chunked zarr array in parallel

    :param z_arr_in: input zarr array
    :param factors: tuple of floats with downsample factors
    :param output_path: path for the output zarr array
    :param kwargs: additional keyword arguments passed to zarr.open()
    :param nb_workers: number of workers to downsample in parallel
    :return:
    """

    if kwargs.pop('chunks', None) is not None:
        raise ValueError('chunks will be determined by the downsampling factors')

    compressor = kwargs.pop('compressor', None)
    if compressor is None:
        # Use Blosc compressor as default
        compressor = Blosc(cname='zstd', clevel=1, shuffle=Blosc.BITSHUFFLE)

    if z_arr_in.chunks is None:
        raise ValueError('input zarr array must contain chunks')

    chunks = np.array([int(np.ceil(i/f)) for i, f in zip(z_arr_in.chunks, factors)])
    nb_chunks = utils.chunk_dims(z_arr_in.shape, z_arr_in.chunks)
    shape = np.asarray(chunks) * np.asarray(nb_chunks)

    z_arr_out = zarr.open(output_path,
                          mode='w',
                          shape=shape,
                          chunks=chunks,
                          dtype=z_arr_in.dtype,
                          compressor=compressor,
                          **kwargs)

    args = []
    for chunk_idx in product(*tuple(range(n) for n in nb_chunks)):
        args.append((z_arr_in, z_arr_out, chunk_idx, factors))

    with multiprocessing.Pool(processes=nb_workers) as pool:
        list(tqdm.tqdm(pool.imap(downsample_chunk, args), total=len(args)))


# TODO: fix issues with non-integer downsampling factors
def downsample_voxel_dims(z_arr_in, voxel_dims_old, voxel_dims_new, output_path, nb_workers=1, **kwargs):
    factors = np.asarray(voxel_dims_new) / np.asarray(voxel_dims_old)
    downsample_zarr(z_arr_in, factors, output_path, nb_workers, **kwargs)


# TODO: this can probably also use io.zarr.write_chunk
def fetch_and_write_chunk(source, dest, dest_chunk_idx):
    start = np.array(dest_chunk_idx) * np.array(dest.chunks)
    stop = np.array(
        [(i+1)*c if (i+1)*c <= s else s for i, c, s in zip(dest_chunk_idx, dest.chunks, dest.shape)])
    data = source[start[0]:stop[0], start[1]:stop[1], start[2]:stop[2]]
    write_subarray(data, dest, start)


def rechunk_zarr(source, dest, nb_workers):
    """ Copy a chunked zarr array to a new zarr array with different parameters

    :param source: source zarr array
    :param dest: destination zarr array
    :param nb_workers: number of workers to write in parallel
    :return:
    """
    nb_chunks = utils.chunk_dims(dest.shape, dest.chunks)

    args = []
    for chunk_idx in product(*tuple(range(n) for n in nb_chunks)):
        args.append((source, dest, chunk_idx))

    with multiprocessing.Pool(nb_workers) as pool:
        pool.starmap(fetch_and_write_chunk, args)


