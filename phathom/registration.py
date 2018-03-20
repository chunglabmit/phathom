from itertools import product
from functools import partial
import numpy as np
import zarr
from skimage import feature, filters
import multiprocessing
from . import utils
from . import pcloud



def chunk_coordinates(shape, chunks):
    """ Calculate the global coordaintes for each chunk's starting position

    :param shape: shape of the image to chunk
    :param chunks: shape of each chunk
    :return: a list containing the starting indices of each chunk
    """
    nb_chunks = utils.chunk_dims(shape, chunks)
    start = []
    for indices in product(*tuple(range(n) for n in nb_chunks)):
        start.append(tuple(i*c for i, c in zip(indices, chunks)))
    return start


def chunk_bboxes(shape, chunks, overlap):
    """ Calculates the bounding box coordinates for each overlapped chunk

    :param shape: overall shape
    :param chunks: tuple containing the shape of each chunk
    :param overlap: int indicating number of voxels to overlap adjacent chunks
    :return: tuple containing the start and stop coordinates for each bbox
    """
    chunk_coords = chunk_coordinates(shape, chunks)
    start = []
    stop = []
    for coord in chunk_coords:
        start.append(tuple(max(0, s-overlap) for s in coord))
        stop.append(tuple(min(e, s+c+overlap) for s,c,e in zip(coord, chunks, shape)))
    return chunk_coords, start, stop


def chunk_generator(z_arr, overlap):
    """ Reads the chunks from a 3D on-disk zarr array

    :param z_arr: input zarr array
    :param overlap: int indicating number of voxels to overlap adjacent chunks
    :return: the next chunk in the zarr array
    """
    _, starts, stops = chunk_bboxes(z_arr.shape, z_arr.chunks, overlap)
    for start, stop in zip(starts, stops):
        z0, y0, x0 = start
        z1, y1, x1 = stop
        yield z_arr[z0:z1, y0:y1, x0:x1]


def detect_blobs(img, sigma):
    """ Detects blobs in an image using local maxima

    :param z_arr: input zarr array
    :param sigma: float for gaussian blurring
    :return: an (N,3) ndarray of blob coordinates
    """
    smoothed = filters.gaussian(img, sigma=sigma)
    peaks = feature.peak_local_max(smoothed, min_distance=2, threshold_abs=smoothed.mean())
    # Note that using mean here can introduce from chunking artifacts
    return peaks


def detect_blobs_parallel(z_arr, sigma, nb_workers, overlap):
    """ Detects blobs in a chunked zarr array in parallel using local maxima

    :param z_arr: input zarr array
    :param sigma: float for gaussian blurring
    :return: an (N,3) ndarray of blob coordinates
    """
    chunk_coords, starts, _ = chunk_bboxes(z_arr.shape, z_arr.chunks, overlap)

    detect_blobs_in_chunk = partial(detect_blobs, sigma=sigma)
    chunk_gen = chunk_generator(z_arr, overlap)

    pts_list = []
    with multiprocessing.Pool(nb_workers) as pool:
        print('Running blob detection with {} workers'.format(nb_workers))

        for i, pts_local in enumerate(pool.imap(detect_blobs_in_chunk, chunk_gen)):
            chunk_coord = np.array(chunk_coords[i])
            start = np.array(starts[i])

            local_start = chunk_coord - start
            local_stop = local_start + np.array(z_arr.chunks)

            idx = np.all(np.logical_and(local_start <= pts_local, pts_local < local_stop), axis=1)
            pts_trim = pts_local[idx]

            pts_list.append(pts_trim + start)

    return np.concatenate(pts_list)


def pts_to_img(pts, shape, path):
    """ Saves a set of points into a binary image

    :param pts: an (N, D) array with N D-dimensional points
    :param shape: a tuple describing the overall shape of the image
    :param path: path to save the output tiff image
    :return: a uint8 ndarray
    """
    from skimage.external import tifffile
    img = np.zeros(shape, dtype='uint8')
    img[tuple(pts.T)] = 255
    tifffile.imsave(path, img)
    return img


def main():
    fixed_zarr_path = 'D:/Justin/coregistration/fixed/C0/roi3.zarr'
    moving_zarr_path = 'D:/Justin/coregistration/moving/C0/roi3.zarr'
    sigma = (1, 2.0, 2.0)
    nb_workers = 44
    overlap = 8

    fixed_img = zarr.open(fixed_zarr_path)
    moving_img = zarr.open(moving_zarr_path)

    fixed_pts = detect_blobs_parallel(fixed_img, sigma, nb_workers, overlap)
    # moving_pts = detect_blobs_parallel(moving_img, sigma, nb_workers, overlap)

    print('extracting features')
    fixed_features = pcloud.geometric_features(fixed_pts, nb_workers)
    # moving_features = pcloud.geometric_features(moving_pts, nb_workers)



    # fixed_blobs = detect_blobs(fixed_img)

    # Read a chunk, detect blobs, return blob locations in global coordiantes
    # img in, list or ndarray out

    # Parallel processing
    # Generate


if __name__ == '__main__':
     