"""Filtering Module

This module provides some basic filters that are useful in cleaning up data

"""
from functools import partial
import multiprocessing
import os
import numpy as np
from skimage.exposure import equalize_adapthist
from skimage.filters import gaussian
import tqdm
from phathom import io
from phathom.utils import tifs_in_dir, make_dir, extract_box, extract_ghosted_chunk, insert_box, pmap_chunks


def clahe_2d(image, kernel_size, clip_limit=0.01, nbins=256):
    return equalize_adapthist(image, kernel_size, clip_limit, nbins)


def clahe(image, kernel_size, clip_limit=0.01, nbins=256, nb_workers=None):
    """Apply CLAHE to each z-slice in `image`

    Parameters
    ----------
    image : ndarray
        input image
    kernel_size : int or list-like
        shape of the contextual regions
    clip_limit : float, optional
        limit for number of clipping pixels
    nbins : int, optional
        number of gray bins for histograms
    nb_workers : int, optional
        number of workers to use. Default, cpu_count

    Returns
    -------
    equalized : ndarray
        output image of float32s scaled to [0, 1]

    """
    if nb_workers is None:
        nb_workers = multiprocessing.cpu_count()
    f = partial(equalize_adapthist,
                kernel_size=kernel_size,
                clip_limit=clip_limit,
                nbins=nbins)
    with multiprocessing.Pool(nb_workers) as pool:
        results = list(tqdm.tqdm(pool.imap(f, image), total=image.shape[0]))
    return np.asarray(results)


def remove_background(image, threshold):
    """Threshold an image and use as a mask to remove the background. Used for improved compression

    Parameters
    ----------
    image : ndarray
        input image
    threshold : float
        intensity to threshold the input image

    Returns
    -------
    filtered : ndarray
        output image with background set to 0

    """
    mask = (image >= threshold)
    return image * mask


def gaussian_blur(image, sigma):
    return gaussian(image, sigma, preserve_range=True)


def _gaussian_blur_chunk(input_tuple, sigma, output, overlap):
    arr, start_coord, chunks = input_tuple
    ghosted_chunk, start_ghosted, stop_ghosted = extract_ghosted_chunk(arr,
                                                                       start_coord,
                                                                       chunks,
                                                                       overlap)
    g = gaussian_blur(ghosted_chunk, sigma)
    start_local = start_coord - start_ghosted
    stop_local = np.minimum(start_local + np.asarray(chunks),
                            np.asarray(ghosted_chunk.shape))
    g_valid = extract_box(g, start_local, stop_local)

    stop_coord = start_coord + np.asarray(g_valid.shape)
    insert_box(output, start_coord, stop_coord, g_valid)


def gaussian_blur_parallel(arr, sigma, output, chunks, overlap, nb_workers=None):
    f = partial(_gaussian_blur_chunk, sigma=sigma, output=output, overlap=overlap)
    pmap_chunks(f, arr, chunks, nb_workers=nb_workers, use_imap=True)


def preprocess(tif_path, output_path, threshold=None, kernel_size=127, clip_limit=0.01):
    img = io.tiff.imread(tif_path)
    img_max = img.max()
    img_min = img.min()
    enhanced_normalized = clahe_2d(img, kernel_size, clip_limit)
    enhanced = enhanced_normalized * (img_max - img_min) + img_min
    if threshold is not None:
        output = enhanced * (img >= threshold)
    else:
        output = enhanced
    io.tiff.imsave(output_path, output.astype(img.dtype), compress=1)


def preprocess_batch(path_to_tifs, output_path, threshold=None, kernel_size=127, clip_limit=0.01, nb_workers=None):
    if nb_workers is None:
        nb_workers = multiprocessing.cpu_count()

    paths, filenames = tifs_in_dir(path_to_tifs)
    output_abspath = make_dir(output_path)

    args_list = []
    for path, filename in zip(paths, filenames):
        output_path = os.path.join(output_abspath, filename)
        args = (path, output_path, threshold, kernel_size, clip_limit)
        args_list.append(args)

    with multiprocessing.Pool(nb_workers) as pool:
        pool.starmap(preprocess, args_list)


def main():
    path_to_tifs = '/media/jswaney/Drive/Justin/organoid_etango/syto16'
    output_path = '/media/jswaney/Drive/Justin/organoid_etango/syto16_clahe'
    nb_workers = 12
    threshold = 300
    kernel_size = 127
    clip_limit = 0.005

    # paths, filenames = tifs_in_dir(path_to_tifs)
    #
    # output_abspath = make_dir(output_path)
    #
    # args_list = []
    # for path, filename in zip(paths, filenames):
    #     output_path = os.path.join(output_abspath, filename)
    #     args = (path, output_path, threshold, kernel_size)
    #     args_list.append(args)
    #
    # with multiprocessing.Pool(nb_workers) as pool:
    #     pool.starmap(preprocess, args_list)

    #####

    path_to_tifs = '/media/jswaney/Drive/Justin/organoid_etango/sox2'
    output_path = '/media/jswaney/Drive/Justin/organoid_etango/sox2_clahe'

    paths, filenames = tifs_in_dir(path_to_tifs)

    output_abspath = make_dir(output_path)

    args_list = []
    for path, filename in zip(paths, filenames):
        output_path = os.path.join(output_abspath, filename)
        args = (path, output_path, threshold, kernel_size, clip_limit)
        args_list.append(args)

    with multiprocessing.Pool(nb_workers) as pool:
        pool.starmap(preprocess, args_list)


if __name__ == '__main__':
    main()
