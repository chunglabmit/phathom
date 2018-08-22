"""Filtering Module

This module provides some basic filters that are useful in cleaning up data

"""
from functools import partial
import multiprocessing
import os
import numpy as np
from skimage.exposure import equalize_adapthist
import tqdm
from phathom import io
from phathom.utils import tifs_in_dir, make_dir


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


def preprocess(tif_path, output_path, threshold=None, kernel_size=127):
    img = io.tiff.imread(tif_path)
    img_max = img.max()
    img_min = img.min()
    enhanced_normalized = clahe_2d(img, kernel_size)
    enhanced = enhanced_normalized * (img_max - img_min) + img_min
    output = enhanced
    # if threshold is not None:
    # mask = (img >= threshold)
    # output = enhanced * mask
    # output = remove_background(img, threshold)
    io.tiff.imsave(output_path, output.astype(img.dtype), compress=1)


def main():
    path_to_tifs = '/media/jswaney/Drive/Justin/coregistration/whole_brain_tde/fixed/processed_tiffs'
    output_path = '/media/jswaney/Drive/Justin/coregistration/whole_brain_tde/fixed/processed_tiffs2'
    nb_workers = 12
    threshold = 300
    kernel_size = 127

    paths, filenames = tifs_in_dir(path_to_tifs)

    output_abspath = make_dir(output_path)

    args_list = []
    for path, filename in zip(paths, filenames):
        output_path = os.path.join(output_abspath, filename)
        args = (path, output_path, threshold, kernel_size)
        args_list.append(args)

    with multiprocessing.Pool(nb_workers) as pool:
        pool.starmap(preprocess, args_list)

    #####

    path_to_tifs = '/media/jswaney/Drive/Justin/coregistration/whole_brain_tde/moving/processed_tiffs'
    output_path = '/media/jswaney/Drive/Justin/coregistration/whole_brain_tde/moving/processed_tiffs2'

    paths, filenames = tifs_in_dir(path_to_tifs)

    output_abspath = make_dir(output_path)

    args_list = []
    for path, filename in zip(paths, filenames):
        output_path = os.path.join(output_abspath, filename)
        args = (path, output_path, threshold, kernel_size)
        args_list.append(args)

    with multiprocessing.Pool(nb_workers) as pool:
        pool.starmap(preprocess, args_list)


if __name__ == '__main__':
    main()
