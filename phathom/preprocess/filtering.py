"""Filtering Module

This module provides some basic filters that are useful in cleaning up data

"""
from functools import partial
import multiprocessing
import numpy as np
from skimage.exposure import equalize_adapthist
import tqdm


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
