"""This module contains functions for cell body detection in 3D for single-cell analysis
"""

import os
import multiprocessing
from functools import partial
from tqdm import tqdm_notebook as tqdm
import numpy as np
from scipy.ndimage import label
import skimage
from skimage.transform import integral_image
from skimage.feature import haar_like_feature, hessian_matrix, hessian_matrix_eigvals
from skimage.filters import threshold_otsu, gaussian
import matplotlib.pyplot as plt
from phathom import io
from phathom.segmentation.segmentation import find_centroids
from phathom.utils import pmap_chunks, extract_box, insert_box

from phathom.segmentation.segmentation import eigvals_of_weingarten  # will use torch if available


def detect_blobs(image, sigma):
    image = skimage.img_as_float32(image)
    g = gaussian(image, sigma=sigma)
    eigvals = eigvals_of_weingarten(g)
    eigvals_clipped = np.clip(eigvals, None, 0)
    threshold = -threshold_otsu(-eigvals_clipped)
    loc = np.where(eigvals_clipped[..., -1] < threshold)
    mask = np.zeros(image.shape, np.int)
    mask[loc] = 1
    return mask


def _detect_blobs_chunk(arr, start_coord, chunks, arr_out, sigma, min_intensity, overlap=4):
    # extract ghosted chunk of data
    end_coord = np.minimum(arr.shape, start_coord + np.asarray(chunks))
    chunk_shape = end_coord - start_coord
    start_overlap = np.maximum(np.zeros(arr.ndim, 'int'),
                               np.array([s - overlap for s in start_coord]))
    stop_overlap = np.minimum(arr.shape, np.array([e + overlap for e in end_coord]))
    data_overlap = extract_box(arr, start_overlap, stop_overlap)

    if data_overlap.max() < min_intensity:
        insert_box(arr_out, start_coord, end_coord, np.zeros(chunk_shape, arr_out.dtype))
    else:
        blob_mask = detect_blobs(data_overlap, sigma)
        insert_box(arr_out, start_coord, end_coord, blob_mask)


def detect_blobs_parallel(arr_in, arr_out, sigma, min_intensity, nb_workers=None, chunks=None, overlap=4):
    f = partial(_detect_blobs_chunk,
                arr_out=arr_out,
                sigma=sigma,
                min_intensity=min_intensity,
                overlap=overlap)
    pmap_chunks(f, arr_in, chunks=chunks, nb_workers=nb_workers)
