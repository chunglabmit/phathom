import pandas as pd
import numpy as np
from scipy.ndimage import labeled_comprehension, label
from skimage.transform import resize
from skimage.filters import gaussian
from skimage.morphology import binary_erosion, binary_dilation
from sklearn.neighbors import NearestNeighbors
from functools import partial


def create_dataframe(data, index, columns):
    """Create a dataframe from a numpy array

    Parameters
    ----------
    index
    data
    columns

    Returns
    -------

    """
    df = pd.DataFrame(data, index=index, columns=columns)
    return df


def merge_dataframes(dfs, on):
    """Merges a list of dataframes on a particular key

    Parameters
    ----------
    dfs : List
    on : str

    Returns
    -------

    """
    merged_df = dfs.pop()
    for df in dfs:
        merged_df = merged_df.merge(df, on=on)
    return merged_df


def find_small_nuclei_idx(df, min_vol):
    return np.where(df['volume'] < min_vol)


def find_small_nuclei(df, min_vol):
    idx = find_small_nuclei_idx(df, min_vol)
    return df.iloc[idx]


def find_large_nuclei_idx(df, max_vol):
    return np.where(df['volume'] > max_vol)


def find_large_nuclei(df, min_vol):
    idx = find_large_nuclei_idx(df, min_vol)
    return df.iloc[idx]


def select_value_range(df, name, min_val=None, max_val=None):
    """Filters out entries with values of `name` below `min_vol` and above `max_vol`

    Parameters
    ----------
    df
    name
    min_vol
    max_vol

    Returns
    -------
    df_filtered

    """
    if min_val is None:
        min_val = -np.inf
    else:
        small_idx = find_small_nuclei(df, min_val)
        print('Discarding {} small items'.format(len(small_idx)))

    if max_val is None:
        max_val = np.inf
    else:
        large_idx = find_large_nuclei(df, max_val)
        print('Discarding {} large items'.format(len(large_idx)))

    idx = np.where(np.logical_and(df[name] < max_val, df[name] > min_val))
    return df.iloc[idx]


def filter_label_image(label_image, labels):
    """Filters a label image by keeping the specified `labels` if they are present

    Parameters
    ----------
    label_image : ndarray
    labels : ndarray

    Returns
    -------
    label_image_filtered : ndarray

    """
    indicators = np.in1d(label_image, labels)
    mask = np.reshape(indicators, label_image.shape)
    return label_image * mask


def upsample_segmentation(coarse_seg, shape, sigma):
    """Upsample a segmentation with smoothing

    Parameters
    ----------
    coarse_seg : ndarray
        segmentation to be upsampled
    shape : tuple
        desired shape of the output
    sigma : float
        amount of gaussian smoothing to apply to the segmentation

    Returns
    -------
    seg : ndarray
        upsampled and smoothed segmentation. May contain new labels

    """
    upsample = resize(coarse_seg, shape, order=0, mode='constant', anti_aliasing=True)
    smooth = gaussian(upsample, sigma=sigma, preserve_range=True)
    binary = (smooth > smooth.max() / 2)
    seg, nb_labels = label(binary)
    return seg, nb_labels


def find_bounding_box(mask):
    """Finds the bounding box start and stop indices for a binary mask

    Parameters
    ----------
    mask : ndarray
        binary mask

    Returns
    -------
    start : ndarray
        starting index for bbox
    stop : ndarray
        ending index for bbox

    Raises
    ------
    ValueError
        raised if the proved mask has no objects

    """
    z, y, x = np.where(mask > 0)
    if z.size == 0:
        raise ValueError('the provided mask does not contain any objects')
    start = np.array([z.min(), y.min(), x.min()])
    stop = np.array([z.max()+1, y.max()+1, x.max()+1])
    return start, stop


def select_label(labels, l, binary=False):
    """Returns a labels image containing only the `l` component

    Parameters
    ----------
    labels : ndarray
        labels image to query
    l : int
        label value to select
    binary : bool
        flag for returning a binary mask

    Returns
    -------
    output : ndarray
        label or binary image with the `l` component only

    """
    idx = np.where(labels == l)
    output = np.zeros_like(labels)
    if binary:
        output[idx] = 1
    else:
        output[idx] = l
    return output


def get_mask_bounary(mask, inner=True, selem=None):
    """Exracts the boundary of a mask using morphological erosion or dilation

    Parameters
    ----------
    mask : ndarray
        binary mask of a solid
    inner : bool, optional
        indicates whether to find the inner or outer boundary. Default, True
    selem : ndarray, optional
        structure element for erosion or dilation. Default is 6 connected.

    Returns
    -------
    bounary : ndarray
        binary mask of the solid's boundary

    """
    if inner:
        boundary = mask - binary_erosion(mask, selem=selem)
    else:
        boundary = binary_dilation(mask, selem=selem) - mask
    return boundary


def distance_to_object(query_pts, object_pts):
    """Calculates the distance of all `query_pts` to the nearest `object_pt`

    Parameters
    ----------
    query_pts : ndarray
    object_pts : ndarray

    Returns
    -------
    distances : ndarray

    """
    nbrs = NearestNeighbors(n_neighbors=1).fit(object_pts)
    distances, _ = nbrs.kneighbors(query_pts)
    return distances.ravel()


def distances_to_neighbors(pts, n):
    """Calculates the distances of `pts` to the nearest `n` neighbors

    Parameters
    ----------
    pts : ndarray
        coordinates of M points, probably should be in micron
    n : int
        number of nearest neighbors to find

    Returns
    -------
    distances : ndarray
        (M, n) array of distance
    indices : ndarray
        (M, n) array of indices of the nearest n neighbors in order

    """
    nbrs = NearestNeighbors(n_neighbors=n+1).fit(pts)  # n+1 because includes self
    distances, indices = nbrs.kneighbors(pts)
    return distances[:, 1:], indices[:, 1:]  # Remove distance to self


def select_labels_by_mask(labels, mask):
    """Find all labels that intersect a binary mask

    Parameters
    ----------
    labels : ndarray
        labels array
    mask : ndarray
        binary mask

    Returns
    -------
    output : ndarray
        1D array of the label values within `mask`

    """
    masked_labels = mask * labels
    return np.unique(masked_labels)[1:]


def select_pts_by_mask(pts, mask):
    """Find all points that are contained within a binary mask

    Parameters
    ----------
    pts : tuple
        tuple of index arrays
    mask : ndarray
        binary mask of the same shape relative to the `pts` tuple

    Returns
    -------
    output : tuple
        tuple of index arrays for the points in the mask

    """
    pts_idx = np.ravel_multi_index(pts, mask.shape)
    mask_idx = np.ravel_multi_index(np.where(mask), mask.shape)
    pts_in_mask_idx = np.in1d(pts_idx, mask_idx)
    return np.unravel_index(pts_in_mask_idx, mask.shape)


# Phenotyping

def signal_labeled_comprehension(image, label_image, f, dtype=float):
    """Applies `f` to `image` in each region in `label_image` except the zero label.

    Parameters
    ----------
    image : ndarray
    label_image : ndarray
    f : callable
    dtype : type, optional
        data type that f returns. Default, float

    Returns
    -------
    results : ndarray
        results of applying `f` to `image` at each region with a non-zero label
    labels : ndarray
        the non-zero labels for which the results apply

    """
    results_with_zero = labeled_comprehension(image,
                                              label_image,
                                              np.arange(label_image.max()),
                                              f,
                                              dtype,
                                              None)
    included = ~np.isnan(results_with_zero)  # boolean mask the labels in label_image
    labels_with_zero = np.where(included)[0]  # indices correspond to labels
    return results_with_zero[1:], labels_with_zero[1:]


def signal_mean_labeled(image, label_image):
    """Calculates the mean intensity in `image` at each region in `label_image`

    Parameters
    ----------
    image : ndarray
    label_image : ndarray

    Returns
    -------
    results : ndarray
    labels : ndarray

    """
    return signal_labeled_comprehension(image, label_image, np.mean)


def signal_stdev_labeled(image, label_image):
    """Calculates the stdev of intensity in `image` at each region in `label_image`

    Parameters
    ----------
    image
    label_image

    Returns
    -------
    results : ndarray
    labels : ndarray

    """
    return signal_labeled_comprehension(image, label_image, np.std)


def signal_percentile_labeled(image, label_image, q):
    """Calculates the `q` percentile of intensity in `image` at each region in `label_image`

    Parameters
    ----------
    image : ndarray
    label_image : ndarray
    q : float
        percentile to calculate between 0 and 100

    Returns
    -------
    results : ndarray
    labels : ndarray

    """
    f = partial(np.percentile, q=q)
    return signal_labeled_comprehension(image, label_image, f)


"""
1) Nuclei segmentation

load the nucleus channel
convert to Zarr or SharedMemory
segment the nuclei into labeled regions

2) Create the cell database

calculate nuclei centroids, volumes, etc
create a dataframe indexed by nucleus label with centroids, volumes, etc

3) Calculate signal statistics from other channels

For each cell type marker:
    load the image
    calculate the signal statistics within each nucleus
    add the signal statistics to the dataframe

4) Calculate the intersection and distance to user-defined objects

for each object:
    load the object
    extract a bounding box of the object
    upsample and smooth the object
    find the centroids that intersect the object
    add an indicator for contained-in-object to the dataframe
    extract the object boundary points
    calculate the distance of each nucleus to the nearest point on the object boundary
    add the distance-to-object to the dataframe
 
"""

