import numpy as np
from skimage import img_as_float32
from skimage.filters import gaussian, sobel
from skimage.segmentation import felzenszwalb
from skimage.future import graph
from skimage.morphology import binary_dilation, reconstruction, cube
from skimage import measure

from scipy.ndimage import labeled_comprehension, find_objects, label

from QuickshiftPP import QuickshiftPP

from phathom.segmentation.graphcuts import graph_cuts
from phathom.segmentation.segmentation import find_volumes
from phathom.db.cell_table import filter_label_image



def smooth(image, sigma):
    return gaussian(image, sigma, preserve_range=True)


def pixel_coordinates(shape, voxel_dim=None):
    size = np.asarray(shape).prod()
    grid = np.indices(shape)
    coords = np.reshape(grid, (grid.shape[0], size)).T
    return coords


def build_features(image, gamma=1.0, voxel_dim=None):
    coords = pixel_coordinates(image.shape, voxel_dim)
    X = np.zeros((image.size, image.ndim + 1), np.float32)
    X[:, :-1] = coords
    X[:, -1] = gamma * image.ravel()
    return X


def quickshift(X, k, beta):
    model = QuickshiftPP(k=k, beta=beta)
    model.fit(X)
    return model


def labels_to_segmentation(labels, shape):
    return labels.reshape(shape)


def quickshift_segmentation(image, k, beta, gamma=1.0, voxel_dim=None):
    X = build_features(image, gamma, voxel_dim)
    qs = quickshift(X, k, beta)
    return labels_to_segmentation(qs.memberships, image.shape)


def gc(image, back_mu, obj_mu, w_const):
    return graph_cuts(image, back_mu, obj_mu, w_const)


def merge_mean_intensity(image, labels, thresh):
    rag = graph.rag_mean_color(image, labels)
    return graph.cut_threshold(labels, rag, thresh=thresh)


def merge_sobel(image, labels, thresh):
    edges = sobel(image)
    rag = graph.rag_boundary(labels, edges)
    return graph.cut_threshold(labels, rag, thresh=thresh)


def calculate_region_mfi(image, labels):
    idx = np.arange(1, labels.max()+1)
    return labeled_comprehension(image, labels, idx, np.mean, float, -1)


def calculate_volume(labels):
    regions = measure.regionprops(labels)
    volumes = []
    for i, region in enumerate(regions):
        volumes.append(region.area)
    return np.asarray(volumes)


def calculate_relative_sox2(labels, sox2_img, selem):
    lbls = np.arange(1, labels.max()+1)
    sox2_in = np.zeros(lbls.size)
    sox2_out = np.zeros(lbls.size)
    sox2_ratio = np.zeros(lbls.size)
    for i, lbl in enumerate(lbls):
        inside = (filter_label_image(labels, lbl) > 0)
        dilated = binary_dilation(inside, selem=selem)
        outside = np.logical_and(dilated, np.logical_not(inside))
        sox2_in[i] = sox2_img[inside].mean()
        sox2_out[i] = sox2_img[outside].mean()
        sox2_ratio[i] = sox2_out[i] / (sox2_in[i] + 1e-6)
    return sox2_in, sox2_out, sox2_ratio


def remove_largest_region(labels):
    volumes = find_volumes(labels)
    keep = np.where(volumes < volumes.max())[0] + 1
    labels_filtered = filter_label_image(labels, keep)
    labels_largest = labels - labels_filtered
    mask_largest = (labels_largest > 0)
    return labels_filtered, mask_largest


def remove_small_regions(seg, min_volume):
    labels, nb_lbls = label(seg)
    volumes = find_volumes(labels)
    keep = np.where(volumes >= min_volume)[0] + 1
    return filter_label_image(labels, keep)

