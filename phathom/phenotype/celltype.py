import multiprocessing
from functools import partial
import warnings
import numpy as np
import tqdm
from sklearn.neighbors import NearestNeighbors, kneighbors_graph
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.interpolate import griddata
import skimage
from skimage import morphology
import scipy.ndimage as ndi
from scipy.special import expit
from skimage.filters import gaussian
from skimage.feature import peak_local_max
from phathom import utils
from phathom.preprocess import filtering
from phathom.segmentation.segmentation import (eigvals_of_weingarten,
                                               seed_probability,
                                               convex_seeds,
                                               find_centroids)

import matplotlib.colors as mcolors
import matplotlib.cm as cm
import torch
use_cuda = torch.cuda.is_available()
if use_cuda:
    from phathom.segmentation.segmentation import cpu_eigvals_of_weingarten

warnings.simplefilter(action='ignore', category=FutureWarning)


def smooth(image, sigma):
    image = skimage.img_as_float32(image)
    g = gaussian(image, sigma=sigma, preserve_range=True)
    return g


def calculate_eigvals(g):
    eigvals = cpu_eigvals_of_weingarten(g)
    return eigvals


def negative_curvature_product(image, sigma):
    g = smooth(image, sigma)
    eigvals = calculate_eigvals(g)
    return convex_seeds(eigvals)


def sigmoid(x):
    return expit(x)


def curvature_probability(eigvals, steepness, offset):
    """Calculate the interest point probability based on 3D curvature eigenvalues

    Parameters
    ----------
    eigvals : ndarray
        4D array of curvature eigenvalues
    steepness : float
        Slope of the logistic function. Larger gives sharper transition between nuclei and background
    offset : float
        Translation of the logistic function. Larger biases towards more negative curvatures

    Returns
    -------
    prob : ndarray
        Curvature interest point probability

    """
    p0 = sigmoid(-steepness * (eigvals[..., 0] + offset))
    p1 = sigmoid(-steepness * (eigvals[..., 1] + offset))
    p2 = sigmoid(-steepness * (eigvals[..., 2] + offset))
    return p0 * p1 * p2


def intensity_probability(image, I0=None, stdev=None):
    """Calculate the foreground probability using exponential distribution

    Parameters
    ----------
    image : ndarray
        Input image
    I0 : float, optional
        Normalization value. Default, mean of image
    stdev : float, optional
        Width of the transition to foreground. Default, stdev of normalized image

    Returns
    -------
    prob : ndarray
        Foreground probability map

    """
    if I0 is None:
        I0 = image.mean()
    normalized = image / I0
    if stdev is None:
        stdev = normalized.std()
        print(stdev)
    return 1 - np.exp(-normalized ** 2 / (2 * stdev ** 2))


def nucleus_probability(image, sigma, steepness=500, offset=0.0005, I0=None, stdev=None):
    """Calculate the nucleus probability map using logistic regression over
    curvature eigenvalues

    Parameters
    ----------
    image : ndarray
        3D image volume of nuclei staining
    sigma : int or tuple
        Amount to blur the image before processing
    steepness : float
        Slope of the logistic function. Larger gives sharper transition between nuclei and background
    offset : float
        Translation of the logistic function. Larger biases towards more negative curvatures

    Returns
    -------
    prob : ndarray
        Nuclei probability map

    """
    g = smooth(image, sigma)
    eigvals = calculate_eigvals(g)
    p_curvature = curvature_probability(eigvals, steepness, offset)
    p_intensity = intensity_probability(g, I0, stdev)
    return p_curvature * p_intensity


def nuclei_centers_ncp(image, sigma, **plm_kwargs):
    ncp = negative_curvature_product(image, sigma)
    return peak_local_max(-ncp, **plm_kwargs)


def nuclei_centers_probability(prob, threshold, h):
    prob = filtering.remove_background(prob, threshold)
    hmax = morphology.reconstruction(prob - h, prob, 'dilation')
    extmax = prob - hmax
    seeds = (extmax > 0)
    labels = ndi.label(seeds, morphology.cube(width=3))[0]
    return np.round(find_centroids(labels)).astype(np.int)


def nuclei_centers_probability2(prob, threshold, min_dist):
    prob = filtering.remove_background(prob, threshold)
    return peak_local_max(prob, min_distance=min_dist, threshold_abs=threshold)


def _detect_nuclei_chunk(input_tuple, overlap, sigma, min_intensity, steepness, offset, I0=None, stdev=None, prob_thresh=0.5, min_dist=1, prob_output=None):
    arr, start_coord, chunks = input_tuple

    ghosted_chunk, start_ghosted, _ = utils.extract_ghosted_chunk(arr, start_coord, chunks, overlap)

    if ghosted_chunk.max() < min_intensity:
        return None

    prob = nucleus_probability(ghosted_chunk, sigma, steepness, offset, I0, stdev)

    if prob_output is not None:
        start_local = start_coord - start_ghosted
        stop_local = np.minimum(start_local + np.asarray(chunks),
                                np.asarray(ghosted_chunk.shape))
        prob_valid = utils.extract_box(prob, start_local, stop_local)
        stop_coord = start_coord + np.asarray(prob_valid.shape)
        utils.insert_box(prob_output, start_coord, stop_coord, prob_valid)

    centers_local = nuclei_centers_probability2(prob, prob_thresh, min_dist)

    if centers_local.size == 0:
        return None

    # Filter out any centers detected in ghosted area
    centers_interior = utils.filter_ghosted_points(start_ghosted, start_coord, centers_local, chunks, overlap)

    # change to global coordinates
    centers = centers_interior + start_ghosted
    return centers


def detect_nuclei_parallel(z_arr, sigma, min_intensity, steepness, offset, I0, stdev, prob_thresh, min_dist, chunks, overlap, nb_workers=None, prob_output=None):
    f = partial(_detect_nuclei_chunk,
                overlap=overlap,
                sigma=sigma,
                min_intensity=min_intensity,
                steepness=steepness,
                offset=offset,
                I0=I0,
                stdev=stdev,
                prob_thresh=prob_thresh,
                min_dist=min_dist,
                prob_output=prob_output)
    results = utils.pmap_chunks(f, z_arr, chunks, nb_workers, use_imap=True)
    return results


def sample_intensity_cube(center, image, radius):
    start = [max(0, int(c - radius)) for c in center]
    stop = [min(int(c + radius + 1), d - 1) for c, d in zip(center, image.shape)]
    bbox = utils.extract_box(image, start, stop)
    return bbox.flatten()


def nuclei_centered_intensities(image, centers, radius, mode='cube', nb_workers=None):
    if nb_workers is None:
        nb_workers = multiprocessing.cpu_count()

    if mode == 'cube':
        f = partial(sample_intensity_cube, image=image, radius=radius)
    else:
        raise ValueError('Only cube sampling is currently supported')

    with multiprocessing.Pool(nb_workers) as pool:
        intensities = list(tqdm.tqdm(pool.imap(f, centers), total=centers.shape[0]))

    return intensities


def calculate_mfi(input):
    """Calculate the Mean Fluorescence Intensity for input list of nucleus-centered samples

    Parameters
    ----------
    input : list
        List of ndarrays containing image intensities near nuclei

    Returns
    -------
    output : ndarray
        1D array of MFIs for each nucleus
    """
    return np.asarray([x.mean() for x in input])


def calculate_stdev(input):
    """Calculate the standard deviation for input list of nucleus-centered samples

    Parameters
    ----------
    input : list
        List of ndarrays containing image intensities near nuclei

    Returns
    -------
    output : ndarray
        1D array of standard deviations for each nucleus
    """
    return np.asarray([x.std() for x in input])


def threshold_mfi(mfi, threshold):
    positive_idx = np.where(mfi > threshold)[0]
    labels = np.zeros(mfi.shape, dtype=np.int)
    labels[positive_idx] = 1
    return labels


def query_neighbors(pts, n_neighbors, query_pts=None):
    if query_pts is None:
        query_pts = pts
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='kd_tree', n_jobs=-1).fit(pts)
    distances, indices = nbrs.kneighbors(query_pts)
    return distances, indices


def query_radius(pts, radius, query_pts=None):
    if query_pts is None:
        query_pts = pts
    nbrs = NearestNeighbors(radius=radius, algorithm='kd_tree', n_jobs=-1).fit(pts)
    distances, indices = nbrs.radius_neighbors(query_pts)
    return distances, indices


def local_densities(distances, indices, sox2_labels, tbr1_labels, radius=None):

    features = []

    for dist, idx in zip(distances, indices):

        sox2_flags = sox2_labels[idx]
        tbr1_flags = tbr1_labels[idx]

        nb_cells = len(idx)

        if nb_cells == 0:
            cell_density = 0
            sox2_density = 0
            tbr1_density = 0
        else:
            nb_sox2 = sox2_flags.sum()
            nb_tbr1 = tbr1_flags.sum()
            if radius is None:
                radius = dist.mean()
            cell_density = nb_cells / (4 / 3 * np.pi * radius ** 3)
            sox2_density = nb_sox2 / (4 / 3 * np.pi * radius ** 3)
            tbr1_density = nb_tbr1 / (4 / 3 * np.pi * radius ** 3)

        # if np.any(sox2_flags == 1):
        #     sox2_distances = dist[sox2_flags]
        #     sox2_radius = sox2_distances.max()
        #     sox2_density = nb_sox2 / (4 / 3 * np.pi * radius ** 3)
        # else:
        #     sox2_density = 0
        #
        # if np.any(tbr1_flags == 1):
        #     if len(tbr1_flags) > 1:
        #         tbr1_distances = dist[tbr1_flags]
        #         tbr1_radius = tbr1_distances.max()
        #     tbr1_density = nb_tbr1 / (4 / 3 * np.pi * radius ** 3)
        # else:
        #     tbr1_density = 0

        f = np.array([sox2_density, tbr1_density])
        features.append(f)

    return np.asarray(features)


import matplotlib.pyplot as plt


def cluster_dendrogram(features):
    z = linkage(features, method='ward')
    plt.figure()
    dendrogram(z)
    plt.show()


def connectivity(pts, n_neighbors=2, include_self=False):
    return kneighbors_graph(pts, n_neighbors=n_neighbors, include_self=include_self)


def cluster(features, n_clusters=2, connectivity=None):
    if connectivity is None:
        region_labels = AgglomerativeClustering(n_clusters=n_clusters).fit_predict(features)
    else:
        region_labels = AgglomerativeClustering(n_clusters=n_clusters,
                                                connectivity=connectivity).fit_predict(features)
    return region_labels


def voronoi(pts):
    return Voronoi(pts)


def voronoi_plot(vor, labels=None):
    fig = voronoi_plot_2d(vor, show_points=True, show_vertices=False, s=1)
    if labels is not None:
        norm = mcolors.Normalize(vmin=0, vmax=1, clip=True)
        mapper = cm.ScalarMappable(norm=norm, cmap=cm.Blues_r)
        for r in range(len(vor.point_region)):
            region = vor.regions[vor.point_region[r]]
            if not -1 in region:
                polygon = [vor.vertices[i] for i in region]
                plt.fill(*zip(*polygon), color=mapper.to_rgba(labels[r]))
    plt.show()


def rasterize_regions(pts, labels, shape):
    grid_z, grid_y, grid_x = np.mgrid[0:shape[0], 0:shape[1], 0:shape[2]]
    return griddata(pts, labels, (grid_z, grid_y, grid_x), method='nearest').astype(np.uint8)
