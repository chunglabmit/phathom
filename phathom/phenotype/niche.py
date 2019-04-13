import numpy as np
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
from scipy.spatial.distance import pdist
import multiprocessing
from phathom.phenotype.mesh import separate_by_labels, make_bins


def fit_neighbors(pts):
    return NearestNeighbors(algorithm='kd_tree', n_jobs=-1).fit(pts)


def query_kneighbors(nbrs, pts, k):
    distances, indices = nbrs.kneighbors(pts, k)
    # indices is len(pts) by k, in order of decreasing distance
    return distances, indices


def query_radius(nbrs, pts, radius):
    return nbrs.radius_neighbors(pts, radius)


def neighborhood_counts(indices, labels=None):
    counts = np.zeros(len(indices), np.int)
    for n, idx in enumerate(indices):
        if labels is not None:
            neighborhood_labels = labels[idx]
            idx = idx[np.where(neighborhood_labels > 0)[0]]
        counts[n] = len(idx)
    return counts


pts_global = None
labels_global = None


def calculate_directionality(pts_batch, idx_batch):
    global pts_global
    global labels_global

    vectors_batch = np.zeros((len(pts_batch), pts_batch[0].shape[-1]))

    for i, (pt, idx) in enumerate(zip(pts_batch, idx_batch)):

        # Filter neighborhood by labels
        if labels_global is not None:
            neighborhood_labels = labels_global[idx]
            idx = idx[np.where(neighborhood_labels > 0)[0]]

        # Skip if no cells
        if len(idx) == 0:
            vectors_batch[i] = np.zeros(len(pt))

        # Calculate normalized directionality
        v = pts_global[idx] - pt
        v_hat = np.nan_to_num(v / np.linalg.norm(v))
        vectors_batch[i] = v_hat.sum(axis=0)

    return vectors_batch


def _calculate_directionality(args):
    return calculate_directionality(*args)


def neighborhood_directionality(pts, indices, labels=None, n_workers=None):
    vectors = np.zeros(pts.shape)
    for n, (pt, idx) in tqdm(enumerate(zip(pts, indices)), total=len(pts)):
        # Filter neighborhood by labels
        if labels is not None:
            neighborhood_labels = labels[idx]
            idx = idx[np.where(neighborhood_labels > 0)[0]]
        # Skip if no cells
        if len(idx) == 0:
            continue
        # Calculate normalized directionality
        v = pts[idx] - pt
        v_hat = np.nan_to_num(v / np.linalg.norm(v))
        vectors[n] = v_hat.sum(axis=0)
    return vectors
    # if n_workers is None:
    #     n_workers = multiprocessing.cpu_count()
    #
    # global pts_global
    # global labels_global
    # pts_global = pts
    # labels_global = labels
    #
    # batch_size = len(pts) // n_workers
    # args_list = []
    # for i in range(batch_size):
    #     start = i * batch_size
    #     stop = min((i+1) * batch_size, len(pts))
    #     pts_batch = pts[start:stop]
    #     idx_batch = indices[start:stop]
    #     args_list.append((pts_batch, idx_batch))
    #
    # with multiprocessing.Pool(n_workers) as pool:
    #     results = list(tqdm(pool.imap(_calculate_directionality, args_list), total=len(args_list)))
    # return np.asarray(results)


def directionality_projection(*vectors):
    all_vectors = np.asarray(vectors)  # n_types x len(pts) x 3
    projections = np.zeros((all_vectors.shape[1], all_vectors.shape[0]))
    for n in range(all_vectors.shape[1]):
        x = all_vectors[:, n]  # x is n_types x 3
        y = pdist(x)  # y is condensed distance matrix (n_types,)
        projections[n] = y
    return projections


def calculate_features(pts, sox2_labels, tbr1_labels, radius):
    nbrs = fit_neighbors(pts)
    distances, indices = query_radius(nbrs, pts, radius)

    # Cell counts for each type
    sox2_counts = neighborhood_counts(indices, sox2_labels)
    tbr1_counts = neighborhood_counts(indices, tbr1_labels)
    sox2_counts = neighborhood_counts(indices, ~np.logical_or(sox2_labels, tbr1_labels))

    # Directionalities for each type


def radial_profile(pts, distances, indices, radius, bins, labels=None):
    profiles = np.zeros((pts.shape[0], bins))

    vbin_edges, vbin_width = make_bins(0, 4/3*radius**3, bins)  # Equal-volume concentric shells
    bin_edges = (3/4*vbin_edges)**(1/3)

    for n, (pt, idx, dist) in tqdm(enumerate(zip(pts, indices, distances)), total=len(pts)):

        # Filter neighborhood by labels
        if labels is not None:
            neighborhood_labels = labels[idx]
            positive_idx = np.where(neighborhood_labels > 0)[0]
            idx = idx[positive_idx]
            dist = dist[positive_idx]  # select distances of the labelled cells

        # Skip if no cells
        if len(idx) == 0:
            continue

        # Calculate cell profile
        bin_idx = np.digitize(dist, bin_edges) - 1
        profiles[n] = np.bincount(bin_idx, minlength=bins)

    return profiles
