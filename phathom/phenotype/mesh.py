from skimage import measure
import numpy as np
np.random.seed(123)
try:
    from MulticoreTSNE import MulticoreTSNE as TSNE
except:
    from sklearn.manifold import TSNE
from tqdm import tqdm
from phathom.preprocess.filtering import gaussian_blur
try:
    from mayavi import mlab
except:
    mlab = None
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster.bicluster import SpectralCoclustering
import multiprocessing
from scipy.spatial import distance
from sklearn import preprocessing
import pandas as pd


def smooth_segmentation(seg, sigma=1, scale_factor=10):
    binary = (seg > 0)
    smooth = scale_factor * gaussian_blur(binary, sigma)
    return smooth.astype(np.float32)


def marching_cubes(seg, level, spacing, step_size):
    return measure.marching_cubes_lewiner(seg, level=level, spacing=spacing, step_size=step_size, allow_degenerate=False)


def plot_mesh(verts, faces, color=(1, 0, 0), figure=None):
    if figure is not None:
        mlab.figure(figure)
    return mlab.triangular_mesh([vert[0] for vert in verts],
                                [vert[1] for vert in verts],
                                [vert[2] for vert in verts],
                                faces,
                                color=color)


def randomly_sample(n, *items, return_idx=False):
    idx = np.arange(len(items[0]))
    np.random.shuffle(idx)
    idx = idx[:n]
    if return_idx:
        return tuple(item[idx] for item in items), idx
    else:
        return tuple(item[idx] for item in items)


def voxels_to_micron(data_voxels, voxel_size):
    return data_voxels * np.asarray(voxel_size)


def make_bins(start, stop, bins):
    bin_edges = np.linspace(start, stop, bins + 1)
    bin_width = bin_edges[1] - bin_edges[0]
    return bin_edges, bin_width


def cross_products(vectors, ref=np.array([1, 0, 0])):
    return np.cross(vectors, ref)


def dot_products(vectors, ref=np.array([1, 0, 0])):
    return np.dot(vectors, ref)


centers_um_global = None
sox2_labels_global = None
tbr1_labels_global = None


def compute_profile(vert, vi, ci, length, bins, radius):
    global centers_um_global
    global sox2_labels_global
    global tbr1_labels_global

    pts = centers_um_global
    sox2_labels = sox2_labels_global
    tbr1_labels = tbr1_labels_global

    # Translate points to origin
    pts_translated = pts - vert

    # Rotate points to align the normal with the z-axis
    v_cross = np.array([[0, -vi[2], vi[1]],
                        [vi[2], 0, -vi[0]],
                        [-vi[1], vi[0], 0]])
    rotation_matrix = np.eye(3) + v_cross + np.matmul(v_cross, v_cross) / (1 + ci)
    pts_translated_rotated = rotation_matrix.dot(pts_translated.T).T

    # Bin count the cells
    bin_edges, bin_height = make_bins(0, length, bins)
    sox2_count = np.zeros(bins, np.int)
    tbr1_count = np.zeros(bins, np.int)
    negative_count = np.zeros(bins, np.int)

    for j, bin_start in enumerate(bin_edges[:-1]):
        bin_stop = bin_start + bin_height
        x, y, z = pts_translated_rotated[:, 2], pts_translated_rotated[:, 1], pts_translated_rotated[:, 0]

        idx = np.where(np.logical_and(x ** 2 + y ** 2 <= radius ** 2, np.logical_and(z >= bin_start, z <= bin_stop)))[0]

        sox2_lbls = sox2_labels[idx]
        tbr1_lbls = tbr1_labels[idx]
        negative_lbls = np.where(np.logical_and(sox2_lbls == 0, tbr1_lbls == 0))[0]

        sox2_count[j] = sox2_lbls.sum()
        tbr1_count[j] = tbr1_lbls.sum()
        negative_count[j] = len(negative_lbls)

    return sox2_count, tbr1_count, negative_count


def _compute_profile(inputs):
    return compute_profile(*inputs)


def compute_profiles(verts, normals, length, bins, radius, centers_um, sox2_labels, tbr1_labels):
    global centers_um_global
    global sox2_labels_global
    global tbr1_labels_global

    centers_um_global = centers_um
    sox2_labels_global = sox2_labels
    tbr1_labels_global = tbr1_labels

    v = cross_products(normals)
    c = dot_products(normals)

    # Get cell density profiles for each cell-type
    args_list = []
    for i, (vi, ci, vert) in tqdm(enumerate(zip(v, c, verts)), total=len(normals)):
        args_list.append((vert, vi, ci, length, bins, radius))

    with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
        results = list(tqdm(pool.imap(_compute_profile, args_list), total=len(args_list)))
    return np.asarray(results)


def counts_to_features(counts):
    features = counts.reshape((len(counts), -1))  # Flattened profiles
    return preprocessing.scale(features)  # Normalize each feature (cell bin) to unit mean, zero variance


def hierarchical_clustering(features, n_clusters, linkage):
    labels = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage).fit_predict(features)
    return labels


def euclidean_distance_matrix(counts, nb_vectors):
    dist = np.zeros((nb_vectors, nb_vectors, counts.shape[1]))
    for c in range(counts.shape[1]):
        counts_channel = counts[:, c]
        Y = distance.squareform(distance.pdist(counts_channel, metric='correlation'))
        Y[np.isnan(Y)] = 1
        dist[..., c] = Y
    D = dist.mean(axis=-1)


def spectral_coclustering(n_clusters, D):
    scc = SpectralCoclustering(n_clusters=n_clusters).fit(D)
    # fit_data = D[np.argsort(scc.row_labels_)]
    # fit_data = fit_data[:, np.argsort(scc.column_labels_)]
    #
    # plt.matshow(D)
    # plt.matshow(fit_data)
    labels = scc.row_labels_
    return labels


def labels_to_cluster_idx(labels):
    n_clusters = len(np.unique(labels))
    return [np.where(labels == i)[0] for i in range(n_clusters)]


def separate_by_labels(labels, *items):
    cluster_idx = labels_to_cluster_idx(labels)
    return cluster_idx, tuple([item[idx] for idx in cluster_idx] for item in items)


def colormap_to_colors(n, name='Set2'):
    cmap = cm.get_cmap(name)
    colors = [tuple(list(cmap(i))[:3]) for i in range(n)]
    return colors


def plot_normals(cluster_verts, cluster_normals, colors, opacity):
    for i, (v, n, color) in enumerate(zip(cluster_verts, cluster_normals, colors)):
        mlab.quiver3d(v[:, 0], v[:, 1], v[:, 2], n[:, 0], n[:, 1], n[:, 2],
                      color=color, opacity=opacity)


def plot_nuclei(centers_um, nb_nuclei, sox2_labels, tbr1_labels, scale_factor=1, figure=None):
    if figure is not None:
        mlab.figure(figure)
    centers_sample, sox2_labels_sample, tbr1_labels_sample = randomly_sample(nb_nuclei,
                                                                             centers_um,
                                                                             sox2_labels,
                                                                             tbr1_labels)

    negative_idx = np.where(np.logical_and(sox2_labels_sample == 0, tbr1_labels_sample == 0))[0]
    sox2_idx = np.where(np.logical_and(sox2_labels_sample > 0, tbr1_labels_sample == 0))[0]
    tbr1_idx = np.where(np.logical_and(sox2_labels_sample == 0, tbr1_labels_sample > 0))[0]

    negative = centers_sample[negative_idx]
    sox2 = centers_sample[sox2_idx]
    tbr1 = centers_sample[tbr1_idx]

    # Plot nuclei
    mlab.points3d(negative[:, 0], negative[:, 1], negative[:, 2], scale_factor=scale_factor, color=(0, 0, 1))
    mlab.points3d(sox2[:, 0], sox2[:, 1], sox2[:, 2], scale_factor=scale_factor, color=(1, 0, 0))
    mlab.points3d(tbr1[:, 0], tbr1[:, 1], tbr1[:, 2], scale_factor=scale_factor, color=(0, 1, 0))


def show3d(stop=False):
    mlab.show(stop=stop)


def plot_clustermap(features, method):
    g = sns.clustermap(features, col_cluster=False, method=method)
    plt.show()
    return g


def plot_tsne(features, labels, colors):
    embedding = TSNE(n_jobs=-1).fit_transform(features)
    cluster_idx, (cluster_tsne,) = separate_by_labels(labels, embedding)
    for i, (tsne, color) in enumerate(zip(cluster_tsne, colors)):
        plt.plot(tsne[:, 0], tsne[:, 1], 'o', c=color, label=f"Cluster {i}")
    plt.legend()
    sns.despine()
    plt.show()


def cluster_sizes(labels):
    cluster_idx = labels_to_cluster_idx(labels)
    return [len(idx) for idx in cluster_idx]


def plot_cluster_profiles(counts, labels):
    n = counts.shape[-1]
    x0 = np.arange(n)
    x = []
    sox2 = []
    tbr1 = []
    dn = []
    lbls = []
    for row, lbl in zip(counts, labels):
        x += list(x0)
        sox2 += list(row[0])
        tbr1 += list(row[1])
        dn += list(row[2])
        lbls += n * [lbl]

    x = 3 * x
    lbls = 3 * lbls
    y = sox2 + tbr1 + dn
    cell_type = len(sox2) * ['sox2'] + len(tbr1) * ['tbr1'] + len(dn) * ['dn']

    df = pd.DataFrame({'x': x, 'y': y, 'cell_type': cell_type, 'labels': lbls})

    sns.relplot(x='x', y='y', hue='cell_type', col='labels',
                kind='line', palette=['r', 'g', 'b'], data=df)
    plt.show()
