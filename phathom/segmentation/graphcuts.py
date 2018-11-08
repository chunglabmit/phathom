import numpy as np
from scipy.stats import poisson
import maxflow
import tqdm
from functools import partial
import multiprocessing
from phathom import utils
# import warnings  # Leads to import errors with skimage.filters.gaussian
# warnings.filterwarnings("error")


def poisson_pdf(x, mu):
    return poisson.pmf(x, mu)


def information_gain(p):
    return np.exp(1-p)


def get_nb_bins(dtype):
    return min(np.iinfo(dtype).max + 1, 2**12)


def hist(data, bins=None, _range=None):
    if bins is None:
        bins = get_nb_bins(data.dtype)
    if _range is None:
        _range = (0, bins)
    freq, edges = np.histogram(
        data,
        bins=bins,
        range=_range
    )
    left_edges = edges[:-1]
    width = edges[1]-edges[0]
    return left_edges, freq, width


def find_nearest_T(T_target, Ts):
    return Ts[np.argmin(np.abs(Ts-T_target))]


def fit_poisson_mixture(hist_output, T):
    # Only use bins for mu estimation
    left_edges, freq, width = hist_output

    back_idx = np.where(left_edges < T)[0]
    obj_idx = np.where(left_edges >= T)[0]

    back_levels = left_edges[back_idx]+width/2
    obj_levels = left_edges[obj_idx]+width/2

    back_total = np.sum(freq[back_idx])
    obj_total = np.sum(freq[obj_idx])

    back_mu = np.sum(back_levels*freq[back_idx])/back_total
    obj_mu = np.sum(obj_levels*freq[obj_idx])/obj_total

    return back_mu, obj_mu


def entropy(back_mu, obj_mu, T, nb_levels=256):
    # Use all grayscale levels in entropy calculation
    x_back = np.arange(T)
    x_obj = np.arange(T+1, nb_levels)

    p_back = poisson_pdf(x_back, back_mu)
    p_obj = poisson_pdf(x_obj, obj_mu)

    info_gain_back = information_gain(p_back)
    info_gain_obj = information_gain(p_obj)

    H = np.sum(p_back*info_gain_back) + np.sum(p_obj*info_gain_obj)
    return H


def max_entropy(hist_output, dtype):
    if not np.issubdtype(dtype, np.unsignedinteger):
        raise ValueError('dtype is not unsigned integer type')

    _, freq, width = hist_output
    nonzeros_levels = np.where(freq > 0)[0]

    Ts = np.arange(np.ceil(nonzeros_levels.min()+width).astype('int'),
                   np.floor(nonzeros_levels.max()-width).astype('int'))

    # Test all thresholds
    numT = len(Ts)
    Hs = np.zeros(numT)
    back_mus = np.zeros(numT)
    obj_mus = np.zeros(numT)
    for i, T in enumerate(Ts):
        back_mu, obj_mu = fit_poisson_mixture(hist_output, T)
        Hs[i] = entropy(back_mu, obj_mu, T)
        back_mus[i] = back_mu
        obj_mus[i] = obj_mu

    idx_star = np.argmax(Hs)
    T_star = Ts[idx_star]
    back_mu_star = back_mus[idx_star]
    obj_mu_star = obj_mus[idx_star]

    if dtype == 'uint16':
        T_star = T_star / 255 * 4095
        back_mu_star = back_mu_star / 255 * 4095
        obj_mu_star = obj_mu_star / 255 * 4095

    return T_star, back_mu_star, obj_mu_star


def histogram_penalty(x, mu_star):
    px = poisson_pdf(x, mu_star)
    # return -np.log(np.clip(px, 1e-20, None))  # This is impacting the seg result by capping penalties
    try:
        penalty = -np.log(px)
    except RuntimeWarning:
        penalty = np.empty(px.shape)
        bad_idx = np.where(px == 0)
        good_idx = np.where(px > 0)
        penalty[good_idx] = -np.log(px[good_idx])
        penalty[bad_idx] = np.inf
    return penalty


def gradient_penalty_lut(data):
    delta = np.std(data)
    bins = get_nb_bins(data.dtype)
    B = np.zeros((bins, bins))
    for i in range(bins):
        for j in range(bins):
            B[i, j] = np.exp(-(i - j) ** 2 / (delta ** 2))
    return B


def histogram_penalty_lut(dtype, mu):
    bins = get_nb_bins(dtype)
    x = np.arange(bins)
    return histogram_penalty(x, mu)


def create_new_graph(shape, use_ints=False):
    if use_ints:
        g = maxflow.Graph[float]()
    else:
        g = maxflow.Graph[int]()
    node_ids = g.add_grid_nodes(shape)
    return g, node_ids


def add_constant_edges(g, node_ids, weight):
    g.add_grid_edges(node_ids, weight)
    return g


def add_terminal_edges(g, node_ids, obj_weights, back_weights):
    g.add_grid_tedges(node_ids, obj_weights, back_weights)
    return g


def add_gradient_edges(g, node_ids, data, B, w0):
    for z in tqdm.tqdm(range(1, data.shape[0]-1)):
        for y in range(1, data.shape[1]-1):
            for x in range(1, data.shape[2]-1):
                # Get valid neighbor indices
                neighbors = [[], [], []]
                if z + 1 < data.shape[0]:
                    neighbors[0].append(z + 1)
                    neighbors[1].append(y)
                    neighbors[2].append(x)
                if z - 1 >= 0:
                    neighbors[0].append(z - 1)
                    neighbors[1].append(y)
                    neighbors[2].append(x)
                if y + 1 < data.shape[1]:
                    neighbors[0].append(z)
                    neighbors[1].append(y + 1)
                    neighbors[2].append(x)
                if y - 1 >= 0:
                    neighbors[0].append(z)
                    neighbors[1].append(y - 1)
                    neighbors[2].append(x)
                if x + 1 < data.shape[2]:
                    neighbors[0].append(z)
                    neighbors[1].append(y)
                    neighbors[2].append(x + 1)
                if x - 1 >= 0:
                    neighbors[0].append(z)
                    neighbors[1].append(y)
                    neighbors[2].append(x - 1)

                # Get node and neighbor ids
                source_id = node_ids[z, y, x]
                sink_ids = node_ids[neighbors]

                # Calculate capacities from data
                I_node = data[z, y, x]  # Voxel intensities
                I_neighbors = data[neighbors]

                # Add n-links to the graph
                for sink_id, I_neighbor in zip(sink_ids, I_neighbors):
                    cap = w0 * B[I_node, I_neighbor]
                    g.add_edge(source_id, sink_id, cap, cap)
    return g


def graph_cuts(data, back_mu, obj_mu, w_const=0, w_grad=0):
    """Segments a 3D image using graph cuts

    Parameters
    ----------
    data : ndarray
        3D array of integers to segment
    back_mu : float
        mean level of the background
    obj_mu : float
        mean level of the foreground / object
    w_const : float, optional
        weight to penalize between all adjacent nodes. Default, 0
    w_grad : float, optional
        weight to penalize between regions with weak edges. Default, 0

    Returns
    -------
    sgm : ndarray
        3D array of the segmentation

    """
    # node_ids number each voxel by column, then row, then slice
    g, node_ids = create_new_graph(data.shape, use_ints=True)

    if w_grad > 0:
        B = gradient_penalty_lut(data)
        g = add_gradient_edges(g, node_ids, data, B, w_grad)

    if w_const > 0:
        g = add_constant_edges(g, node_ids, w_const)

    # Calculate LUT for histogram penalties
    R_back = histogram_penalty_lut(data.dtype, back_mu)
    R_obj = histogram_penalty_lut(data.dtype, obj_mu)

    # Add terminal edges with histogram penalties
    obj_weights = R_obj[data]
    back_weights = R_back[data]

    g = add_terminal_edges(g, node_ids, obj_weights, back_weights)

    # Perform segmentation
    g.maxflow()
    sgm = g.get_grid_segments(node_ids)

    return sgm.astype(np.uint8)


def _chunk_graph_cuts(input_tuple, out_arr, overlap, back_mu, obj_mu, w_const, w_grad):
    arr, start_coord, chunks = input_tuple
    end_coord = np.minimum(arr.shape, start_coord + np.asarray(chunks))

    # extract ghosted chunk of data
    start_overlap = np.maximum(np.zeros(arr.ndim, 'int'),
                               np.array([s - overlap for s in start_coord]))
    stop_overlap = np.minimum(arr.shape, np.array([e + overlap for e in end_coord]))
    data_overlap = utils.extract_box(arr, start_overlap, stop_overlap)

    # segment the chunk
    sgm_overlap = graph_cuts(data_overlap, back_mu, obj_mu, w_const, w_grad)
    start_local = start_coord - start_overlap
    end_local = end_coord - start_overlap
    sgm = utils.extract_box(sgm_overlap, start_local, end_local)

    # write the result
    stop_out = np.minimum(out_arr.shape, end_coord)
    utils.insert_box(out_arr, start_coord, stop_out, sgm)


def parallel_graph_cuts(arr, out_arr, overlap, chunks, nb_workers=None, **kwargs):
    """Perform graph cuts segmentation in parallel with overlapping chunks

    Parameters
    ----------
    arr : array-like or SharedMemory
        reference to Zarr or SharedMemory array to segment
    out_arr : array-like or SharedMemory
        reference to Zarr or SharedMemory array to write segmentation result to
    overlap : int
        number of pixels to overlap adjacent chunks
    chunks : tuple
        shape of each chunk to be segmented (not including the overlap)
    nb_workers : int, optional
        number of workers to use. Default, cpu_count
    kwargs : dict
        dictionary of arguments to pass to graph_cuts

    """
    back_mu = kwargs.pop('back_mu')
    obj_mu = kwargs.pop('obj_mu')
    w_const = kwargs.pop('w_const', 0)
    w_grad = kwargs.pop('w_grad', 0)

    f = partial(_chunk_graph_cuts,
                out_arr=out_arr,
                overlap=overlap,
                back_mu=back_mu,
                obj_mu=obj_mu,
                w_const=w_const,
                w_grad=w_grad)

    utils.pmap_chunks(f, arr, chunks, nb_workers, use_imap=True)
