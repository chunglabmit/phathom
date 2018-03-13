import numpy as np
from scipy.stats import poisson
import maxflow


def poisson_pdf(x, mu):
    return poisson.pmf(x, mu)

def information_gain(p):
    return np.exp(1-p)

def hist(data, bins, _range):
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

    back_mu = np.sum(back_levels*freq[back_idx])/np.sum(freq[back_idx])
    obj_mu = np.sum(obj_levels*freq[obj_idx])/np.sum(freq[obj_idx])
    return back_mu, obj_mu

def entropy(back_mu, obj_mu, T):
    # Use all grayscale levels in entropy calculation
    x_back = np.array(range(T))
    x_obj = np.array(range(T+1,256))

    p_back = poisson_pdf(x_back, back_mu)
    p_obj = poisson_pdf(x_obj, obj_mu)

    info_gain_back = information_gain(p_back)
    info_gain_obj = information_gain(p_obj)

    H = np.sum(p_back*info_gain_back) + np.sum(p_obj*info_gain_obj)
    return H

def max_entropy(data):
    assert data.dtype == 'uint8'

    # Get histogram for mu estimation
    hist_output = hist(data, bins=256, _range=(0, 256))
    width = hist_output[-1]
    Ts = np.array(
        range(
            np.ceil(data.min()+width).astype('int'),
            np.floor(data.max()-width).astype('int')
        ),
        dtype='uint8'
    )

    # Test all thresholds
    numT = len(Ts)
    Hs = np.zeros(numT)
    back_mus = np.zeros(numT)
    obj_mus = np.zeros(numT)
    for i, T in enumerate(Ts):
        back_mu, obj_mu = fit_poisson_mixture(hist_output, T)
        H = entropy(back_mu, obj_mu, T)
        Hs[i] = H
        back_mus[i] = back_mu
        obj_mus[i] = obj_mu

    idx_star = np.argmax(Hs)
    T_star = Ts[idx_star]
    back_mu_star = back_mus[idx_star]
    obj_mu_star = obj_mus[idx_star]

    return T_star, back_mu_star, obj_mu_star

def histogram_penalty(x, mu_star):
    px = poisson_pdf(x, mu_star)
    return -np.log(px)

def graph_cuts(data, R_back, R_obj, B=None, lambdaI=0, lambda_const=0):
    # Create a new graph
    g = maxflow.Graph[float]()

    # node_ids number each voxel by column, then row, then slice
    node_ids = g.add_grid_nodes(data.shape)

    if lambdaI > 0:
        # Add boundary edges - cost for discontinuities
        for z in range(1, data.shape[0]-1):
            print(z)
            for y in range(1, data.shape[1]-1):
                for x in range(1, data.shape[2]-1):

                    # Get valid neighbor indices
                    neighbors = [[],[],[]]
                    if z+1 < data.shape[0]:
                        neighbors[0].append(z+1)
                        neighbors[1].append(y)
                        neighbors[2].append(x)
                    if z-1 >= 0:
                        neighbors[0].append(z-1)
                        neighbors[1].append(y)
                        neighbors[2].append(x)
                    if y+1 < data.shape[1]:
                        neighbors[0].append(z)
                        neighbors[1].append(y+1)
                        neighbors[2].append(x)
                    if y-1 >= 0:
                        neighbors[0].append(z)
                        neighbors[1].append(y-1)
                        neighbors[2].append(x)
                    if x+1 < data.shape[2]:
                        neighbors[0].append(z)
                        neighbors[1].append(y)
                        neighbors[2].append(x+1)
                    if x-1 >= 0:
                        neighbors[0].append(z)
                        neighbors[1].append(y)
                        neighbors[2].append(x-1)

                    # Get node and neighbor ids
                    source_id = node_ids[z,y,x]
                    sink_ids = node_ids[neighbors]

                    # Calculate capacities from data
                    I_node = data[z,y,x] # Voxel intensities
                    I_neighbors = data[neighbors]

                    # Add n-links to the graph
                    for sink_id, I_neighbor in zip(sink_ids, I_neighbors):
                        cap = lambdaI*B[I_node, I_neighbor]
                        g.add_edge(source_id, sink_id, cap, cap)

    if lambda_const > 0:
        # Add n-links
        g.add_grid_edges(node_ids, lambda_const)

    # Add terminals - source and sink capacities are based on intensities
    cap_obj = R_obj[data]
    cap_back = R_back[data]
    g.add_grid_tedges(node_ids, cap_obj, cap_back)

    # Perform segmentation
    g.maxflow()
    sgm = g.get_grid_segments(node_ids)

    return sgm

def run_graph_cuts(data, lambdaI, lambda_const):

    # Convert to 8-bit
    data = (data-data.min())/(data.max()-data.min())*255
    data = data.astype('uint8')

    # Estimate Poisson parameters
    T, back_mu, obj_mu = max_entropy(data)

    # Calculate LUT for histogram penalties
    x = np.array(range(0, 256))
    R_back = histogram_penalty(x, back_mu)
    R_obj = histogram_penalty(x, obj_mu)

    # Calculate LUT for intesity-based boundary regularization
    delta = np.std(data)
    B = np.zeros((256,256))
    for i in range(256):
        for j in range(256):
            B[i,j] = np.exp(-(i-j)**2/(delta**2))

    # Segment with graph cuts
    seg = graph_cuts(data, R_back, R_obj, B, lambdaI=lambdaI, lambda_const=lambda_const)
    return seg