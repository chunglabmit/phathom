from phathom.segmentation import partition
from phathom.segmentation import graphcuts
from phathom import utils
from phathom.io import conversion as imageio
import argparse
import numpy as np
import scipy.ndimage as ndi
import skimage
from skimage import segmentation, feature, morphology, filters, exposure, transform, measure
from skimage.filters import gaussian
import multiprocessing
from functools import partial
import os.path
import shutil
from warnings import warn


# Set consistent numpy random state
np.random.seed(123)

# # Initialize the output list
# outputs = []
#
#
# def add_output(name, data):
#     outputs.append({'name': name, 'data': data})
# def write_output():
#     if outputs:
#         for d in outputs:
#             name = d['name']
#             print('saving {}'.format(name))
#             imageio.imsave('../data/{}.tif'.format(d['name']), d['data'].astype('float32'))


def gradient(data):
    fz, fy, fx = np.gradient(data, edge_order=2)
    grad = np.zeros((*data.shape, 3))
    grad[..., 0] = fz
    grad[..., 1] = fy
    grad[..., 2] = fx
    return grad


def hessian_2d(data, sigma):
    l1 = np.zeros(data.shape)
    for i in range(data.shape[0]):
        img = data[i]
        Hyy, Hyx, Hxx = skimage.feature.hessian_matrix(img, sigma=sigma, mode='reflect')
        l1[i], l2 = skimage.feature.hessian_matrix_eigvals(Hyy, Hyx, Hxx)


def hessian(data):
    grad = np.gradient(data)
    n_dims = len(grad)
    H = np.zeros((*data.shape, n_dims, n_dims))
    for i, first_deriv in enumerate(grad):
        for j in range(i, n_dims):
            second_deriv = np.gradient(first_deriv, axis=j)
            H[..., i, j] = second_deriv
            if i != j:
                H[..., j, i] = second_deriv
    return H


def structure_tensor(grad):
    S = np.zeros((*grad.shape, 3))
    fz = grad[..., 0]
    fy = grad[..., 1]
    fx = grad[..., 2]
    S[..., 0, 0] = fz**2
    S[..., 1, 1] = fy**2
    S[..., 2, 2] = fx**2
    S[..., 0, 1] = fz*fy
    S[..., 0, 2] = fz*fx
    S[..., 1, 2] = fy*fx
    S[..., 1, 0] = fz*fy
    S[..., 2, 0] = fz*fx
    S[..., 2, 1] = fy*fx
    return S


def weingarten(g):
    grad = gradient(g)
    l = np.sqrt(1+np.linalg.norm(grad, axis=-1))
    H = hessian(g)
    S = structure_tensor(grad)
    L = np.zeros(S.shape)
    for i in range(3):
        for j in range(3):
            L[..., i, j] = l
    eye = np.zeros(S.shape)
    eye[..., 0, 0] = 1
    eye[..., 1, 1] = 1
    eye[..., 2, 2] = 1
    B = S+eye
    A = H*np.linalg.inv(B)/L
    return A


def eigvalsh(A):
    return np.linalg.eigvalsh(A)


def eigvals_of_weingarten(g):
    return eigvalsh(weingarten(g))


def convex_seeds(eigvals):
    eig_neg = np.clip(eigvals, None, 0)
    det_neg = eig_neg.prod(axis=-1)
    return det_neg


def positive_curvatures(eigvals):
    eig_pos = np.clip(eigvals, 0, None)
    pos_curv = np.linalg.norm(eig_pos, axis=-1)
    return pos_curv


def seed_probability(eigen_seeds):
    return 1-np.exp(-eigen_seeds**2/(2*eigen_seeds.std()**2))


def regionprops(intensity_img, labels_img):
    # Get region statistics
    # print('Getting region statistics')
    verbose = False
    nb_regions = labels_img.max()
    properties = measure.regionprops(labels_img, intensity_image=intensity_img)
    centroids = np.zeros((nb_regions, 3))
    for i, region in enumerate(properties):
        # Shape based
        zc, yc, xc = region.centroid
        nb_vox = region.area
        # volume = nb_vox*voxel_volume
        eq_diam = region.equivalent_diameter
        princple_interia = region.inertia_tensor_eigvals
        inertia_ratio = princple_interia[0]/max(1e-6, princple_interia[1])
        # Intensity-based
        max_intensity = region.max_intensity
        mean_intensity = region.mean_intensity
        min_intensity = region.min_intensity
        # Tabulate data
        centroids[i] = region.centroid
    return None


def find_volumes(l):
    z, y, x = np.where(l > 0)
    a = np.bincount(l[z, y, x])
    return a[1:]


def find_centroids(l):
    z, y, x = np.where(l > 0)
    a = np.bincount(l[z, y, x])
    a[0] = 1
    xx = np.bincount(l[z, y, x], x)
    yy = np.bincount(l[z, y, x], y)
    zz = np.bincount(l[z, y, x], z)
    return np.column_stack(
        (zz.astype(float) / a, yy.astype(float) / a, xx.astype(float) / a))[1:]


def _calc_chunk_histogram(arr, start_coord, chunks):
    stop_coord = np.minimum(arr.shape, start_coord + np.asarray(chunks))
    data = utils.extract_box(arr, start_coord, stop_coord)
    if data.dtype != 'uint8':
        data = data.astype('float32')
        data /= 4095
        data *= 255
        data = data.astype('uint8')
    _, freq, _ = graphcuts.hist(data, bins=256)
    return freq


def calculate_histogram(image, chunks, nb_workers):
    freqs = utils.pmap_chunks(_calc_chunk_histogram, image, chunks, nb_workers)
    freq = np.zeros_like(freqs[0])
    for f in freqs:
        freq += f
    # freq[-1] = 0  # Override the number of saturated pixels
    if image.dtype != 'uint8':
        left_edges, _, width = graphcuts.hist(np.zeros(1, dtype=np.uint8))
    else:
        left_edges, _, width = graphcuts.hist(np.zeros(1, dtype=image.dtype))
    return left_edges, freq, width


def otsu_threshold(freq):
    total = np.sum(freq)
    sumB = 0
    wB = 0
    maximum = 0.0
    sum1 = np.sum(np.arange(len(freq)) * freq)
    for i, f in enumerate(freq):
        wB = wB + f
        wF = total - wB
        if wB == 0 or wF == 0:
            continue
        sumB = sumB + (i-1)*f
        mF = (sum1 - sumB) / wF
        between = wB * wF * ((sumB / wB) - mF) * ((sumB / wB) - mF)
        if between >= maximum:
            level = i
            maximum = between
    return level


def mean_threshold(freq):
    return np.sum(np.arange(len(freq)) * freq) / np.sum(freq)


def calculate_seeds(eigvals, mask, h):
    eigen_seeds = convex_seeds(eigvals)
    seed_prob = seed_probability(eigen_seeds) * mask  # filters out background seeds
    hmax = morphology.reconstruction(seed_prob - h, seed_prob, 'dilation')
    extmax = seed_prob - hmax
    seeds = (extmax > 0) * mask  # avoid any seeds in the background
    return seeds


def segment_nuclei(image, mask, sigma, h):
    """Segment nuclei using curvature-based watershed

    Parameters
    ----------
    image : ndarray
        nuclei image to segment
    mask : ndarray
        foreground mask
    sigma : float or tuple
        amount to smooth the image
    h : float
        non-maximum suppression amount

    Returns
    -------
    binary_seg : ndarray
        bianry segmentation of nuclei with watershed lines
    nb_nuclei : int
        number of nuclei detected in `image`

    """
    image = skimage.img_as_float32(image)
    g = gaussian(image, sigma=sigma)
    eigvals = eigvals_of_weingarten(g)
    pos_curv = positive_curvatures(eigvals)
    seeds = calculate_seeds(eigvals, mask, h)
    markers = ndi.label(seeds, morphology.cube(width=3))[0]
    nb_nuclei = int(markers.max())
    labels = morphology.watershed(pos_curv, markers, mask=mask, watershed_line=True)
    binary_seg = labels > 0
    return binary_seg, nb_nuclei


def _segment_chunk(in_arr, start_coord, chunks, out_arr, overlap, back_mu, obj_mu,
                        w_const, w_grad, sigma, h):
    # extract ghosted chunk of data
    end_coord = np.minimum(in_arr.shape, start_coord + np.asarray(chunks))
    start_overlap = np.maximum(np.zeros(in_arr.ndim, 'int'),
                               np.array([s - overlap for s in start_coord]))
    stop_overlap = np.minimum(in_arr.shape, np.array([e + overlap for e in end_coord]))
    data_overlap = utils.extract_box(in_arr, start_overlap, stop_overlap)

    # segment the chunk
    mask_overlap = graphcuts.graph_cuts(data_overlap, back_mu, obj_mu, w_const, w_grad)
    binary_seg_overlap, nb_nuclei = segment_nuclei(data_overlap, mask_overlap, sigma, h)
    binary_seg_overlap_eroded = ndi.binary_erosion(binary_seg_overlap)

    # write the segmentation result
    start_local = start_coord - start_overlap
    end_local = end_coord - start_overlap
    binary_seg = utils.extract_box(binary_seg_overlap_eroded, start_local, end_local)
    stop_out = np.minimum(out_arr.shape, end_coord)
    utils.insert_box(out_arr, start_coord, stop_out, binary_seg)
    return nb_nuclei


def parallel_segment_nuclei(in_arr, out_arr, w_const, w_grad, sigma, h, chunks, overlap=0, nb_workers=None):
    """Segments nuclei in parallel using overlapping chunks

    Parameters
    ----------
    in_arr : array-like
        Zarr or SharedMemory array of nuclei image
    out_arr : array_like
        Zarr of SharedMemory array for labeled segmentation result
    w_const : float
        constance edge weight for graph cuts foreground mask
    w_grad : float
        gradient-dependend weight for graph cuts foreground mask
    sigma : float or tuple
        amount to smooth the input image before segmentation calcualtions
    h : float
        height of extended maxima in seed calculation
    chunks : tuple
        size of the chunks to processes in each worker
    overlap : int
        number of pixels to overlap adjacent chunks. Default, 0
    nb_workers : int, optional
        number of parallel processes to use. Default, cpu_count

    Returns
    -------
    nb_nuclei : int
        number of nuclei detected

    """
    if overlap == 0:
        warn('Using zero overlap, may see artifacts at chunk boundaries')
    if nb_workers is None:
        nb_workers = multiprocessing.cpu_count()

    print('calculating the histogram')
    hist_output = calculate_histogram(in_arr, chunks, nb_workers)  # 8-bit histogram

    print('estimating foreground and background levels')
    _, freq, _ = hist_output
    threshold = mean_threshold(freq) * (2/3)  # Heuristic threshold
    back_mu, obj_mu = graphcuts.fit_poisson_mixture(hist_output, threshold)
    if in_arr.dtype != 'uint8':  # checking if 12-bit
        back_mu *= (4095 / 255)
        obj_mu *= (4095 / 255)
    print('threshold {}, back_mu {}, obj_mu {}'.format(threshold, back_mu, obj_mu))

    f = partial(_segment_chunk,
                out_arr=out_arr,
                overlap=overlap,
                back_mu=back_mu,
                obj_mu=obj_mu,
                w_const=w_const,
                w_grad=w_grad,
                sigma=sigma,
                h=h)

    arr = out_arr[:]  # Need to load into memory to do the labeling
    label_image = ndi.label(arr, morphology.cube(width=3))[0]
    out_arr[:] = label_image  # overwrite the binary seg with labeled seg
    nb_nuclei_list = utils.pmap_chunks(f, in_arr, chunks, nb_workers)
    return sum(nb_nuclei_list)


def segment_chunks(working_dir, nb_workers, upsampling, sigma, h, T):
    input_dir = os.path.join(working_dir, 'input/')
    foreground_dir = os.path.join(working_dir, 'foreground/')
    direction_dir = os.path.join(working_dir, 'direction/')
    distance_dir = os.path.join(working_dir, 'distance/')
    segmentation_dir = os.path.join(working_dir, 'segmentation/')

    foregorund_abspath = utils.make_dir(foreground_dir)
    direction_abspath = utils.make_dir(direction_dir)
    distance_abspath = utils.make_dir(distance_dir)
    segmentation_abspath = utils.make_dir(segmentation_dir)

    input_paths, input_files = utils.tifs_in_dir(input_dir)
    args = []
    for i, (input_path, input_file) in enumerate(zip(input_paths, input_files)):
        output_paths = {
            'foreground_path': os.path.join(foregorund_abspath, input_file),
            'direction_path': os.path.join(direction_abspath, input_file),
            'distance_path': os.path.join(distance_abspath, input_file),
            'segmentation_path': os.path.join(segmentation_abspath, input_file),
        }
        args.append((input_path, output_paths, upsampling, sigma, h, T))

    with multiprocessing.Pool(processes=nb_workers) as pool:
        pool.starmap(segment, args)

    # Copy over the metadata
    shutil.copy(os.path.join(input_dir, 'metadata.pkl'), segmentation_abspath)

    print('Done!')


def _add_parser_args(parser):
    parser.add_argument(
        "--input-path",
        help="Path to the image to be segmented")
    parser.add_argument(
        "--output-path",
        help="Path to the segmentation file to be generated")
    parser.add_argument(
        "--centroids-path",
        help="Path to the .npy file containing the centroids of "
             "detected cells"
    )
    parser.add_argument(
        "--upsampling",
        default="1,1,1")
    parser.add_argument(
        "--sigma",
        default="1.5,1.5,1.5"
    )
    parser.add_argument(
        "--maxima-suppression-threshold",
        type=float,
        default=0.01)
    parser.add_argument(
        "--foreground-threshold",
        type=float,
        default=0.01)
    parser.add_argument(
        "--n-workers",
        type=int,
        default=8
    )
    parser.add_argument(
        "--min-voxels", type=int, default=0
    )
    return parser


def main():
    parser = argparse.ArgumentParser()
    parser = _add_parser_args(parser)
    args = parser.parse_args()

    upsampling = list(map(float, args.upsampling.split(",")))
    sigma = list(map(float, args.sigma.split(",")))
    h = args.maxima_suppression_threshold
    minT = args.foreground_threshold
    if True:
        binary_seg = segment(
            args.input_path, {}, upsampling, sigma, h, minT)
    # TODO: make segmentation in chunks into an option
    else:
        working_dir = '../data/control/'
        nb_workers = args.n_workers
        segment_chunks(working_dir, nb_workers, upsampling, sigma, h, minT)
        binary_seg = partition.combine_chunks(os.path.join(working_dir, 'segmentation/'))

    centroids_file = args.centroids_path
    labeled_seg, count = ndi.label(binary_seg)
    #
    # Relabel the segmentation consecutively
    #
    if args.min_voxels > 0 and count > 0:
        voxel_counts = np.bincount(labeled_seg[labeled_seg != 0])
        w = np.where(voxel_counts[1:] >= args.min_voxels)[0]
        if len(w) == 0:
            labeled_seg[:] = 0
        else:
            relabel = np.zeros(len(voxel_counts), np.uint32)
            relabel[w+1] = np.arange(len(w)) + 1
            labeled_seg = relabel[labeled_seg]
    imageio.imsave(args.output_path, labeled_seg.astype(np.uint32))
    centroids = find_centroids(labeled_seg)
    np.save(file=centroids_file, arr=centroids)

#
# Monkey-patch eigvals_of_weingarten if PyTorch is available.
#
try:
    import torch
    if torch.cuda.is_available():
        cpu_eigvals_of_weingarten = eigvals_of_weingarten
        from .torch_impl import eigvals_of_weingarten
except:
    pass

if __name__ == '__main__':
    main()
