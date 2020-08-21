import typing

from phathom.segmentation import partition
from phathom.segmentation import graphcuts
from phathom import utils
from phathom.synthetic import points_to_binary
from phathom.io import conversion as imageio
import argparse
import itertools
import numpy as np
import scipy.ndimage as ndi
import skimage
from skimage import feature, morphology, filters, exposure, transform, measure
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


def gradient(data, zum=1.0, yum=1.0, xum=1.0):
    """Compute the gradient in units of intensity / micron

    :param data: 3d numpy array
    :param zum: size of a voxel in the z direction - defaults to 1
    :param yum: size of a voxel in the y direction - defaults to 1
    :param xum: size of a voxel in the x direction - defaults to 1
    :return: a 4-dimensional matrix with the last dimension being the z==0,
    y==1, x==2 selector of the gradient direction
    """
    fz, fy, fx = np.gradient(data, edge_order=2)
    grad = np.zeros((*data.shape, 3))
    grad[..., 0] = fz / zum
    grad[..., 1] = fy / yum
    grad[..., 2] = fx / xum
    return grad


def hessian_2d(data, sigma):
    l1 = np.zeros(data.shape)
    for i in range(data.shape[0]):
        img = data[i]
        Hyy, Hyx, Hxx = skimage.feature.hessian_matrix(img, sigma=sigma, mode='reflect')
        l1[i], l2 = skimage.feature.hessian_matrix_eigvals(Hyy, Hyx, Hxx)


def hessian(data, zum=1.0, yum=1.0, xum=1.0):
    microns = np.array([zum, yum, xum]).reshape(3, 1, 1, 1)
    grad = np.gradient(data) / microns
    n_dims = len(grad)
    H = np.zeros((*data.shape, n_dims, n_dims))
    for i, first_deriv in enumerate(grad):
        for j in range(i, n_dims):
            second_deriv = np.gradient(first_deriv, axis=j) / microns[j]
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


def weingarten(g, zum=1.0, yum=1.0, xum=1.0):
    grad = gradient(g, zum=zum, yum=yum, xum=xum)
    l = np.sqrt(1+np.linalg.norm(grad, axis=-1))
    H = hessian(g, zum=zum, yum=yum, xum=xum)
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


def eigvals_of_weingarten(g, zum=1.0, yum=1.0, xum=1.0):
    return eigvalsh(weingarten(g, zum=zum, yum=yum, xum=xum))


def convex_seeds(eigvals):
    eig_neg = np.clip(eigvals, None, 0)
    det_neg = eig_neg.prod(axis=-1)
    return det_neg


def positive_curvatures(eigvals):
    eig_pos = np.clip(eigvals, 0, None)
    pos_curv = np.linalg.norm(eig_pos, axis=-1)
    return pos_curv


def seed_probability(eigen_seeds, stdev=None):
    if stdev is None:
        stdev = eigen_seeds.std()
    return 1-np.exp(-eigen_seeds**2/(2*stdev**2))


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


def adaptive_threshold(img,
                       method=filters.threshold_otsu,
                       low_threshold=np.finfo(np.float32).min,
                       high_threshold=np.finfo(np.float32).max,
                       sigma=1,
                       blocksize=50):
    """Derive an adaptive threshold for each voxel

    Apply the method in blocks, smooth the result, then zoom the resulting grid

    :param img: The image to estimate threshold
    :param method: the thresholding method, a function that returns a threshold
    default is otsu_threshold
    :param low_threshold: Only take voxels with intensities above the
    low threshold. Default is everything
    :param high_threshold: Only take voxels with intensities below the
    high threshold. Default is everything.
    :param sigma: Smooth the resulting grid by this sigma. Can be a constant
    or 3-tuple
    :param blocksize: The size of the blocks. Can be a constant or a 3-tuple
    :return: a similarly sized threshold per voxel
    """
    if np.isscalar(blocksize):
        blocksize = (blocksize, blocksize, blocksize)
    if np.isscalar(sigma):
        sigma = (sigma, sigma, sigma)
    grid_z, grid_y, grid_x = [
        np.linspace(
            0, img.shape[idx],
            1 + int((img.shape[idx] + blocksize[idx] - 1) / blocksize[idx]))
        .astype(int)
        for idx in range(3)]
    sz, sy, sx = [len(grid) - 1 for grid in (grid_z, grid_y, grid_x)]
    threshold_grid = np.zeros((sz, sy, sx), img.dtype)
    for xi, yi, zi in itertools.product(range(sx), range(sy), range(sz)):
        (x0, x1), (y0, y1), (z0, z1) = [
            (grid[idx], grid[idx+1])
            for grid, idx in ((grid_x, xi), (grid_y, yi), (grid_z, zi))]
        block = img[z0:z1, y0:y1, x0:x1]
        block = block[(block >= low_threshold) & (block < high_threshold)]
        if len(block) == 0:
            if np.all(block < low_threshold):
                threshold_grid[zi, yi, xi] = low_threshold
            else:
                threshold_grid[zi, yi, xi] = high_threshold
        else:
            try:
                threshold_grid[zi, yi, xi] = method(block)
            except ValueError:
                # Otsu gets here if all the same
                # Set the threshold just above the value.
                if np.all(block == block[0]):
                    threshold_grid[zi, yi, xi] = \
                        block[0] + np.finfo(block.dtype).eps
                else:
                    raise
    if np.any(sigma) > 0:
        threshold_grid = ndi.gaussian_filter(threshold_grid, sigma)
    output = np.ones_like(img)
    zzi = np.linspace(-.5, threshold_grid.shape[0]-.5, img.shape[0])\
        .reshape(img.shape[0], 1, 1)
    yyi = np.linspace(-.5, threshold_grid.shape[1]-.5, img.shape[1])\
        .reshape(1, img.shape[1], 1)
    xxi = np.linspace(-.5, threshold_grid.shape[2]-.5, img.shape[2])\
        .reshape(1, 1, img.shape[2])
    zz, yy, xx = [output * _ for _ in (zzi, yyi, xxi)]
    ndi.map_coordinates(threshold_grid, (zz, yy, xx), output=output,
                        mode="nearest")
    return output


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


def watershed_centers(image, centers, mask, **watershed_kwargs):
    seeds = points_to_binary(tuple(centers.T), image.shape, cval=1)
    markers = ndi.label(seeds)[0]
    labels = morphology.watershed(-image, markers, mask=mask, **watershed_kwargs)
    return labels


def _watershed_probability_chunk(input_tuple, output, centers, mask, overlap, **watershed_kwargs):
    arr, start_coord, chunks = input_tuple

    # extract ghosted chunk of data
    data_overlap, start_ghosted, stop_ghosted = utils.extract_ghosted_chunk(arr, start_coord, chunks, overlap)
    mask_overlap, _, _ = utils.extract_ghosted_chunk(mask, start_coord, chunks, overlap)

    # Find seeds within the ghosted chunk
    centers_internal = utils.filter_points_in_box(centers, start_ghosted, stop_ghosted)
    centers_internal_local = centers_internal - start_ghosted

    # segment the chunk
    labels_overlap = watershed_centers(data_overlap,
                                       centers_internal_local,
                                       mask_overlap,
                                       watershed_line=True)
    binary_overlap = (labels_overlap > 0)
    # binary_overlap_eroded = ndi.binary_erosion(binary_overlap)

    # write the segmentation result
    start_local = start_coord - start_ghosted
    stop_local = np.minimum(start_local + np.asarray(chunks), np.asarray(arr.shape) - start_ghosted)
    # binary_seg = utils.extract_box(binary_overlap_eroded, start_local, stop_local)
    binary_seg = utils.extract_box(binary_overlap, start_local, stop_local)
    stop_coord = start_coord + np.asarray(binary_seg.shape)
    utils.insert_box(output, start_coord, stop_coord, binary_seg)


def watershed_centers_parallel(prob, centers, mask, output, chunks, overlap, nb_workers=None):
    f = partial(_watershed_probability_chunk,
                output=output,
                centers=centers,
                mask=mask,
                overlap=overlap)
    utils.pmap_chunks(f, prob, chunks, nb_workers, use_imap=True)

def detect_blobs(a:np.ndarray,
                 zyxmin:typing.Tuple[int, int, int],
                 zyxmax:typing.Tuple[int, int, int],
                 pad:typing.Tuple[int, int, int],
                 sigma_low:typing.Tuple[float, float, float],
                 sigma_high:typing.Tuple[float, float, float],
                 threshold:float,
                 min_distance:float) -> np.ndarray:
    """
    Detect blobs using the local maxima of a difference of gaussians. This is an alternative to the curvature
    approach.

    :param a: Detect blobs within this array. Although typed as an ndarray, ZARR or Neuroglancer precomputed
    ArrayReader will work.
    :param zyxmin: a 3-tuple of z0, y0 and x0 for the slice to be processed
    :param zyxmax: a 3-tuple of z1, y1 and x1 for the slice to be processed
    :param pad: amount of padding for the block to be processed
    :param sigma_low: the standard deviation for the foreground of the difference of gaussians.
    :param sigma_high: the standard deviation for the background of the difference of gaussians.
    :param threshold: the threshold cutoff for a local maximum
    :param min_distance: the distance a local maximum must be from a value of higher intensity
    :return: an N x 3 array of blob coordinates.
    """
    z0, y0, x0 = zyxmin
    z1, y1, x1 = zyxmax
    z0p, y0p, x0p = [max(0, a - b) for a, b in zip(zyxmin, pad)]
    z1p, y1p, x1p = [min(s, a + b) for s, a, b in zip(a.shape, zyxmax, pad)]
    block = a[z0p:z1p, y0p:y1p, x0p:x1p].astype(np.float32)
    dog = ndi.gaussian_filter(block, sigma_low) - ndi.gaussian_filter(block, sigma_high)
    iradius_z, iradius_y, iradius_x = [int(np.ceil(_)) for _ in min_distance]
    grid = np.mgrid[-iradius_z:iradius_z + 1, -iradius_y:iradius_y + 1, -iradius_x:iradius_x + 1]
    footprint = np.sqrt(np.sum(np.square(grid / np.array(min_distance).reshape(3, 1, 1, 1)), 0)) <= 1
    mask = (ndi.grey_dilation(dog, footprint=footprint) == dog) & (dog > threshold)
    l, c = ndi.label(mask)
    if c == 0:
        return np.zeros((0, 3))
    z, y, x = np.where(l > 0)
    area = np.bincount(l[z, y, x])
    idx = np.where(area > 0)[0]
    area = area[idx]
    xc = np.bincount(l[z, y, x], weights=x)[idx] / area + x0p
    yc = np.bincount(l[z, y, x], weights=y)[idx] / area + y0p
    zc = np.bincount(l[z, y, x], weights=z)[idx] / area + z0p
    mask = (xc >= x0) & (xc < x1) & (yc >= y0) & (yc < y1) & (zc >= z0) & (zc < z1)
    return np.column_stack((zc[mask], yc[mask], xc[mask]))


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
    cpu_eigvals_of_weingarten = eigvals_of_weingarten

if __name__ == '__main__':
    main()
