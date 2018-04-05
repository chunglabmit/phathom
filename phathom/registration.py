from itertools import product
from functools import partial
import numpy as np
import zarr
from skimage import feature, filters
from scipy.optimize import minimize
from scipy.ndimage import map_coordinates
from skimage.external import tifffile
import tqdm
from phathom import utils
from phathom import pcloud
from phathom import conversion


def chunk_coordinates(shape, chunks):
    """ Calculate the global coordaintes for each chunk's starting position

    :param shape: shape of the image to chunk
    :param chunks: shape of each chunk
    :return: a list containing the starting indices of each chunk
    """
    nb_chunks = utils.chunk_dims(shape, chunks)
    start = []
    for indices in product(*tuple(range(n) for n in nb_chunks)):
        start.append(tuple(i*c for i, c in zip(indices, chunks)))
    return start


def chunk_bboxes(shape, chunks, overlap):
    """ Calculates the bounding box coordinates for each overlapped chunk

    :param shape: overall shape
    :param chunks: tuple containing the shape of each chunk
    :param overlap: int indicating number of voxels to overlap adjacent chunks
    :return: tuple containing the start and stop coordinates for each bbox
    """
    chunk_coords = chunk_coordinates(shape, chunks)
    start = []
    stop = []
    for coord in chunk_coords:
        start.append(tuple(max(0, s-overlap) for s in coord))
        stop.append(tuple(min(e, s+c+overlap) for s,c,e in zip(coord, chunks, shape)))
    return chunk_coords, start, stop


def chunk_generator(z_arr, overlap):
    """ Reads the chunks from a 3D on-disk zarr array

    :param z_arr: input zarr array
    :param overlap: int indicating number of voxels to overlap adjacent chunks
    :return: the next chunk in the zarr array
    """
    # Only one instance of this generator is dishing up chunks -> slow
    # Open the zarr array in the worker processes
    _, starts, stops = chunk_bboxes(z_arr.shape, z_arr.chunks, overlap)
    for start, stop in zip(starts, stops):
        yield start, stop
        # z0, y0, x0 = start
        # z1, y1, x1 = stop
        # yield z_arr[z0:z1, y0:y1, x0:x1]


def detect_blobs(bbox, z_arr, sigma, min_distance, min_intensity):
    """ Detects blobs in an image using local maxima

    :param bbox: tuple of two tuples with start-stop indices of chunk
    :param z_arr: reference to persistent zarr array
    :param sigma: float for gaussian blurring
    :return: an (N,3) ndarray of blob coordinates
    """
    start, stop = bbox
    z0, y0, x0 = start
    z1, y1, x1 = stop
    img = z_arr[z0:z1, y0:y1, x0:x1]

    smoothed = filters.gaussian(img, sigma=sigma, preserve_range=True)

    peaks = feature.peak_local_max(smoothed,
                                   min_distance=min_distance,
                                   threshold_abs=max(min_intensity, smoothed.mean()))
    # Note that using mean here can introduce from chunk artifacts
    return peaks


def detect_blobs_parallel(z_arr, sigma, min_distance, min_intensity, overlap):
    """ Detects blobs in a chunked zarr array in parallel using local maxima

    :param z_arr: input zarr array
    :param sigma: float for gaussian blurring
    :return: an (N,3) ndarray of blob coordinates
    """
    chunk_coords, starts, _ = chunk_bboxes(z_arr.shape, z_arr.chunks, overlap)

    detect_blobs_in_chunk = partial(detect_blobs,
                                    z_arr=z_arr,
                                    sigma=sigma,
                                    min_distance=min_distance,
                                    min_intensity=min_intensity)
    chunks = list(chunk_generator(z_arr, overlap))

    pts_list = []
    r = list(tqdm.tqdm(utils.parallel_map(detect_blobs_in_chunk, chunks),
                       total=len(chunks)))
    for i, pts_local in enumerate(r):
        if len(pts_local) == 0:
            continue
        chunk_coord = np.array(chunk_coords[i])
        start = np.array(starts[i])

        local_start = chunk_coord - start
        local_stop = local_start + np.array(z_arr.chunks)

        idx = np.all(np.logical_and(local_start <= pts_local, pts_local < local_stop), axis=1)
        pts_trim = pts_local[idx]

        pts_list.append(pts_trim + start)
    if len(pts_list) == 0:
        return np.zeros((0, 3))
    return np.concatenate(pts_list)


def pts_to_img(pts, shape, path):
    """ Saves a set of points into a binary image

    :param pts: an (N, D) array with N D-dimensional points
    :param shape: a tuple describing the overall shape of the image
    :param path: path to save the output tiff image
    :return: a uint8 ndarray
    """
    from skimage.external import tifffile
    img = np.zeros(shape, dtype='uint8')
    img[tuple(pts.T)] = 255
    tifffile.imsave(path, img)
    return img


def estimate_rigid(fixed_inliers, moving_inliers):
    # Find centroids
    fixed_centroid = np.mean(fixed_inliers, axis=0)
    moving_centroid = np.mean(moving_inliers, axis=0)
    # Find covariance matrix
    M = np.zeros((3, 3))
    for f, m in zip(fixed_inliers, moving_inliers):
        M += np.outer(f - fixed_centroid, m - moving_centroid)
    # Get rigid transformation
    u, s, vh = np.linalg.svd(M)
    r = vh.T.dot(u.T)
    t = moving_centroid - r.dot(fixed_centroid)
    return t, r


def rigid_transformation(t, r, pts):
    return r.dot(pts.T).T + t


def indices_to_um(pts, voxel_dimensions):
    return np.array([d*pts[:, i] for d, i in zip(voxel_dimensions, range(len(voxel_dimensions)))]).T


def um_to_indices(pts_um, voxel_dimensions):
    return np.array([pts_um[:, i]/d for d, i in zip(voxel_dimensions, range(len(voxel_dimensions)))]).T


def use_um(voxel_dimensions):
    def micron_transformation(transformation):
        def wrapper(pts):
            pts = um_to_indices(pts, voxel_dimensions)
            return transformation(pts=pts)
        return wrapper
    return micron_transformation


def rigid_residuals(t, r, fixed_pts, moving_pts):
    return moving_pts - rigid_transformation(t, r, fixed_pts)


def residuals_to_distances(residuals):
    return np.linalg.norm(residuals, axis=-1)


def average_distance(distances):
    return np.mean(distances)


def shape_to_coordinates(shape):
    indices = np.indices(shape)
    z_idx = indices[0].flatten()
    y_idx = indices[1].flatten()
    x_idx = indices[2].flatten()
    return np.array([z_idx, y_idx, x_idx]).T


def interpolate(image, coordinates):
    output = map_coordinates(image,
                             coordinates.T,
                             output=None,
                             order=1,
                             mode='constant',
                             cval=0.0,
                             prefilter=True)
    return output


def correlation_coef(a, b):
    return np.corrcoef(a.ravel(), b.ravel())[0, -1]


def covariance(a, b):
    return np.cov(a.ravel(), b.ravel())[0, -1]


def square_distance(a, b):
    return np.linalg.norm(a-b)


def abs_difference(a, b):
    return np.sum(np.abs(a-b))


def ncc(fixed, registered):
    idx = np.where(registered>0)
    a = fixed[idx]
    b =registered[idx]
    return np.sum((a-a.mean())*(b-b.mean())/((a.size-1)*a.std()*b.std()))


def rotation_matrix(thetas):
    Rz = np.eye(3)
    Rz[1,1] = np.cos(thetas[0])
    Rz[2,2] = np.cos(thetas[0])
    Rz[1,2] = -np.sin(thetas[0])
    Rz[2,1] = np.sin(thetas[0])

    Ry = np.eye(3)
    Ry[0, 0] = np.cos(thetas[1])
    Ry[2, 2] = np.cos(thetas[1])
    Ry[0, 2] = np.sin(thetas[1])
    Ry[2, 0] = -np.sin(thetas[1])

    Rx = np.eye(3)
    Rx[0, 0] = np.cos(thetas[2])
    Rx[1, 1] = np.cos(thetas[2])
    Rx[0, 1] = -np.sin(thetas[2])
    Rx[1, 0] = np.sin(thetas[2])

    return Rz.dot(Ry).dot(Rx)


def unpack_variables(x):
    t = x[:3]
    thetas = x[3:]
    r = rotation_matrix(thetas)
    return t, r


def pack_variables(t, r):
    return np.concatenate((t, r.flatten()))


def transform_density(density, output_shape, transformation, bin_dimensions):
    pts = shape_to_coordinates(output_shape) # These are indices of the fixed density image
    pts_um = indices_to_um(pts, bin_dimensions)
    warped_coords_um = transformation(pts=pts_um)
    warped_coords = um_to_indices(warped_coords_um, bin_dimensions)
    warped_intensities = interpolate(density, warped_coords)
    return np.reshape(warped_intensities, output_shape)


def density_objective(x, fixed_density, moving_density, bin_dimensions):
    t, r = unpack_variables(x) # these are for pts in um units
    transformation = partial(rigid_transformation, t=t, r=r)
    registered_density = transform_density(moving_density, fixed_density.shape, transformation, bin_dimensions)
    objective = -ncc(fixed_density, registered_density)
    # print('Objective function value: {}'.format(objective))
    return objective


def maximize_density_correlation(fixed_density, moving_density, bin_dimensions, verbose=False):
    res = minimize(fun=density_objective,
                   x0=np.zeros(6),
                   args=(fixed_density, moving_density, bin_dimensions),
                   # method=None,
                   bounds=None,
                   options={'disp': verbose})
    print('Optimization status code {} and exit flag {}'.format(res.status, res.success))
    print('Final correlation coefficient: {}'.format(-res.fun))
    t, r = unpack_variables(res.x)
    return t, r


def register_chunk(moving_img, fixed_img, output_img, transformation, start, batch_size=None):
    # Get dimensions
    chunks = np.array(output_img.chunks)
    img_shape = np.array(fixed_img.shape)

    # Find the appropriate global stop coordinate and chunk shape accounting for boundary cases
    stop = np.minimum(start + chunks, img_shape)
    chunk_shape = np.array([b-a for a, b in zip(start, stop)])

    # Find all global coordinates in the fixed image for this chunk
    local_coords = shape_to_coordinates(chunk_shape)
    global_coords = start + local_coords

    # Find the coordinates on the moving image to be interpolated
    moving_coords = transformation(pts=global_coords)

    # Find the padded bounding box of the warped chunk coordinates
    padding = 4
    transformed_start = tuple(np.floor(moving_coords.min(axis=0)-padding).astype('int'))
    transformed_stop = tuple(np.ceil(moving_coords.max(axis=0)+padding).astype('int'))

    # Read in the available portion data (not indexing outside the moving image boundary)
    moving_start = tuple(max(0, s) for s in transformed_start)
    moving_stop = tuple(min(e, s) for e, s in zip(moving_img.shape, transformed_stop))
    moving_coords_local = moving_coords - np.array(moving_start)
    moving_data = moving_img[moving_start[0]:moving_stop[0],
                             moving_start[1]:moving_stop[1],
                             moving_start[2]:moving_stop[2]]

    # interpolate the moving data
    interp_values = interpolate(moving_data, moving_coords_local)
    interp_chunk = np.reshape(interp_values, chunk_shape)

    # write results to disk
    output_img[start[0]:stop[0], start[1]:stop[1], start[2]:stop[2]] = interp_chunk


def register(moving_img, fixed_img, output_img, transformation, nb_workers, batch_size=None):
    start_coords = chunk_coordinates(fixed_img.shape, fixed_img.chunks)

    args_list = []
    for i, start_coord in enumerate(start_coords):
        start = np.array(start_coord)
        args = (moving_img, fixed_img, output_img, transformation, start, batch_size)
        args_list.append(args)

    with multiprocessing.Pool(processes=nb_workers) as pool:
        pool.starmap(register_chunk, args_list)


def main():
    # Input images
    voxel_dimensions = (2.0, 1.6, 1.6)
    fixed_zarr_path = 'D:/Justin/coregistration/fixed/C0/roi1.zarr'
    moving_zarr_path = 'D:/Justin/coregistration/moving/C0/roi1.zarr'
    registered_zarr_path = 'D:/Justin/coregistration/moving/C0/roi3_reg.zarr'
    registered_tif_path = 'D:/Justin/coregistration/moving/C0/roi3_reg.tif'
    # Processing
    overlap = 8
    # Keypoints
    sigma = (1.2, 2.0, 2.0)
    min_distance = 3
    min_intensity = 100
    # Density maps
    bin_size_um = 60
    # Matching
    signif_thresh = 0.4
    dist_thresh = None
    # Transform estimation (RANSAC)
    min_samples = 20
    residual_threshold = 2
    # Interpolation
    batch_size = None

    print('opening input images')
    fixed_img = zarr.open(fixed_zarr_path)
    moving_img = zarr.open(moving_zarr_path)

    print('detecting keypoints')
    fixed_pts = detect_blobs_parallel(fixed_img, sigma, min_distance, min_intensity, overlap)
    moving_pts = detect_blobs_parallel(moving_img, sigma, min_distance, min_intensity, overlap)
    print('found {} keypoints in fixed image'.format(len(fixed_pts)))
    print('found {} keypoints in moving image'.format(len(moving_pts)))

    print('calculating density maps')
    # Convert indices into physical coordiantes in micron
    fixed_pts_um = np.array([dim*fixed_pts[:,i] for dim, i in zip(voxel_dimensions, range(len(voxel_dimensions)))]).T
    moving_pts_um = np.array([dim*moving_pts[:,i] for dim, i in zip(voxel_dimensions, range(len(voxel_dimensions)))]).T
    # Get the physical size of the whole image
    fixed_max_um = np.array([dim*s for dim, s in zip(voxel_dimensions, fixed_img.shape)])
    moving_max_um = np.array([dim*s for dim, s in zip(voxel_dimensions, moving_img.shape)])
    # Calculate the number of bins needed for each image in each dimension
    fixed_bins = np.ceil(fixed_max_um/bin_size_um)
    moving_bins = np.ceil(moving_max_um/bin_size_um)
    # Approximate the point density using a histogram
    fixed_density, fixed_edges = np.histogramdd(fixed_pts_um, bins=fixed_bins)
    moving_density, moving_edges = np.histogramdd(moving_pts_um, bins=moving_bins)

    fixed_density = filters.gaussian(fixed_density, sigma=0.8)
    moving_density = filters.gaussian(moving_density, sigma=0.8)

    bin_dimensions = tuple(bin_size_um for _ in range(3))
    t, r = maximize_density_correlation(fixed_density, moving_density, bin_dimensions, verbose=True)

    registered_density = transform_density(moving_density,
                                           fixed_density.shape,
                                           partial(rigid_transformation, t=t, r=r),
                                           bin_dimensions)

    import matplotlib.pyplot as plt
    plt.imshow(fixed_density[2], clim=[0, 20])
    plt.show()
    plt.imshow(moving_density[2], clim=[0, 20])
    plt.show()
    plt.imshow(registered_density[2], clim=[0, 20])
    plt.show()


    # print('extracting features')
    # fixed_features = pcloud.geometric_features(fixed_pts, nb_workers)
    # moving_features = pcloud.geometric_features(moving_pts, nb_workers)
    #
    # print('matching points')
    # fixed_idx, moving_idx = pcloud.match_pts(fixed_features, moving_features, signif_thresh)
    # fixed_matches = fixed_pts[fixed_idx]
    # moving_matches = moving_pts[moving_idx]
    # print('found {} matches'.format(len(fixed_matches)))
    # # pcloud.plot_pts(fixed_pts, moving_pts, candid1=fixed_matches, candid2=moving_matches)
    #
    # print('estimating affine transformation')
    # ransac, inlier_idx = pcloud.estimate_affine(fixed_matches,
    #                                             moving_matches,
    #                                             min_samples=min_samples,
    #                                             residual_threshold=residual_threshold)
    # fixed_inliers = fixed_matches[inlier_idx]
    # moving_inliers = moving_matches[inlier_idx]
    # print('{} matches remain after RANSAC'.format(len(inlier_idx)))
    # # pcloud.plot_pts(fixed_pts, moving_pts, candid1=fixed_inliers, candid2=moving_inliers)
    #
    # print('performing coarse alignment')
    # print('pass #1')
    # t, r = estimate_rigid(fixed_inliers, moving_inliers)
    # # Find average residual
    # coarse_residuals = rigid_residuals(t, r, fixed_inliers, moving_inliers)
    # coarse_distances = residuals_to_distances(coarse_residuals)
    # coarse_ave_distance = average_distance(coarse_distances)
    # print('average error: {0:.1f} voxels'.format(coarse_ave_distance))
    # if dist_thresh is None:
    #     import matplotlib.pyplot as plt
    #     plt.hist(coarse_distances, bins=100)
    #     plt.show()
    #     dist_thresh = coarse_ave_distance
    #     print('using {} voxels for distance threshold'.format(dist_thresh))
    # inliers_dist_idx = np.where(coarse_distances <= dist_thresh)
    # fixed_inliers_dist = fixed_inliers[inliers_dist_idx]
    # moving_inliers_dist = moving_inliers[inliers_dist_idx]
    # print('{} matches remain after distance filtering'.format(len(fixed_inliers_dist)))
    # print('pass #2')
    # t, r = estimate_rigid(fixed_inliers_dist, moving_inliers_dist)
    # coarse_residuals = rigid_residuals(t, r, fixed_inliers_dist, moving_inliers_dist)
    # coarse_distances = residuals_to_distances(coarse_residuals)
    # coarse_ave_distance = average_distance(coarse_distances)
    # print('average error: {0:.1f} voxels'.format(coarse_ave_distance))
    #
    # print(r, t)

    # # print('applying rigid transformation to all fixed points')
    # # fixed_registered = rigid_transformation(t, r, fixed_pts)
    # # pcloud.plot_pts(fixed_registered, moving_pts)
    #
    # print('registering the moving image')
    # output_img = zarr.open(registered_zarr_path,
    #                        mode='w',
    #                        shape=fixed_img.shape,
    #                        chunks=fixed_img.chunks,
    #                        dtype=fixed_img.dtype)
    # transformation = partial(rigid_transformation, t=t, r=r)
    # register(moving_img, fixed_img, output_img, transformation, nb_workers, batch_size)
    #
    # print('converting zarr to tiffs')
    # conversion.zarr_to_tifs(registered_zarr_path, registered_tif_path, nb_workers)

if __name__ == '__main__':
    main()
