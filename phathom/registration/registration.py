from itertools import product
from functools import partial
import numpy as np
import zarr
from numcodecs import Blosc
from skimage import feature, filters
import multiprocessing
from scipy.optimize import minimize, basinhopping, differential_evolution
from scipy.ndimage import map_coordinates
from skimage.external import tifffile
from skimage.filters import threshold_otsu
import tqdm
import time
from phathom import utils
from phathom.registration import pcloud
from phathom.io import conversion
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


def plot_pts(pts1, pts2=None, alpha=1, candid1=None, candid2=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(pts1[:,2], pts1[:,1], pts1[:,0], c='b', marker='o', label='Stationary', alpha=alpha)

    if pts2 is not None:
        ax.scatter(pts2[:,2], pts2[:,1], pts2[:,0], c='r', marker='o', label='Moving', alpha=alpha)

    if candid1 is not None and candid2 is not None:
        for i in range(candid1.shape[0]):
            x = [candid1[i,2], candid2[i, 2]]
            y = [candid1[i,1], candid2[i, 1]]
            z = [candid1[i,0], candid2[i, 0]]
            ax.plot(x, y, z, c='g', alpha=0.5)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    if pts2 is not None:
        max_xyz = max(pts1.max(), pts2.max())
        min_xyz = min(pts1.min(), pts2.min())
    else:
        max_xyz = pts1.max()
        min_xyz = pts1.min()
    ax.set_xlim(min_xyz, max_xyz)
    ax.set_ylim(min_xyz, max_xyz)
    ax.set_zlim(min_xyz, max_xyz)
    ax.legend()
    plt.show()


def plot_densities(fixed, moving=None, registered=None, z=0, mip=False, clim=None):
    if mip:
        fixed_img = fixed.max(axis=0)
    else:
        fixed_img = fixed[z]
    plt.imshow(fixed_img, clim=clim)
    plt.show()
    if moving is not None:
        if mip:
            moving_img = moving.max(axis=0)
        else:
            moving_img = moving[z]
        plt.imshow(moving_img, clim=clim)
        plt.show()
    if registered is not None:
        if mip:
            registered_img = registered.max(axis=0)
        else:
            registered_img = registered[z]
        plt.imshow(registered_img, clim=clim)
        plt.show()


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
    _, starts, stops = chunk_bboxes(z_arr.shape, z_arr.chunks, overlap)
    for start, stop in zip(starts, stops):
        yield start, stop


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
    if np.any(smoothed > 0):
        thresh = threshold_otsu(smoothed)
    else:
        thresh = 1.0  # Otsu fails if all voxels are black
    peaks = feature.peak_local_max(smoothed,
                                   min_distance=min_distance,
                                   threshold_abs=max(min_intensity, thresh))
    # Note that using mean here can introduce from chunk artifacts
    return peaks


def detect_blobs_parallel(z_arr, sigma, min_distance, min_intensity, overlap, nb_workers):
    """ Detects blobs in a chunked zarr array in parallel using local maxima

    :param z_arr: input zarr array
    :param sigma: float for gaussian blurring
    :param min_distance: minimum distance in voxels allowed between blobs
    :param min_intensity: minimum gray-level intensity allowed for blobs
    :param overlap: int indicating how much overlap to include between adjacent chunks
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
    with multiprocessing.Pool(nb_workers) as pool:
        r = list(tqdm.tqdm(pool.imap(detect_blobs_in_chunk, chunks),
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


def mark_pts(arr, pts):
    """ Mark a list of points in an array using 3-voxel cubes

    :param arr: Input array to modify
    :param pts: Points to mark
    :return: Original array with the blob marked
    """
    for pt in pts:
        arr[pt[0], pt[1], pt[2]] = 255
        if 0 < pt[0] < arr.shape[0]:
            if 0 < pt[1] < arr.shape[1]:
                if 0 < pt[2] < arr.shape[2]:
                    arr[pt[0]-1:pt[0]+1, pt[1]-1:pt[1]+1, pt[2]-1:pt[2]+1] = 255
    return arr


def label_pts(arr, pts):
    """ Mark a list of points in an array using 3-voxel cubes

    :param arr: Input array to modify
    :param pts: Points to label
    :return: Original array with the blob marked
    """
    for i, pt in enumerate(pts):
        label = i+1
        arr[pt[0], pt[1], pt[2]] = label
        if 2 < pt[0] < arr.shape[0]-3:
            if 2 < pt[1] < arr.shape[1]-3:
                if 2 < pt[2] < arr.shape[2]-3:
                    arr[pt[0]-3:pt[0]+3, pt[1]-3:pt[1]+3, pt[2]-3:pt[2]+3] = label
    return arr


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


def interpolate(image, coordinates, order=3):
    output = map_coordinates(image,
                             coordinates.T,
                             output=None,
                             order=order,
                             mode='constant',
                             cval=0.0,
                             prefilter=True)
    return output


def rotation_matrix(thetas):
    rz = np.eye(3)
    rz[1, 1] = np.cos(thetas[0])
    rz[2, 2] = np.cos(thetas[0])
    rz[1, 2] = -np.sin(thetas[0])
    rz[2, 1] = np.sin(thetas[0])
    ry = np.eye(3)
    ry[0, 0] = np.cos(thetas[1])
    ry[2, 2] = np.cos(thetas[1])
    ry[0, 2] = np.sin(thetas[1])
    ry[2, 0] = -np.sin(thetas[1])
    rx = np.eye(3)
    rx[0, 0] = np.cos(thetas[2])
    rx[1, 1] = np.cos(thetas[2])
    rx[0, 1] = -np.sin(thetas[2])
    rx[1, 0] = np.sin(thetas[2])
    return rz.dot(ry).dot(rx)


def ncc(fixed, registered):
    idx = np.where(registered>0)
    a = fixed[idx]
    b = registered[idx]
    return np.sum((a-a.mean())*(b-b.mean())/((a.size-1)*a.std()*b.std()))


def mean_square_error(fixed, transformed):
    idx = np.where(transformed > 0)
    a = fixed[idx]
    b = transformed[idx]
    return np.mean(np.linalg.norm(a-b))


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
    # TODO: Figure out why this is slow
    if batch_size is None:
        moving_coords = transformation(pts=global_coords)
    else:
        moving_coords = np.empty_like(global_coords)
        nb_pts = len(global_coords)
        nb_batches = int(np.ceil(nb_pts/batch_size))
        for i in range(nb_batches):
            batch_start = i*batch_size
            if i == nb_batches-1:
                batch_stop = nb_pts
            else:
                batch_stop = batch_start + batch_size
            moving_coords[batch_start:batch_stop] = transformation(pts=global_coords[batch_start:batch_stop])

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
    interp_values = interpolate(moving_data, moving_coords_local, order=1)
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


def genetic_optimization():
    # res = differential_evolution(registration_objective,
    #                              bounds=[(0, 2*np.pi) for _ in range(3)],
    #                              args=(fixed_density, moving_density),
    #                              strategy='best1bin',
    #                              maxiter=1000,
    #                              popsize=15,
    #                              tol=1e-3,
    #                              mutation=(0.5, 1.5),
    #                              recombination=0.5,
    #                              disp=True)
    # thetas = res.x
    # print('converged theta (deg): {}'.format(thetas / np.pi * 180))
    # print('Final correlation coefficient: {}'.format(-res.fun))
    pass


def svd_coarse_alignment():
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
    pass


def rigid_registration():
    # print('applying rigid transformation to all fixed points')
    # fixed_registered = rigid_transformation(t, r, fixed_pts)
    # pcloud.plot_pts(fixed_registered, moving_pts)

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
    pass


def pcloud_hist(pts, dimensions, bin_size):
    # Find CoM (TUP offsets in um)
    centroid = pts.mean(axis=0)

    # Calculate histogram bins and bounds
    bins = np.ceil(dimensions / bin_size)
    density_dim = bins * bin_size
    density_excess = density_dim - dimensions
    density_range = np.array((-density_excess/2, dimensions + density_excess / 2))

    # print('  center of mass (um): {}'.format(centroid))
    # print('  # of bins: {}'.format(bins))
    # print('  density dimensions: {}'.format(density_dim))
    # print('  density excess: {}'.format(density_excess))
    # print('  density range (um): {}'.format(density_range))

    # Calculate histograms
    density, edges = np.histogramdd(pts, bins=bins, range=density_range.T)

    return density

def main():
    # Input images
    voxel_dimensions = (2.0, 1.6, 1.6)
    fixed_zarr_path = '/media/jswaney/Drive/Justin/coregistration/1mm_slice/fixed.zarr'
    moving_zarr_path = '/media/jswaney/Drive/Justin/coregistration/1mm_slice/moving.zarr'
    registered_zarr_path = '/media/jswaney/Drive/Justin/coregistration/1mm_slice/registered_affine.zarr'
    preview_zarr_path = '/media/jswaney/Drive/Justin/coregistration/1mm_slice/registered_preview.zarr'
    preview_tif_path = '/media/jswaney/Drive/Justin/coregistration/1mm_slice/registered_preview.tif'
    # Caching intermediate data
    fixed_pts_path = '/media/jswaney/Drive/Justin/coregistration/1mm_slice/fixed_pts.npy'
    moving_pts_path = '/media/jswaney/Drive/Justin/coregistration/1mm_slice/moving_pts.npy'
    fixed_pts_img_path = '/media/jswaney/Drive/Justin/coregistration/1mm_slice/fixed_pts.tif'
    moving_pts_img_path = '/media/jswaney/Drive/Justin/coregistration/1mm_slice/moving_pts.tif'
    fixed_matches_img_path = '/media/jswaney/Drive/Justin/coregistration/1mm_slice/fixed_matches.tif'
    moving_matches_img_path = '/media/jswaney/Drive/Justin/coregistration/1mm_slice/moving_matches.tif'
    fixed_features_path = '/media/jswaney/Drive/Justin/coregistration/1mm_slice/fixed_features.npy'
    moving_features_path = '/media/jswaney/Drive/Justin/coregistration/1mm_slice/moving_features.npy'
    fixed_idx_path = '/media/jswaney/Drive/Justin/coregistration/1mm_slice/fixed_idx.npy'
    moving_idx_path = '/media/jswaney/Drive/Justin/coregistration/1mm_slice/moving_idx.npy'
    # Processing
    nb_workers = 2
    overlap = 8
    # Keypoints
    sigma = (1.2, 2.0, 2.0)
    min_distance = 3
    min_intensity = 600
    # Density maps
    bin_size_um = 200
    sigma_density = 2.0
    niter = 100
    # Matching
    prominence_thresh = 0.2
    max_distance = 300
    max_feat_dist = 1.0
    dist_thresh = None
    # Transform estimation (RANSAC)
    min_samples = 12
    residual_threshold = 2
    # Interpolation
    batch_size = 10000
    # Output
    compression = 1

    t0 = time.time()

    print('opening input images')
    fixed_img = zarr.open(fixed_zarr_path, mode='r')
    moving_img = zarr.open(moving_zarr_path, mode='r')

    # print('detecting keypoints')
    # t1 = time.time()
    # fixed_pts = detect_blobs_parallel(fixed_img, sigma, min_distance, min_intensity, nb_workers, overlap)
    # moving_pts = detect_blobs_parallel(moving_img, sigma, min_distance, min_intensity, nb_workers, overlap)
    # t2 = time.time()
    # print('  found {} keypoints in fixed image'.format(len(fixed_pts)))
    # print('  found {} keypoints in moving image'.format(len(moving_pts)))
    # print('  Took {0:.2f} seconds'.format(t2-t1))
    #
    # print('saving keypoint locations and images')
    # np.save(fixed_pts_path, fixed_pts)
    # np.save(moving_pts_path, moving_pts)
    # fixed_blob_arr = mark_pts(np.zeros(fixed_img.shape, dtype='uint8'), fixed_pts)
    # moving_blob_arr = mark_pts(np.zeros(moving_img.shape, dtype='uint8'), moving_pts)
    # conversion.imsave(fixed_pts_img_path, fixed_blob_arr, compress=1)
    # conversion.imsave(moving_pts_img_path, moving_blob_arr, compress=1)

    print('loading precalculated keypoints')
    fixed_pts = np.load(fixed_pts_path)
    moving_pts = np.load(moving_pts_path)

    # print('extracting features')
    # t1 = time.time()
    # fixed_features = pcloud.geometric_features(fixed_pts_um, nb_workers)
    # moving_features = pcloud.geometric_features(moving_pts_um, nb_workers)
    # t2 = time.time()
    # print('  Took {0:.2f} seconds'.format(t2 - t1))
    #
    # print('saving features')
    # np.save(fixed_features_path, fixed_features)
    # np.save(moving_features_path, moving_features)

    print('loading precalculated features')
    fixed_features = np.load(fixed_features_path)
    moving_features = np.load(moving_features_path)

    print('performing coarse registration')
    t1 = time.time()
    # Convert to um wrt top-upper-left (TUP)
    fixed_img_shape_um = np.array(fixed_img.shape) * np.array(voxel_dimensions)
    moving_img_shape_um = np.array(moving_img.shape) * np.array(voxel_dimensions)
    fixed_pts_um = fixed_pts * np.array(voxel_dimensions)
    moving_pts_um = moving_pts * np.array(voxel_dimensions)
    print('  mixed image dimensions (um): {}'.format(fixed_img_shape_um))
    print('  moving image dimensions (um): {}'.format(moving_img_shape_um))

    # Make the moving histogram (coarse registration target)
    moving_centroid_um = moving_pts_um.mean(axis=0)
    moving_density = pcloud_hist(moving_pts_um, fixed_img_shape_um, bin_size_um)
    moving_density_smooth = filters.gaussian(moving_density, sigma=sigma_density)

    def registration_objective(theta, fixed_pts_um):
        r = rotation_matrix(theta)
        fixed_coords_um_zeroed = fixed_pts_um - fixed_pts_um.mean(axis=0)
        rotated_coords_um_zeroed = rigid_transformation(np.zeros(3), r, fixed_coords_um_zeroed)
        transformed_coords_um = rotated_coords_um_zeroed + moving_centroid_um
        transformed_density = pcloud_hist(transformed_coords_um, fixed_img_shape_um, bin_size_um)
        transformed_density_smooth = filters.gaussian(transformed_density, sigma=sigma_density)
        return -ncc(moving_density_smooth, transformed_density_smooth)

    res = basinhopping(registration_objective,
                       x0=np.array([0, 0, 0]),
                       niter=niter,
                       T=1.0,
                       stepsize=1.0,
                       minimizer_kwargs={
                           'method': 'L-BFGS-B',
                           'args': fixed_pts_um,
                           'bounds': [(0, 2*np.pi) for _ in range(3)],
                           'tol': 0.01,
                           'options': {'disp': False}
                       },
                       disp=True)
    thetas = res.x
    t2 = time.time()
    print('  converged theta (deg): {}'.format(thetas / np.pi * 180))
    print('  Final correlation coefficient: {}'.format(-res.fun))
    print('  Took {0:.2f} seconds'.format(t2 - t1))

    print('transforming the fixed point cloud')
    t1 = time.time()

    def transform_pts(theta, pts_um):
        r = rotation_matrix(theta)
        fixed_coords_um_zeroed = pts_um - pts_um.mean(axis=0)
        rotated_coords_um_zeroed = rigid_transformation(np.zeros(3), r, fixed_coords_um_zeroed)
        transformed_coords_um = rotated_coords_um_zeroed + moving_centroid_um
        return transformed_coords_um

    transformed_pts_um = transform_pts(thetas, fixed_pts_um)
    t2 = time.time()
    print('  Took {0:.2f} seconds'.format(t2-t1))

    # print('matching points')
    # t1 = time.time()
    # fixed_idx, moving_idx = pcloud.match_pts(transformed_pts_um,
    #                                          moving_pts_um,
    #                                          fixed_features,
    #                                          moving_features,
    #                                          max_feat_dist,
    #                                          prominence_thresh,
    #                                          max_distance,
    #                                          nb_workers)
    # print('  found {} matches for {} cells'.format(len(fixed_idx), len(fixed_pts)))
    #
    # print('saving matching indices')
    # np.save(fixed_idx_path, fixed_idx)
    # np.save(moving_idx_path, moving_idx)

    print('Loading cached matches')
    fixed_idx = np.load(fixed_idx_path)
    moving_idx = np.load(moving_idx_path)

    print('indexing...')
    fixed_matches = fixed_pts[fixed_idx]
    moving_matches = moving_pts[moving_idx]

    coarse_error = np.linalg.norm(moving_pts_um[moving_idx]-transformed_pts_um[fixed_idx], axis=-1).mean()
    print('Coarse registration error: {} voxels'.format(coarse_error))

    # print('Saving match images')
    # fixed_matches_img = label_pts(np.zeros(fixed_img.shape, dtype='uint16'), fixed_matches)
    # moving_matches_img = label_pts(np.zeros(moving_img.shape, dtype='uint16'), moving_matches)
    # conversion.imsave(fixed_matches_img_path, fixed_matches_img, compress=1)
    # conversion.imsave(moving_matches_img_path, moving_matches_img, compress=1)

    # pcloud.plot_pts(fixed_pts, moving_pts, candid1=fixed_matches, candid2=moving_matches)

    print('estimating affine transformation with RANSAC')
    t1 = time.time()
    ransac, inlier_idx = pcloud.estimate_affine(fixed_matches,
                                                moving_matches,
                                                min_samples=min_samples,
                                                residual_threshold=residual_threshold)
    transformed_matches = pcloud.register_pts(fixed_matches, ransac)
    average_residual = np.linalg.norm(transformed_matches - moving_matches, axis=-1).mean()
    t2 = time.time()
    print('  {} matches before RANSAC'.format(len(fixed_matches)))
    print('  {} matches remain after RANSAC'.format(len(inlier_idx)))
    print('  Ave. residual: {0:.1f} voxels'.format(average_residual))
    print('  Took {0:.2f} seconds'.format(t2 - t1))

    # print('estimating non-rigid transformation with RBFs')
    # nb_samples = 1000
    #
    # sample_idx = np.random.choice(len(fixed_matches), nb_samples, replace=False)
    # fixed_sample = fixed_matches[sample_idx]
    # moving_sample = moving_matches[sample_idx]
    #
    # correspondence_sample = np.hstack((fixed_sample, moving_sample))
    #
    # from scipy.interpolate import Rbf
    #
    # rbf_z = Rbf(correspondence_sample[:, 0],  # fixed z
    #             correspondence_sample[:, 1],  # fixed y
    #             correspondence_sample[:, 2],  # fixed x
    #             correspondence_sample[:, 3],  # moving z (output)
    #             function='thin-plate',
    #             epsilon=None,
    #             smooth=0)
    #
    # rbf_y = Rbf(correspondence_sample[:, 0],  # fixed z
    #             correspondence_sample[:, 1],  # fixed y
    #             correspondence_sample[:, 2],  # fixed x
    #             correspondence_sample[:, 4],  # moving y (output)
    #             function='thin-plate',
    #             epsilon=None,
    #             smooth=0)
    #
    # rbf_x = Rbf(correspondence_sample[:, 0],  # fixed z
    #             correspondence_sample[:, 1],  # fixed y
    #             correspondence_sample[:, 2],  # fixed x
    #             correspondence_sample[:, 5],  # moving x (output)
    #             function='thin-plate',
    #             epsilon=None,
    #             smooth=0)
    #
    # zm = rbf_z(correspondence_sample[:, 0], correspondence_sample[:, 1], correspondence_sample[:, 2])
    # ym = rbf_y(correspondence_sample[:, 0], correspondence_sample[:, 1], correspondence_sample[:, 2])
    # xm = rbf_x(correspondence_sample[:, 0], correspondence_sample[:, 1], correspondence_sample[:, 2])
    # ave_keypt_resid = np.linalg.norm(np.vstack([zm, ym, xm]).T - moving_sample, axis=-1).mean()
    # print('RBF average residual at keypoints: {0:.1f} voxels'.format(ave_keypt_resid))
    #
    # zm = rbf_z(fixed_matches[:,0], fixed_matches[:,1], fixed_matches[:,2])
    # ym = rbf_y(fixed_matches[:,0], fixed_matches[:,1], fixed_matches[:,2])
    # xm = rbf_x(fixed_matches[:,0], fixed_matches[:,1], fixed_matches[:,2])
    # ave_test_resid = np.linalg.norm(np.vstack([zm, ym, xm]).T - moving_matches, axis=-1).mean()
    # print('RBF average residual on test set: {0:.1f} voxels'.format(ave_test_resid))

    # z = np.linspace(0, fixed_img.shape[0], 10)
    # y = np.linspace(0, fixed_img.shape[1], 10)
    # x = np.linspace(0, fixed_img.shape[2], 10)
    # X, Y, Z = np.meshgrid(x, y, z)
    #
    # zm = rbf_z(Z, Y, X)
    # ym = rbf_y(Z, Y, X)
    # xm = rbf_x(Z, Y, X)
    #
    # print(X.shape, xm.shape)
    # print(zm.max(), ym.max(), xm.max())
    #
    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # ax.view_init(0, 0)
    # ax.quiver(X, Y, Z, xm, ym, zm, length=0.1)
    # plt.show()

    # fixed_inliers = fixed_matches[inlier_idx]
    # moving_inliers = moving_matches[inlier_idx]
    # pcloud.plot_pts(fixed_pts, moving_pts, candid1=fixed_inliers, candid2=moving_inliers)

    # print('registering the moving image')
    # t1 = time.time()
    # output_img = zarr.open(registered_zarr_path,
    #                        mode='w',
    #                        shape=fixed_img.shape,
    #                        chunks=fixed_img.chunks,
    #                        dtype=fixed_img.dtype,
    #                        compressor=Blosc(cname='zstd', clevel=1, shuffle=Blosc.BITSHUFFLE))
    # transformation = partial(pcloud.register_pts, linear_model=ransac)
    # register(moving_img, fixed_img, output_img, transformation, nb_workers, batch_size)
    # t2 = time.time()
    # print('  Took {0:.2f} seconds'.format(t2 - t1))
    #
    # print('downsamplingx the registered zarr array')
    # t1 = time.time()
    # conversion.downsample_zarr(output_img, (4, 4, 4), preview_zarr_path, 44)
    # t2 = time.time()
    # print('  Took {0:.2f} seconds'.format(t2 - t1))
    #
    # print('converting zarr to tiffs')
    # t1 = time.time()
    # conversion.zarr_to_tifs(preview_zarr_path, preview_tif_path, nb_workers, compression)
    # t2 = time.time()
    # print('  Took {0:.2f} seconds'.format(t2 - t1))
    #
    # t3 = time.time()
    # print('Total time: {0:.2f} seconds'.format(t3-t0))


if __name__ == '__main__':
    main()
