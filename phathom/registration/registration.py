import uuid
from itertools import product
from functools import partial
import numpy as np
import zarr
from numcodecs import Blosc
from skimage import feature, filters
import multiprocessing
from scipy.optimize import minimize, basinhopping, differential_evolution
from scipy.ndimage import map_coordinates
from scipy import spatial
from scipy.interpolate import Rbf, RegularGridInterpolator
try:
    import tifffile
except:
    from skimage.external import tifffile
from skimage.filters import threshold_otsu
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import tqdm
import time
from functools import partial
from phathom import utils
from phathom.registration import pcloud
#from phathom.io import conversion


def chunk_coordinates(shape, chunks):
    """Calculate the global coordaintes for each chunk's starting position

    Parameters
    ----------
    shape : tuple
        shape of the image to chunk
    chunks : tuple
        shape of each chunk

    Returns
    -------
    start : list
        the starting indices of each chunk

    """
    nb_chunks = utils.chunk_dims(shape, chunks)
    start = []
    for indices in product(*tuple(range(n) for n in nb_chunks)):
        start.append(tuple(i*c for i, c in zip(indices, chunks)))
    return start


def chunk_bboxes(shape, chunks, overlap):
    """Calculates the bounding box coordinates for each overlapped chunk

    Parameters
    ----------
    shape : tuple
        overall shape
    chunks : tuple
        tuple containing the shape of each chunk
    overlap : int
        int indicating number of voxels to overlap adjacent chunks

    Returns
    -------
    chunk_coords : list
        starting indices for each chunk wrt the top-upper-left
    start : list
        starting indices for each bbox with overlap
    stop : list
        stopping indices for each bbox with overlap

    """
    chunk_coords = chunk_coordinates(shape, chunks)
    start = []
    stop = []
    for coord in chunk_coords:
        start.append(tuple(max(0, s-overlap) for s in coord))
        stop.append(tuple(min(e, s+c+overlap) for s,c,e in zip(coord, chunks, shape)))
    return chunk_coords, start, stop


def chunk_generator(z_arr, overlap):
    """Reads the chunks from a 3D on-disk zarr array

    Parameters
    ----------
    z_arr : zarr
        input zarr array
    overlap : int
        int indicating number of voxels to overlap adjacent chunks

    Yields
    ------
    start : tuple
        starting index of the next chunk in the `z_arr`
    stop : tuple
        stopping index of the next chunk in the `z_arr`

    """
    _, starts, stops = chunk_bboxes(z_arr.shape, z_arr.chunks, overlap)
    for start, stop in zip(starts, stops):
        yield start, stop


def detect_blobs(bbox, z_arr, sigma, min_distance, min_intensity):
    """Detects blobs in an image using local maxima

    Parameters
    ----------
    bbox : tuple
        tuple of two tuples with start-stop indices of chunk
    z_arr : zarr
        reference to persistent zarr array
    sigma : float
        float for gaussian blurring

    Returns
    -------
    array
        (N,3) ndarray of blob coordinates

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
    """Detects blobs in a chunked zarr array in parallel using local maxima

    Parameters
    ----------
    z_arr : zarr
        input zarr array
    sigma : float
        float for gaussian blurring
    min_distance : float
        minimum distance in voxels allowed between blobs
    min_intensity : float
        minimum gray-level intensity allowed for blobs
    overlap : float
        int indicating how much overlap to include between adjacent chunks
    nb_workers : int
        number of parallel processes to use

    Returns
    -------
    array
        (N,3) ndarray of blob coordinates

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
        r = list(tqdm.tqdm(pool.imap(detect_blobs_in_chunk, chunks), total=len(chunks)))

    for i, pts_local in enumerate(r):
        if len(pts_local) == 0:
            continue

        chunk_coord = np.asarray(chunk_coords[i])
        start = np.asarray(starts[i])

        local_start = chunk_coord - start
        local_stop = local_start + np.array(z_arr.chunks)

        idx = np.all(np.logical_and(local_start <= pts_local, pts_local < local_stop), axis=1)
        pts_trim = pts_local[idx]

        pts_list.append(pts_trim + start)

    if len(pts_list) == 0:
        return np.zeros((0, 3))
    else:
        return np.concatenate(pts_list)


def pts_to_img(pts, shape, path):
    """Saves a set of points into a binary image

    Parameters
    ----------
    pts: an (N, D) array with N D-dimensional points
    shape: a tuple describing the overall shape of the image
    path: path to save the output tiff image

    Returns
    -------
    img : array
        An 8-bit image array

    """
    from skimage.external import tifffile
    img = np.zeros(shape, dtype='uint8')
    img[tuple(pts.T)] = 255
    tifffile.imsave(path, img)
    return img


def mark_pts(arr, pts, cval=None):
    """Mark a list of points in an array using 3-voxel cubes

    Parameters
    ----------
    arr : array
        Input array to modify
    pts : array
        Points to mark
    cval : int, optional
        fill value, defaults to unique labels

    Returns
    -------
    arr : array
        Original array with the blob marked

    """

    for i, pt in enumerate(pts):
        if cval is None:
            label = i+1
        else:
            label = cval
        arr[pt[0], pt[1], pt[2]] = label
        if 1 < pt[0] < arr.shape[0]-2:
            if 1 < pt[1] < arr.shape[1]-2:
                if 1 < pt[2] < arr.shape[2]-2:
                    arr[pt[0]-2:pt[0]+2, pt[1]-2:pt[1]+2, pt[2]-2:pt[2]+2] = cval
    return arr


def estimate_rigid(fixed_inliers, moving_inliers):
    """Estimate a rigid transformation from fixed to moving points using SVD

    Parameters
    ----------
    fixed_inliers : array
        array (N, 3) of fixed coordinates
    moving_inliers: array
        array (N, 3) of corresponding moving coordinates

    Returns
    -------
    t : array
        translation vector
    r : array
        rotation matrix

    """
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


def indices_to_um(pts, voxel_dimensions):
    """Convert indicies to micron units wrt the top-upper-left

    Parameters
    ----------
    pts : array
        2D array (N, D) of ints to convert
    voxel_dimensions : array
        1D array (D,) of floats representing voxel shape

    """
    return np.array([d*pts[:, i] for d, i in zip(voxel_dimensions, range(len(voxel_dimensions)))]).T


def um_to_indices(pts_um, voxel_dimensions):
    """Convert micron units wtf top-upper-left to indices

    Parameters
    ----------
    pts_um : array
        2D array (N, D) of floats in micron to convert
    voxel_dimensions: array
        1D array (D,) of floats representing voxel shape

    """
    return np.array([pts_um[:, i]/d for d, i in zip(voxel_dimensions, range(len(voxel_dimensions)))]).T


def rigid_transformation(t, r, pts):
    """Apply rotation and translation (rigid transformtion) to a set of points

    Parameters
    ----------
    t : array
        1D array representing the translation vector
    r : array
        2D array representing the rotation matrix

    """
    return r.dot(pts.T).T + t


def rigid_residuals(t, r, fixed_pts, moving_pts):
    """Compute the residuals for all points after the rigid transformation

    Parameters
    ----------
    t : array
        1D array (D,) of the translation
    r : array
        2D array (D, D) of the rotation matrix
    fixed_pts : array
        2D array (N, D) of points to transform
    moving_pts : array
        2D array (N, D) of target points

    """
    return moving_pts - rigid_transformation(t, r, fixed_pts)


def residuals_to_distances(residuals):
    """Compute the Euclidean distances given residuals in each dimension

    Parameters
    ----------
    residuals : array
        2D array (N, D) of residuals

    """
    return np.linalg.norm(residuals, axis=-1)


def average_distance(distances):
    """Compute the average Euclidean distance over a sequence of distances

    Parameters
    ----------
    distances : array
        1D array (N,) of distances

    """
    return np.mean(distances)


def shape_to_coordinates(shape):
    """Build an array containing all array indices for a given shape

    Parameters
    ----------
    shape : array-like
        array-like containing 3 ints representing the array shape

    """
    indices = np.indices(shape)
    coords = indices.reshape((indices.shape[0], np.prod(indices.shape[1:]))).T
    return coords


def interpolate(image, coordinates, order=3):
    """Interpolate an image at a list of coordinates

    Parameters
    ----------
    image : array
        array to interpolate
    coordinates : array
        2D array (N, D) of N coordinates to be interpolated
    order : int
        polynomial order of the interpolation (default: 3, cubic)

    """
    output = map_coordinates(image,
                             coordinates.T,
                             output=None,
                             order=order,
                             mode='constant',
                             cval=0.0,
                             prefilter=True)
    return output


def mean_square_error(fixed, transformed):
    """Calculate the nmean squared error between two images

    Parameters
    ----------
    fixed : array
        array of first image to be compared
    transformed : array
        array of second image to be compared

    """
    idx = np.where(transformed > 0)
    a = fixed[idx]
    b = transformed[idx]
    return np.mean(np.linalg.norm(a-b))


transformation = None


def register_slice(moving_img, zslice, output_shape, transformation, batch_size=None, padding=4):
    """Apply transformation and interpolate for a single z slice in the output

    Parameters
    ----------
    moving_img : zarr array
        input image to be interpolated
    zslice : int
        index of the z-slice to compute
    output_shape : tuple
        shape of the output image
    transformation : callable
        mapping from output image coordinates to moving image coordinates
    batch_size : int, optional
        number of points to transform at a time. Default, all coordinates
    padding : int, optional
        amount of padding to use when extracting pixels for interpolation

    Returns
    -------
    registered_img : ndarray
        registered slice from the moving image

    """
    img_shape = np.array(output_shape)
    local_coords = shape_to_coordinates(img_shape)
    global_coords = np.hstack((zslice*np.ones((local_coords.shape[0], 1)), local_coords))

    if batch_size is None:
        moving_coords = transformation(pts=global_coords)
    else:
        moving_coords = np.empty_like(global_coords)
        nb_pts = len(global_coords)
        nb_batches = int(np.ceil(nb_pts/batch_size))
        for i in tqdm.tqdm(range(nb_batches)):
            batch_start = i*batch_size
            if i == nb_batches-1:
                batch_stop = nb_pts
            else:
                batch_stop = batch_start + batch_size
            moving_coords[batch_start:batch_stop] = transformation(pts=global_coords[batch_start:batch_stop])

    # Find the padded bounding box of the warped chunk coordinates
    transformed_start = tuple(np.floor(moving_coords.min(axis=0) - padding).astype('int'))
    transformed_stop = tuple(np.ceil(moving_coords.max(axis=0) + padding).astype('int'))

    # Read in the available portion data (not indexing outside the moving image boundary)
    moving_start = tuple(max(0, s) for s in transformed_start)
    moving_stop = tuple(min(e, s) for e, s in zip(moving_img.shape, transformed_stop))
    moving_coords_local = moving_coords - np.array(moving_start)

    print(moving_start, moving_stop)

    moving_data = moving_img[moving_start[0]:moving_stop[0],
                             moving_start[1]:moving_stop[1],
                             moving_start[2]:moving_stop[2]]

    # interpolate the moving data
    interp_values = interpolate(moving_data, moving_coords_local, order=1)
    registered_img = np.reshape(interp_values, output_shape)
    return registered_img


def register_chunk(moving_img, chunks, output_img, start, fixed_img, batch_size=None, padding=4):
    """Apply transformation and interpolate for a single chunk in the output

    Parameters
    ----------
    moving_img : zarr array
        zarr array with read access to interpolate
    chunks : tuple
        chunk size of the output image (and ideally the fixed image too)
    output_img : zarr array
        zarr array with write access for output
    transformation : callable
        callable that takes a single "pts" argument
    start : tuple
        starting index of the chunk to write
    batch_size : int, optional
        number of points to apply the transformation on at once. Default, whole chunk.
    padding : int, optional
        number of pixels to borrow from adjacent chunks in `fixed_img`. Default, 4.

    """
    global transformation

    # zarr.blosc.use_threads = True

    # Get dimensions
    chunks = np.array(chunks)
    img_shape = np.array(output_img.shape)

    # Find the appropriate global stop coordinate and chunk shape accounting for boundary cases
    stop = np.minimum(start + chunks, img_shape)
    chunk_shape = np.array([b-a for a, b in zip(start, stop)])

    # Check the target to see if we need to do anything
    fixed_data = fixed_img[start[0]:stop[0],
                           start[1]:stop[1],
                           start[2]:stop[2]]
    if not np.any(fixed_data):
        output_img[start[0]:stop[0], start[1]:stop[1], start[2]:stop[2]] = np.zeros(chunk_shape, output_img.dtype)
        return

    # Find all global coordinates in the fixed image for this chunk
    local_coords = shape_to_coordinates(chunk_shape)
    global_coords = start + local_coords

    # Find the coordinates on the moving image to be interpolated
    if batch_size is None:
        moving_coords = transformation(pts=global_coords)  # This is using multiple cores
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
    transformed_start = tuple(np.floor(moving_coords.min(axis=0)-padding).astype('int'))
    transformed_stop = tuple(np.ceil(moving_coords.max(axis=0)+padding).astype('int'))

    if np.any(np.asarray(transformed_stop) < 0):  # Chunk is outside for some dimension
        interp_chunk = np.zeros(chunk_shape, output_img.dtype)
    elif np.any(np.greater(np.asarray(transformed_start), np.asarray(output_img.shape))): # Chunk is outside
        interp_chunk = np.zeros(chunk_shape, output_img.dtype)
    else:
        # Read in the available portion data (not indexing outside the moving image boundary)
        moving_start = tuple(max(0, s) for s in transformed_start)
        moving_stop = tuple(min(e, s) for e, s in zip(moving_img.shape, transformed_stop))
        moving_coords_local = moving_coords - np.array(moving_start)
        moving_data = moving_img[moving_start[0]:moving_stop[0],
                                 moving_start[1]:moving_stop[1],
                                 moving_start[2]:moving_stop[2]]
        if not np.any(moving_data):  # No need to interpolate if moving image is just zeros
            interp_chunk = np.zeros(chunk_shape, dtype=output_img.dtype)
        else:
            # interpolate the moving data
            interp_values = interpolate(moving_data, moving_coords_local, order=1)
            interp_chunk = np.reshape(interp_values, chunk_shape)

    # write results to disk
    output_img[start[0]:stop[0], start[1]:stop[1], start[2]:stop[2]] = interp_chunk


def _register_chunk(args):
    arr, start_coord, chunks, moving_img, fixed_img, batch_size, padding = args
    register_chunk(moving_img, chunks, arr, start_coord, fixed_img, batch_size, padding)


def register(moving_img, output_img, fixed_img, transform_path, nb_workers, batch_size=None, padding=4):
    """Transform a moving zarr array for registration

    Parameters
    ----------
    moving_img: zarr array
        zarr array with read access to be interpolated
    output_img : zarr array
        zarr array with write access for the output
    transformation : callable
        function that takes one "pts" argument and warps them
    nb_workers : int
        number of processes to work on separate chunks
    batch_size : int, optional
        number of points to apply the transformation on at once. Default, whole chunk.
    padding : int, optional
        number of pixels to borrow from adjacent chunks in `fixed_img`. Default, 4.

    """
    global transformation

    # Get transformation
    transformation = utils.pickle_load(transform_path)

    start_coords = chunk_coordinates(output_img.shape, output_img.chunks)
    args_list = []
    for i, start_coord in tqdm.tqdm(enumerate(start_coords)):
        start = np.asarray(start_coord)
        # args = (moving_img, output_img.chunks, output_img, start, batch_size, padding)
        args = (output_img, start, output_img.chunks, moving_img, fixed_img, batch_size, padding)
        args_list.append(args)
        # register_chunk(*args)

    with multiprocessing.Pool(processes=nb_workers) as pool:
        list(tqdm.tqdm(pool.imap_unordered(_register_chunk, args_list), total=len(args_list)))

    # f = partial(_register_chunk,
    #             moving_img=moving_img,
    #             batch_size=batch_size,
    #             padding=padding)
    #
    # utils.pmap_chunks(f, output_img, output_img.chunks, nb_workers)



def coherence(n_neighbors, fixed_pts_um, moving_pts_um):
    """Calculate the cosine similarity between displacement vectors using `n_neighbors`
    """
    nbrs = NearestNeighbors(n_neighbors=n_neighbors+1, algorithm='kd_tree', n_jobs=-1)
    nbrs.fit(fixed_pts_um)
    distances, indices = nbrs.kneighbors(fixed_pts_um)

    cosine_similarity = np.zeros((fixed_pts_um.shape[0], n_neighbors))
    for i, idxs in enumerate(indices):
        displacement = moving_pts_um[i] - fixed_pts_um[i]

        neighbor_idxs = idxs[1:]
        fixed_neighbors = fixed_pts_um[neighbor_idxs]
        moving_neighbors = moving_pts_um[neighbor_idxs]
        displacement_neighbors = moving_neighbors - fixed_neighbors

        for j, d in enumerate(displacement_neighbors):
            cosine_similarity[i, j] = 1 - spatial.distance.cosine(displacement, d)

    return cosine_similarity.mean(axis=-1)


def match_distance(pts1, pts2):
    """Calculate the distance between matches points"""
    return np.linalg.norm(pts1-pts2, axis=-1)


def fit_polynomial_transform(fixed_keypts, moving_keypts, degree):
    """Fit a low-order polynomial mapping from fixed to moving keypoints"""
    fixed_poly = PolynomialFeatures(degree=degree).fit_transform(fixed_keypts)
    model_z = LinearRegression(fit_intercept=False).fit(fixed_poly,
                                                        moving_keypts[:, 0])
    model_y = LinearRegression(fit_intercept=False).fit(fixed_poly,
                                                        moving_keypts[:, 1])
    model_x = LinearRegression(fit_intercept=False).fit(fixed_poly,
                                                        moving_keypts[:, 2])
    return model_z, model_y, model_x


def polynomial_transform(pts, degree, model_z, model_y, model_x):
    """Apply a low-order polynomial transformation to pts"""
    poly = PolynomialFeatures(degree=degree).fit_transform(pts)
    transformed_keypts = np.empty_like(pts)
    transformed_keypts[:, 0] = model_z.predict(poly)
    transformed_keypts[:, 1] = model_y.predict(poly)
    transformed_keypts[:, 2] = model_x.predict(poly)
    return transformed_keypts


def fit_rbf(affine_pts, moving_pts, smooth=0, mode='thin_plate'):
    rbf_z = Rbf(affine_pts[:, 0], affine_pts[:, 1], affine_pts[:, 2], moving_pts[:, 0], smooth=smooth, function=mode)
    rbf_y = Rbf(affine_pts[:, 0], affine_pts[:, 1], affine_pts[:, 2], moving_pts[:, 1], smooth=smooth, function=mode)
    rbf_x = Rbf(affine_pts[:, 0], affine_pts[:, 1], affine_pts[:, 2], moving_pts[:, 2], smooth=smooth, function=mode)
    return rbf_z, rbf_y, rbf_x


def rbf_transform(pts, rbf_z, rbf_y, rbf_x):
    zi = rbf_z(pts[:, 0], pts[:, 1], pts[:, 2])
    yi = rbf_y(pts[:, 0], pts[:, 1], pts[:, 2])
    xi = rbf_x(pts[:, 0], pts[:, 1], pts[:, 2])
    return np.column_stack([zi, yi, xi])


def nonrigid_transform(pts, affine_transform, rbf_z, rbf_y, rbf_x):
    affine_pts = affine_transform(pts)
    return rbf_transform(affine_pts, rbf_z, rbf_y, rbf_x)

TRANSFORMS = {}
GRIDS = {}

def wrg_transform(my_uuid, start, end):
    transform = TRANSFORMS[my_uuid]
    grid = GRIDS[my_uuid]
    return transform(grid[start:end])

def warp_regular_grid(np_pts, z, y, x, transform, n_processes=1,
                      chunk_size=250):
    Z, Y, X = np.meshgrid(z, y, x, indexing='ij')
    grid = np.column_stack([Z.ravel(), Y.ravel(), X.ravel()])
    if n_processes == 1:
        values = transform(grid)
    else:
        my_uuid = uuid.uuid4()
        TRANSFORMS[my_uuid] = transform
        GRIDS[my_uuid] = grid
        try:
            n_grid = len(grid)
            my_chunk_size = min((n_grid + n_processes - 1) // n_processes,
                                chunk_size)
            starts = np.arange(0, n_grid, my_chunk_size)
            ends = np.concatenate((starts[1:], [n_grid]))
            with multiprocessing.Pool(n_processes) as pool:
                futures = []
                for start, end in zip(starts, ends):
                    future = pool.apply_async(transform, (grid[start:end],))
                    futures.append(future)
                values = []
                for future in tqdm.tqdm(futures):
                    values.append(future.get())
        finally:
            del TRANSFORMS[my_uuid]
            del GRIDS[my_uuid]
        values = np.concatenate(values)
    grid_shape = values.shape[-1] * (np_pts,)  # If same # for each
    values_z = np.reshape(values[:, 0], grid_shape)
    values_y = np.reshape(values[:, 1], grid_shape)
    values_x = np.reshape(values[:, 2], grid_shape)
    return values_z, values_y, values_x


def fit_grid_interpolator(z, y, x, values):
    interp_z = RegularGridInterpolator((z, y, x), values[0])  # Could be useful to use map_coordinates here instead
    interp_y = RegularGridInterpolator((z, y, x), values[1])
    interp_x = RegularGridInterpolator((z, y, x), values[2])
    return interp_z, interp_y, interp_x


def interpolator(pts, interp):
    interp_z, interp_y, interp_x = interp
    values_z = interp_z(pts)
    values_y = interp_y(pts)
    values_x = interp_x(pts)
    return np.column_stack([values_z, values_y, values_x])


class MapCoordinatesInterpolator:

    def __init__(self, values, shape, order=1):
        self.values = values
        self.shape = shape
        self.order = order

    def __call__(self, pts):
        # pts must be (n, 3)
        # scale pixel coordinates to grid coordinates
        coords = tuple(pts[:, i]/(self.shape[i]-1)*(self.values.shape[i]-1) for i in range(pts.shape[-1]))
        coords = np.asarray(coords)
        results = map_coordinates(self.values, coords, order=self.order)
        return results


def fit_map_interpolator(values, shape, order=1):
    interp_z = MapCoordinatesInterpolator(values[0], shape, order)
    interp_y = MapCoordinatesInterpolator(values[1], shape, order)
    interp_x = MapCoordinatesInterpolator(values[2], shape, order)
    return interp_z, interp_y, interp_x


def main2():
    import os
    import zarr
    from precomputed_tif.zarr_stack import ZarrStack
    from phathom import io
    from phathom.utils import pickle_load

    working_dir = '/home/jswaney/coregistration'

    # Open images
    fixed_zarr_path = 'fixed/zarr_stack/1_1_1'
    moving_zarr_path = 'moving/zarr_stack/1_1_1'

    fixed_img = io.zarr.open(os.path.join(working_dir, fixed_zarr_path), mode='r')
    moving_img = io.zarr.open(os.path.join(working_dir, moving_zarr_path), mode='r')

    # Load the coordinate interpolator
    interpolator_path = 'map_interpolator.pkl'

    interpolator = pickle_load(os.path.join(working_dir, interpolator_path))

    # Create a new zarr array for the registered image
    nonrigid_zarr_path = 'moving/registered/1_1_1'

    nonrigid_img = io.zarr.new_zarr(os.path.join(working_dir, nonrigid_zarr_path),
                                    fixed_img.shape,
                                    fixed_img.chunks,
                                    fixed_img.dtype)

    # Warp the entire moving image
    nb_workers = 1
    batch_size = None
    padding = 2

    register(moving_img,
             nonrigid_img,
             fixed_img,
             os.path.join(working_dir, interpolator_path),
             nb_workers,
             batch_size=batch_size,
             padding=padding)


def main():
    # Working directory
    # project_path = '/media/jswaney/Drive/Justin/coregistration/whole_brain/'
    project_path = '/home/jswaney/coregistration/'

    # Input images
    voxel_dimensions = (2.0, 1.6, 1.6)
    fixed_zarr_path = project_path + 'fixed/zarr_stack/1_1_1'
    moving_zarr_path = project_path + 'moving/zarr_stack/1_1_1'
    # registered_zarr_path = project_path + 'registered_affine.zarr'
    # preview_zarr_path = project_path + 'registered_preview.zarr'
    # preview_tif_path = project_path + 'registered_preview.tif'

    # Caching intermediate data
    fixed_pts_path = project_path + 'fixed_blobs.npy'
    moving_pts_path = project_path + 'moving_blobs_1200.npy'
    # fixed_pts_img_path = project_path + 'fixed_pts.tif'
    # moving_pts_img_path = project_path + 'moving_pts.tif'
    # fixed_matches_img_path = project_path + 'fixed_matches.tif'
    # moving_matches_img_path = project_path + 'moving_matches.tif'
    fixed_features_path = project_path + 'fixed_features.npy'
    moving_features_path = project_path + 'moving_features.npy'
    # fixed_idx_path = project_path + 'fixed_idx.npy'
    # moving_idx_path = project_path + 'moving_idx.npy'

    # Processing
    nb_workers = 48
    overlap = 8

    # Keypoints
    sigma = (1.2, 2.0, 2.0)
    min_distance = 3
    min_intensity = 600

    # Coarse registration
    niter = 100

    # Nuclei matching
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

    # ---------------------------------- #

    t0 = time.time()

    # print('opening input images')
    # fixed_store = zarr.NestedDirectoryStore(fixed_zarr_path)
    # moving_store = zarr.NestedDirectoryStore(moving_zarr_path)
    # fixed_img = zarr.open(fixed_store, mode='r')
    # moving_img = zarr.open(moving_store, mode='r')

    # print('detecting keypoints')
    # t1 = time.time()
    # fixed_pts = detect_blobs_parallel(fixed_img, sigma, min_distance, min_intensity, nb_workers, overlap)
    # moving_pts = detect_blobs_parallel(moving_img, sigma, min_distance, min_intensity, nb_workers, overlap)
    # t2 = time.time()
    # print('  found {} keypoints in fixed image'.format(len(fixed_pts)))
    # print('  found {} keypoints in moving image'.format(len(moving_pts)))
    # print('  Took {0:.2f} seconds'.format(t2-t1))
    #
    # print('saving blob locations')
    # np.save(fixed_pts_path, fixed_pts)
    # np.save(moving_pts_path, moving_pts)

    # print('saving blob images')
    # fixed_blob_arr = mark_pts(np.zeros(fixed_img.shape, dtype='uint8'), fixed_pts)
    # moving_blob_arr = mark_pts(np.zeros(moving_img.shape, dtype='uint8'), moving_pts)
    # conversion.imsave(fixed_pts_img_path, fixed_blob_arr, compress=1)
    # conversion.imsave(moving_pts_img_path, moving_blob_arr, compress=1)

    print('loading precalculated keypoints')
    fixed_pts = np.load(fixed_pts_path)
    moving_pts = np.load(moving_pts_path)
    fixed_pts_um = np.asarray(voxel_dimensions) * fixed_pts
    moving_pts_um = np.asarray(voxel_dimensions) * moving_pts

    print('extracting features')
    t1 = time.time()
    fixed_features = pcloud.geometric_features(fixed_pts_um, nb_workers)
    moving_features = pcloud.geometric_features(moving_pts_um, nb_workers)
    t2 = time.time()
    print('  Took {0:.2f} seconds'.format(t2 - t1))

    print('saving features')
    np.save(fixed_features_path, fixed_features)
    np.save(moving_features_path, moving_features)

    # print('loading precalculated features')
    # fixed_features = np.load(fixed_features_path)
    # moving_features = np.load(moving_features_path)

    # print('performing coarse registration')
    #
    # print(fixed_img.shape, fixed_pts.shape, fixed_features.shape)
    # print(moving_img.shape, moving_pts.shape, moving_features.shape)


    # print('transforming the fixed point cloud')
    # t1 = time.time()
    #
    # def transform_pts(theta, pts_um):
    #     r = rotation_matrix(theta)
    #     fixed_coords_um_zeroed = pts_um - pts_um.mean(axis=0)
    #     rotated_coords_um_zeroed = rigid_transformation(np.zeros(3), r, fixed_coords_um_zeroed)
    #     transformed_coords_um = rotated_coords_um_zeroed + moving_centroid_um
    #     return transformed_coords_um
    #
    # transformed_pts_um = transform_pts(thetas, fixed_pts_um)
    # t2 = time.time()
    # print('  Took {0:.2f} seconds'.format(t2-t1))

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

    # print('Loading cached matches')
    # fixed_idx = np.load(fixed_idx_path)
    # moving_idx = np.load(moving_idx_path)

    # print('Extracting matched points...')
    # fixed_matches = fixed_pts[fixed_idx]
    # moving_matches = moving_pts[moving_idx]

    # coarse_error = np.linalg.norm(moving_pts_um[moving_idx]-transformed_pts_um[fixed_idx], axis=-1).mean()
    # print('Coarse registration error: {} voxels'.format(coarse_error))

    # print('Saving match images')
    # fixed_matches_img = label_pts(np.zeros(fixed_img.shape, dtype='uint16'), fixed_matches)
    # moving_matches_img = label_pts(np.zeros(moving_img.shape, dtype='uint16'), moving_matches)
    # conversion.imsave(fixed_matches_img_path, fixed_matches_img, compress=1)
    # conversion.imsave(moving_matches_img_path, moving_matches_img, compress=1)

    # pcloud.plot_pts(fixed_pts, moving_pts, candid1=fixed_matches, candid2=moving_matches)

    # print('estimating affine transformation with RANSAC')
    # t1 = time.time()
    # ransac, inlier_idx = pcloud.estimate_affine(fixed_matches,
    #                                             moving_matches,
    #                                             min_samples=min_samples,
    #                                             residual_threshold=residual_threshold)
    # transformed_matches = pcloud.register_pts(fixed_matches, ransac)
    # average_residual = np.linalg.norm(transformed_matches - moving_matches, axis=-1).mean()
    # t2 = time.time()
    # print('  {} matches before RANSAC'.format(len(fixed_matches)))
    # print('  {} matches remain after RANSAC'.format(len(inlier_idx)))
    # print('  Ave. residual: {0:.1f} voxels'.format(average_residual))
    # print('  Took {0:.2f} seconds'.format(t2 - t1))

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
    main2()
