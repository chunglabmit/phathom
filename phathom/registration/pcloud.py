"""The pcloud module contains functions related to point cloud generation and matching
"""
import itertools
import numpy as np
import scipy
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors
from sklearn import linear_model
from skimage import filters
import multiprocessing
import tqdm


def rotation_matrix(thetas):
    """Create a 3D rotation matrix given rotations about each axis

    Parameters
    ----------
    thetas : array-like
        array-like with 3 rotation angles in radians

    """
    rz = np.eye(3)
    rz[1, 1] = np.cos(thetas[0])
    rz[2, 2] = np.cos(thetas[0])
    rz[1, 2] = np.sin(thetas[0])
    rz[2, 1] = -np.sin(thetas[0])
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


def rotate(points, thetas, center=None):
    """Rotate `points` in 3D

    Parameters
    ----------
    points : tuple
        tuple of coordiante arrays for points to be rotated
    thetas : array-like
        angles (in radians) to rotate the points for each axis
    center : array-like
        coordinates for the center of rotation

    Returns
    -------
    rotated : tuple
        tuple of coordinate arrays for rotated points

    """
    try:
        if len(points) != len(thetas):
            raise ValueError('thetas must contain a rotation for each dimension')
        else:
            d = len(points)
    except TypeError:
        raise ValueError('points and thetas must both be iterable')
    except IndexError:
        ValueError('points tuple must contain at least one non-empty array')

    if center is None:
        center = np.zeros(d)
    else:
        if len(center) == d:
            center = np.asarray(center)
        else:
            raise ValueError('provided center with the wrong number of dimensions')
    points = np.asarray(points)  # d-by-n
    r = rotation_matrix(thetas)
    center = center[:, np.newaxis]
    shifted = points - center
    shifted_rotated = r.dot(shifted)
    rotated = shifted_rotated + center
    return tuple(rotated)


def geometric_hash(center, vectors):
    """Calculates the geometric hash of a set of `vectors` around `center`

    The geometric hash is a rotation invariant descriptor of the relative
    positions of the `vectors` with respect to `center`

    Parameters
    ----------
    center : ndarray
        coorindates of the center point
    vectors : ndarray
        a (3, 3) array of neighboring point coordinates as rows

    Returns
    -------
    features : array
        a (6,) array of the geometric hashing result

    """
    a = np.vstack([vectors[2]-center, vectors[0]-center, vectors[1]-center]).T
    _, r = np.linalg.qr(a)
    d = np.diag(np.sign(np.diag(r)))
    r = d.dot(r)
    features = r[np.triu_indices(3)]
    return features


def geometric_features(pts, nb_workers):
    """Extract the geometric hash for each point in a point cloud

    Parameters
    ----------
    pts : ndarray
        2D array (N, 3) of points
    nb_workers : int
        number of processes to calculate features in parallel

    Returns
    -------
    features : ndarray
        (N, 6) array of geometric features

    """
    nbrs = NearestNeighbors(n_neighbors=4, algorithm='kd_tree',
                            n_jobs=nb_workers).fit(pts)
    distances, indices = nbrs.kneighbors(pts)
    # indices is len(pts) by 3, in order of decreasing distance

    args = []
    for i, (center, row) in enumerate(zip(pts, indices)):
        args.append((center, pts[row[1:]]))

    with multiprocessing.Pool(processes=nb_workers) as pool:
        features = pool.starmap(geometric_hash, args)

    return np.asarray(features)

def permuted_geometric_features(pts, nb_workers, n_neighbors):
    """Extract the geometric hash for each point in a point cloud

    This version takes all permutations of 3 among N nearest neighbors

    Parameters
    ----------
    pts : ndarray
        2D array (N, 3) of points
    nb_workers : int
        number of processes to calculate features in parallel
    n_neighbors : int
        number of nearest neighbors (3+)
    Returns
    -------
    features : ndarray
        (N, 6) array of geometric features

    """
    assert(n_neighbors >= 3)
    nbrs = NearestNeighbors(n_neighbors=n_neighbors+1, algorithm='kd_tree',
                            n_jobs=nb_workers).fit(pts)
    distances, indices = nbrs.kneighbors(pts)
    # indices is len(pts) by n_neighbors+1, in order of decreasing distance
    #
    # Get all combinations
    combinations = np.array(list(
        itertools.combinations(range(1, n_neighbors+1), 3)))
    combinations.sort(1)

    features = []
    with multiprocessing.Pool(processes=nb_workers) as pool:
        for idxs in tqdm.tqdm(combinations):
            args = []
            for i, (center, row) in enumerate(zip(pts, indices)):
                args.append((center, pts[row[idxs]]))
            features.append(np.stack(pool.starmap(geometric_hash, args)))
    return np.stack(features, 1)


def check_distance(dists, max_dist):
    """Check whether or not `dists` are less than `max_dist`

    Parameters
    ----------
    dists : ndarray
        (N,) array with input distances to check
    max_dist : float
        maximum Euclidean distance for a candidate point to be considered close

    Returns
    -------
    close : tuple
        tuple of coordinate arrays for those points that are close.

    """
    if dists.ndim != 1:
        raise ValueError('candidates must be one dimensional')
    indicators = list(map(lambda d: d < max_dist, dists))
    close = np.where(indicators)
    return close


def prominence(d1, d2):
    """Calculate the prominence of set of points based on the two nearest neighbor distances

    Prominence is defined as the ratio of nearest and 2nd nearest neighbor distances.
    Lower prominence values indicate nearest neighbors that "stand out" more.
    This function doesn't verify that all d1 values are less than or equal to d2 values.

    Parameters
    ----------
    d1 : ndarray
        (N,) array containing the distances of the nearest neighbors
    d2 : ndarray
        (N,) array containing the distance of the 2nd-nearest neighbors

    Returns
    -------
    prom : ndarray
        (N,) array of calculated prominences for each point

    """
    if d1.shape != d2.shape:
        raise ValueError('d1 and d2 must have the same shape')
    else:
        if d1.ndim != 1:
            raise ValueError('d1 and d2 must both be one dimensional')

    return d1 / np.clip(d2, 1e-6, None)


def check_prominence(prom, threshold):
    """Check which prominences in `prom` are below a given `threshold`

    Parameters
    ----------
    prom : ndarray
        (N,) array of prominences
    threshold : float
        maximum prominence value for a point to be considered "prominent"

    Returns
    -------
    prominent : tuple
        a tuple of coordinate arrays for the found prominent points

    """
    indicators = list(map(lambda p: p < threshold, prom))
    prominent = np.where(indicators)
    return prominent


def build_neighbors(X, n_neighbors=1, radius=1.0, algorithm='auto', n_jobs=1):
    """Instantiate a NearestNeighbors classifier and fit it to `points`

    Parameters
    ----------
    X : ndarray
        (N, D) array of D-dimensional data points
    n_neighbors : int
        number of neighbors to use for kneighbor queries by default
    radius : float
        range of parameter space to use for radius_neighbor queries by default
    algorithm : {'auto', 'ball_tree', 'kd_tree', 'brute'}, optional
        algorithm used to calculate the nearest neighbors
    n_jobs : int
        number of parallel jobs used to run for kneighbors searches. -1 uses all CPU cores.

    Returns
    -------
    nbrs : NearestNeighbors
        NearestNeighbors object containing a built kd-tree

    """
    nbrs = NearestNeighbors(n_neighbors=n_neighbors,
                            radius=radius,
                            algorithm=algorithm,
                            n_jobs=n_jobs)
    return nbrs.fit(X)


def feature_matching(feat_fixed, feat_moving, max_fdist=None, prom_thresh=None, algorithm='auto', n_jobs=-1):
    """Find moving point matches for all fixed points based on feature distance and prominence

    Parameters
    ----------
    feat_fixed : ndarray
        (N, D) array of D dimensional features for N fixed points
    feat_moving : ndarray
        (M, D) array of D dimensional features for M moving points
    max_fdist : float
        maximum allowed Euclidean distance in feature space to be considered a match
    prom_thresh : float
        maximum allowed prominence ratio to be considered a match
    algorithm : {'auto', 'ball_tree', 'kd_tree', 'brute'}, optional
        algorithm used to compute the nearest neighbors
    n_jobs : int, optional
        number of parallel processes to use for kneighbors queries. Default uses the number of CPU cores

    Returns
    -------
    idx_fixed : ndarray
        indices of fixed point matches. Empty if no matches are found
    idx_moving : ndarray
        indices of moving point matches. Empty if no matches are found

    """
    if feat_fixed.ndim == 1:
        feat_fixed = feat_fixed.reshape(-1, 1)
    try:
        if feat_fixed.ndim != 2:
            raise ValueError('feat_fixed is not two-dimensional')
        if feat_moving.ndim != 2:
            raise ValueError('feat_moving is not two-dimensional')
        if feat_fixed.shape[1] != feat_moving.shape[1]:
            raise ValueError('the second dimension of feat_fixed and feat_moving do not match')
    except TypeError:
        raise ValueError('feat_fixed or feat_moving is not an ndarray')

    if feat_moving.shape[0] == 1:
        n_neighbors = 1
    else:
        n_neighbors = 2

    nbrs = build_neighbors(X=feat_moving, n_neighbors=n_neighbors, algorithm=algorithm, n_jobs=n_jobs)
    fdists, idxs_moving = nbrs.kneighbors(feat_fixed)
    idx_fixed = np.arange(feat_fixed.shape[0])

    # filter on feature distance
    if max_fdist is not None:
        close = check_distance(fdists[:, 0], max_fdist)
        fdists = fdists[close]
        idxs_moving = idxs_moving[close]
        idx_fixed = idx_fixed[close]
        if close[0].size == 0:
            return idx_fixed, idxs_moving[:, 0]

    # filter on prominence
    if prom_thresh is not None and feat_moving.shape[0] > 1:
        prom = prominence(fdists[:, 0], fdists[:, 1])
        prominent = check_prominence(prom, prom_thresh)
        fdists = fdists[prominent]
        idxs_moving = idxs_moving[prominent]
        idx_fixed = idx_fixed[prominent]

    return idx_fixed, idxs_moving[:, 0]


def _feature_matching(input_zip):
    feat_fixed, feat_moving, matching_kwargs = input_zip
    feat_fixed = feat_fixed.reshape(1, -1)
    if feat_moving.shape[0] > 0:
        max_fdist = matching_kwargs.pop('max_fdist', None)
        prom_thresh = matching_kwargs.pop('prom_thresh', None)
        algorithm = matching_kwargs.pop('algorithm', 'auto')
        n_jobs = matching_kwargs.pop('n_jobs', 1)
        matches = feature_matching(feat_fixed, feat_moving,
                                   max_fdist=max_fdist,
                                   prom_thresh=prom_thresh,
                                   algorithm=algorithm,
                                   n_jobs=n_jobs)
    else:
        matches = (np.array([]), np.array([]))
    return matches


def radius_matching(points_fixed, points_moving, feat_fixed, feat_moving, radius, nb_workers=None, batch_size=None, matching_kwargs=None):
    """Match fixed points to moving points within a search radius

    Parameters
    ----------
    points_fixed : ndarray
        (N, D) array of the D spatial coordinates of N fixed points
    points_moving : ndarray
        (M, D) array of the D spatial coordinates of M fixed points
    feat_fixed : ndarray
        (N, F) array of F features for N fixed points
    feat_moving : ndarray
        (M, F) array of F features for M moving points
    radius : float
        maximum distance to search for moving match around each fixed point
    nb_workers : int, optional
        number of processes to perform neighborhood search in parallel. Defaults to cpu_count()
    batch_size : int, optional
        number of fixed points to consider at a time. Defaults to N // nb_workers
    matching_kwargs : dict, optional
        keyword arguments to pass to `feature_matching` for filtering the candidate matches.
        The `n_jobs` kwarg will be respected by each neighborhood worker, so the default is 1.

    Returns
    -------
    idx_fixed : ndarray
        indices of fixed point matches. Empty if no matches are found
    idx_moving : ndarray
        indices of moving point matches. Empty if no matches are found

    """
    if nb_workers is None:
        nb_workers = multiprocessing.cpu_count()

    if batch_size is None:
        batch_size = int(np.ceil(points_fixed.shape[0]/nb_workers))

    if matching_kwargs is None:
        matching_kwargs = {}

    spatial_nbrs_moving = build_neighbors(points_moving, n_neighbors=2, radius=radius, n_jobs=-1)

    nb_points_fixed = points_fixed.shape[0]
    nb_batches = int(np.ceil(nb_points_fixed / batch_size))

    matches_fixed = []
    matches_moving = []

    with multiprocessing.Pool(nb_workers) as pool:

        for b in range(nb_batches):
            # Index range for the current batch
            batch_start = b * batch_size
            batch_stop = min((b + 1) * batch_size, nb_points_fixed)

            # Get the coordiantes and features for the current batch
            points_fixed_batch = points_fixed[batch_start: batch_stop]
            feat_fixed_batch = feat_fixed[batch_start: batch_stop]

            # Find neighborhoods for current batch (list of arrays for each fixed point in batch)
            idxs_nearby = spatial_nbrs_moving.radius_neighbors(points_fixed_batch,
                                                               return_distance=False)

            # Get the features for the neighboring moving points
            feat_moving_nearby = []
            for idxs in idxs_nearby:
                feat_moving_nearby.append(feat_moving[idxs])  # variable number of neighbors

            # Search each neighborhood in parallel for matches
            result = list(tqdm.tqdm(
                pool.imap(_feature_matching,
                    zip(feat_fixed_batch,
                    feat_moving_nearby,
                    feat_fixed_batch.shape[0]*[matching_kwargs]),
                          chunksize=1000),
                    total=int(batch_stop-batch_start),
                                    desc="batch {}/{}".format(b+1, nb_batches),
                                    leave=False))
            # result is a list of tuples containing matching fixed and moving indices in arrays

            # filter out neighborhoods without matches--may contain a variable number of matches for each batch
            matches_batch = np.asarray([[i, m[0]] for i, (f, m) in enumerate(result) if f.size != 0 and m.size != 0])

            if len(matches_batch) == 0:
                continue

            # matches_batch[:, 0] contains batch indices for fixed matches
            # matches_batch[:, 1] contains batch indices for moving matches (can exceed batch_size)

            # convert batch match indices into global match indices
            batch_idxs = np.arange(batch_start, batch_stop)
            idxs_nearby_matched = idxs_nearby[matches_batch[:, 0]]  # list of arrays (neighborhoods with matches)

            for nbrhd, match in zip(idxs_nearby_matched, matches_batch):
                matches_fixed.append(batch_idxs[match[0]])
                matches_moving.append(nbrhd[match[1]])

    return np.asarray(matches_fixed), np.asarray(matches_moving)


"""
Find neighborhoods in smaller batches using nbrs.radius_neighbors(batch, n_jobs=-1)
This will be multi-core, and avoids OS differences involved with forking and CoW
We can find matches in a batch of neighborhoods by using feature_matching
on each fixed point in the batch
Each fixed point has it's own potential moving matches, independent from each other
Thus, we are mapping feature_matching over a sequence of fixed_points and nbrhoods 
Rather than spinning up the pool on each batch, we should do:

with mp.Pool(n) as p:
    for b in range(batches):
        nbrhoods = nbrs.radius_neighbors(batch, n_jobs=-1)
        idx_f, idx_m = p.starmap(feature_matching, zip(batch, nbrhoods))

"""

# Old code


def augment(pts):
    nb_pts = pts.shape[0]
    return np.hstack([pts, np.ones((nb_pts, 1))])


def augmented_matrix(pts):
    nb_pts = pts.shape[0]
    submatrix = augment(pts)  # [z, y, x, 1]
    A = np.zeros((3*nb_pts, 12))
    A[:nb_pts, :4] = submatrix
    A[nb_pts:2*nb_pts, 4:8] = submatrix
    A[2*nb_pts:, 8:] = submatrix
    return A


def flatten(pts):
    return pts.T.flatten()


def unflatten(pts_vector):
    nb_pts = int(len(pts_vector)/3)
    return np.reshape(pts_vector, (3, nb_pts)).T


def estimate_affine(batch_stationary, batch_moving, mode='ransac', min_samples=4, residual_threshold=None):
    """Estimate an affine transformation from fixed to moving points

    Parameters
    ----------
    batch_stationary : array
        array (N, 3) of matched fixed points
    batch_moving : array
        array (N, 3) of matched moving points
    mode : str
        Either 'ransac' or 'lstsq'
    min_samples : int
        int for the number of samples to use for RANSAC
    residual_threshold : float
        float for the inlier-outlier cutoff residual

    Returns
    -------
    model : linear_model or array
        ransac or affine matrix

    """
    if batch_stationary.shape[0] != batch_moving.shape[0]:
        raise ValueError('Batches need equal number of points')
    b = flatten(batch_moving)
    A = augmented_matrix(batch_stationary)
    if mode == 'ransac':
        ransac = linear_model.RANSACRegressor(min_samples=min_samples,
                                              loss='absolute_loss',
                                              residual_threshold=residual_threshold).fit(A, b)
        inlier_mask = unflatten(ransac.inlier_mask_)
        inlier_idx = np.where(inlier_mask.any(axis=-1))[0]
        return ransac, inlier_idx
    elif mode == 'lstsq':
        x, resid, _, _ = np.linalg.lstsq(A, b, rcond=None)
        t_hat = np.eye(4)
        t_hat[:3] = np.reshape(x, (3, 4))
        return t_hat
    else:
        raise ValueError('Only supported modes are ransac and lstsq')


def register_pts(pts, linear_model):
    # Solve with lstsq
    # t_hat = estimate_affine(candidates_stationary, candidates_moving, mode='lstsq')
    # pts_moving_aug = augment(pts_moving)
    # pts_registered = transform_pts(pts_moving_aug, t_hat)[:,:3]
    # residuals = np.linalg.norm(candidates_stationary-pts_registered[moving_idx], axis=-1)
    # print(f'Average residual: {residuals.mean():.2e}')
    # plot_pts(pts_stationary, pts_registered)
    return unflatten(linear_model.predict(augmented_matrix(pts)))


def average_residual(pts1, pts2):
    return np.linalg.norm(pts1-pts2, axis=-1).mean()


def generate_3d_image(size, center, sigma):
    I = np.zeros((size, size, size))
    center_idx = tuple(int(size/2) for _ in range(3))
    idx = tuple(c+t for c,t in zip(center_idx, center))
    I[idx] = 1
    I = filters.gaussian(I, sigma=sigma)
    return I/I.max()


def coord_mapping(fixed_coords, c, matches_stationary, smoothing):
    """Apply 3D thin-plate spline mapping to fixed_coords.

    The RBF keypoints are given in matches_stationary. Smoothing is a
    surface energy regularization parameter.

    Parameters
    ----------
    fixed_coords : ndarray
        (N, 3) array of points to warp
    c : ndarray
        (M, 3) array of spline coefficients for z, y, and x axes
    matches_stationary : ndarray
        (M, 3) array of keypoint coordinates
    smoothing : float
        surface energy regularization parameter


    Returns
    -------
    moving_coords : ndarray
        (N, 3) array of warped points

    """
    nb_pts = fixed_coords.shape[0]
    dist = scipy.spatial.distance.cdist(fixed_coords, matches_stationary)
    nb_matches = matches_stationary.shape[0]
    T = np.zeros((nb_pts, 4+nb_matches))
    T[:, :4] = np.hstack([np.ones((nb_pts,1)), fixed_coords])
    T[:, 4:] = dist + nb_matches*smoothing
    moving_coords = T.dot(c)
    return moving_coords


def get_nb_chunks(img):
    return tuple(int(np.ceil(img_dim/chunk_dim)) for img_dim, chunk_dim in zip(img.shape, img.chunks))


def get_chunk_index(chunk_shape, chunk_idx):
    return np.array([int(i*dim) for i, dim in zip(chunk_idx, chunk_shape)])


# Functions for point-cloud density registration


def calculate_image_dimensions(shape, voxel_dims):
    return np.asarray(shape) * np.asarray(voxel_dims)


def index_to_um_tup(pts, voxel_dims):
    return pts * np.asarray(voxel_dims)


def pcloud_hist(pts, dimensions, bin_size):
    """Compute the histogram of a 3D point cloud

    Parameters
    ----------
    pts : array
        2D array (N, D) of points
    dimensions : tuple or array
        overall dimensions of the histogram (image bounds)
    bin_size : float
        the physical dimension of each bin

    """
    # Calculate histogram bins and bounds
    bins = np.ceil(dimensions / bin_size)
    density_dim = bins * bin_size
    density_excess = density_dim - dimensions
    density_range = np.array((-density_excess/2, dimensions + density_excess / 2))

    # Calculate histograms
    density, edges = np.histogramdd(pts, bins=bins, range=density_range.T)

    return density


def ncc(fixed, registered):
    """Calculate the normalized cross-correlation between two images

    Parameters
    ----------
    fixed : array
        array of first image to be compared
    registered: array
        array of second image to be compared

    """
    idx = np.where(registered)
    a = fixed[idx]
    b = registered[idx]
    return np.sum((a-a.mean())*(b-b.mean())/((a.size-1)*a.std()*b.std()))


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


def pcloud_density_registration(fixed_img, moving_img, fixed_pts, moving_pts, bin_size_um, sigma_density, niter, voxel_dims):

    fixed_img_shape_um = calculate_image_dimensions(fixed_img.shape, voxel_dims)
    moving_img_shape_um = calculate_image_dimensions(moving_img.shape, voxel_dims)

    # Convert to um wrt top-upper-left (TUP)
    fixed_pts_um = index_to_um_tup(fixed_pts, voxel_dims)
    moving_pts_um = index_to_um_tup(moving_pts, voxel_dims)

    # Make the moving histogram (coarse registration target)
    moving_centroid_um = moving_pts_um.mean(axis=0)
    moving_density = pcloud_hist(moving_pts_um, fixed_img_shape_um, bin_size_um)
    moving_density_smooth = filters.gaussian(moving_density, sigma=sigma_density)

    def registration_objective(theta, fixed_pts_um):
        r = pcloud.rotation_matrix(theta)
        fixed_coords_um_zeroed = fixed_pts_um - fixed_pts_um.mean(axis=0)
        rotated_coords_um_zeroed = pcloud.rigid_transformation(np.zeros(3), r, fixed_coords_um_zeroed)
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

    def transform_pts(theta, pts_um):
        r = pcloud.rotation_matrix(theta)
        fixed_coords_um_zeroed = pts_um - pts_um.mean(axis=0)
        rotated_coords_um_zeroed = rigid_transformation(np.zeros(3), r, fixed_coords_um_zeroed)
        transformed_coords_um = rotated_coords_um_zeroed + moving_centroid_um
        return transformed_coords_um

    transformed_pts_um = transform_pts(thetas, fixed_pts_um)

    return transformed_pts_um


def main():
    # Point cloud synthesis
    nb_pts = 100
    mu = np.array([0, 0, 0])
    sigma = np.array([5, 3, 1])
    # Ground truth transformation
    zdeg = 132
    ydeg = -45
    xdeg = 62
    translation = np.array([200, 150, 100])
    img_size = 32
    # Random deformation, missing points
    # noise_level = 1e-6
    # missing_frac = 0.0
    # Input images
    # fixed_img_file = '../data/spim_registration/spim_fixed.tif'
    # moving_img_file = '../data/spim_registration/spim_moving.tif'
    fixed_zarr_file = '../data/spim_registration/spim_fixed.zarr'
    moving_zarr_file = '../data/spim_registration/spim_moving.zarr'
    # batch_size = 1_000_000
    # Load pts
    pts_fixed_file = 'centroids_fixed.npy'
    pts_moving_file = 'centroids_moving.npy'
    nb_sample = 1000
    nb_workers = 12
    # registration parameters
    signif_T = 0.3
    min_samples = 20
    resid_T = 0.1
    smoothing = 1000

    # # Generate point cloud of stationary image
    # pts_stationary = synthesize_pts(nb_pts, mu, sigma)

    # # Generate point cloud of moving image
    # T = get_transformation(zdeg, ydeg, xdeg)
    # pts_moving = transform_pts(pts_stationary, T, noise_level)+translation
    # pts_moving = remove_random_pts(pts_moving, missing_frac)

    # Load pts
    # pts_stationary = np.load(pts_fixed_file)
    # pts_moving = np.load(pts_moving_file)

    # np.random.shuffle(pts_moving)

    # plot_pts(pts_stationary, pts_moving)

    # subset = pts_moving[:10_000]
    # coords = np.vstack((2.5*subset[:,0], 1.25*subset[:,1], 1.25*subset[:,2])).T
    # print(coords.shape)
    # print(pts_moving.shape)
    # plot_pts(coords)

    # # Extract geometric features
    # feat_stationary = geometric_features(pts_stationary, nb_workers)
    # feat_moving = geometric_features(pts_moving, nb_workers)

    # stationary_idx, moving_idx = match_pts(feat_stationary, feat_moving, signif_T, display=False)
    # matches_stationary = pts_stationary[stationary_idx]
    # matches_moving = pts_moving[moving_idx]

    # # plot_pts(pts_stationary, pts_moving, matches_stationary, matches_moving)

    # # Solve for Affine transform with RANSAC
    # ransac, inlier_idx = estimate_affine(matches_stationary, matches_moving, min_samples=min_samples, residual_threshold=resid_T)
    # pts_registered = register_pts(pts_stationary, ransac)
    # ave_residual = average_residual(matches_moving, pts_registered[stationary_idx])
    # print(f'Average affine residual: {ave_residual:.2e}')

    # # plot_pts(pts_moving, pts_registered)

    # matches_stationary = matches_stationary[inlier_idx]
    # matches_moving = matches_moving[inlier_idx]

    # # # Fine adjustment using thin-plate spline
    # # # Use affine transformed pts as moving pts again

    # if len(inlier_idx) > nb_sample:
    # 	sample_idx = np.random.choice(inlier_idx, size=nb_sample, replace=False)
    # 	matches_stationary = matches_stationary[sample_idx]
    # 	matches_moving = matches_moving[sample_idx]

    # # Estimate thin-plate spline coefficients using RANSAC inliers
    # xp = np.hstack([matches_moving[:,0].flatten(), np.zeros(4)])
    # yp = np.hstack([matches_moving[:,1].flatten(), np.zeros(4)])
    # zp = np.hstack([matches_moving[:,2].flatten(), np.zeros(4)])

    # nb_matches = matches_stationary.shape[0]
    # submat1 = np.hstack([np.ones((nb_matches,1)), matches_stationary])

    # dist = scipy.spatial.distance.cdist(matches_stationary, matches_stationary)
    # T = np.zeros((nb_matches+4, nb_matches+4))
    # T[:-4,:4] = submat1
    # T[-4:,4:] = submat1.T
    # T[:-4,4:] = dist + nb_matches*smoothing*np.eye(nb_matches)

    # cx, _, _, _ = np.linalg.lstsq(T, xp, rcond=None)
    # cy, _, _, _ = np.linalg.lstsq(T, yp, rcond=None)
    # cz, _, _, _ = np.linalg.lstsq(T, zp, rcond=None)

    # c = np.vstack([cx, cy, cz]).T
    # pts_transformed = T[:-4].dot(c)
    # ave_residual = average_residual(matches_moving, pts_transformed)
    # print(f'Average spline residual: {ave_residual:.2e}')

    # plot_pts(matches_moving, pts_transformed)

    # img_moving = zarr.open('spim_moving.zarr', mode='r')
    # img_fixed = zarr.open('spim_fixed.zarr', mode='r')
    # img_registered = zarr.open('spim_registered.zarr',
    # 	mode='w',
    # 	shape=img_fixed.shape,
    # 	chunks=img_fixed.chunks,
    # 	dtype='float32')

    # nb_chunks = get_nb_chunks(img_fixed)

    # start_indices = []
    # for z in range(nb_chunks[0]):
    # 	for y in range(nb_chunks[1]):
    # 		for x in range(nb_chunks[2]):
    # 			start_indices.append(get_chunk_index(img_fixed.chunks, (z,y,x)))
    # start_coords = np.array(start_indices)

    # local_indices = np.indices(img_fixed.chunks)
    # z_idx = local_indices[0].flatten()
    # y_idx = local_indices[1].flatten()
    # x_idx = local_indices[2].flatten()
    # local_coords = np.array([z_idx, y_idx, x_idx]).T

    # args_list = []
    # for i, start in enumerate(start_coords):
    # 	args = (img_fixed, img_moving, img_registered,
    # 		start, local_coords, batch_size, c, matches_stationary, smoothing)
    # 	args_list.append(args)

    # with multiprocessing.Pool(processes=nb_workers) as pool:
    # 	pool.starmap(do_work, args_list)


if __name__ == '__main__':
    main()
