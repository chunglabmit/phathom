import numpy as np
import scipy
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors, kneighbors_graph
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
            n = len(points[0])
    except TypeError:
        raise ValueError('points and thetas must both be iterables')
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
    nbrs = NearestNeighbors(n_neighbors=4, algorithm='kd_tree', n_jobs=-1).fit(pts)
    distances, indices = nbrs.kneighbors(pts)
    # indices is len(pts) by 3, in order of decreasing distance

    args = []
    for i, (center, row) in enumerate(zip(pts, indices)):
        args.append((center, pts[row[1:]]))

    with multiprocessing.Pool(processes=nb_workers) as pool:
        features = pool.starmap(geometric_hash, args)

    return np.asarray(features)


def calculate_distance(target, candidates):
    dists = cdist(target.reshape(1, -1), candidates)[0]
    return dists


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
    close : tuple or None
        tuple of coordinate arrays for those points that are close or None.

    """
    if dists.ndim != 1:
        raise ValueError('candidates must be one dimensional')
    indicators = list(map(lambda d: d < max_dist, dists))
    close = np.where(indicators)
    if len(close[0]) == 0:
        close = None
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
    prominent : tuple or None
        a tuple of coordinate arrays for the found prominent points

    """
    indicators = list(map(lambda p: p < threshold, prom))
    prominent = np.where(indicators)

    if len(prominent[0]) == 0:
        prominent = None

    return prominent


def global_matching(feat_fixed, feat_moving, max_fdist=None, prom_thresh=None):
    """Perform point matching based on feature distances

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

    Returns
    -------
    idx_fixed : ndarray or None
        indices of fixed point matches. None is returned is no matches are found
    idx_moving : ndarray or None
        indices of moving point matches. None is returned if no matches are found

    """
    # TODO: this kd_tree building and querying should be done in a separate function
    nbrs = NearestNeighbors(n_neighbors=2, algorithm='kd_tree', n_jobs=-1).fit(feat_moving)
    fdists, idxs_moving = nbrs.kneighbors(feat_fixed)

    nb_fixed = len(feat_fixed)
    idx_fixed = np.arange(nb_fixed)

    if max_fdist is not None:
        close = check_distance(fdists[:, 0], max_fdist)
        if close is None:
            return None
        else:
            fdists = fdists[close]
            idxs_moving = idxs_moving[close]
            idx_fixed = idx_fixed[close]

    if prom_thresh is not None:
        prom = prominence(fdists[:, 0], fdists[:, 1])
        prominent = check_prominence(prom, prom_thresh)
        if prominent is None:
            return None
        else:
            fdists = fdists[prominent]
            idxs_moving = idxs_moving[prominent]
            idx_fixed = idx_fixed[prominent]

    idx_moving = idxs_moving[:, 0]

    return idx_fixed, idx_moving


def find_similar(feat_stationary, feat_moving):
    feature_nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree', n_jobs=-1).fit(feat_moving)
    return feature_nbrs.kneighbors(feat_stationary)


def neighborhood_init(external_spatial_kdtree, external_feat_moving):
    """Initialize workers that perform neighborhood matching

    Parameters
    ----------
    external_spatial_kdtree : NearestNeighbor
        a NearestNeighbor kdtree fit on moving pts
    external_feat_moving : array
        (N, 6) array of the moving point features

    """
    global spatial_kdtree
    global feat_moving
    spatial_kdtree = external_spatial_kdtree
    feat_moving = external_feat_moving


def neighborhood_matching(args):
    """Search for a matching moving point within a radius neighborhood

    Parameters
    ----------
    args : tuple
        A tuple containing a fixed point and its features

    Returns
    -------
    result : None or array
        position of the found moving point match. None if no match is found.

    """
    pt_stationary, pt_stationary_feat, prominence_thresh, max_feat_dist = args

    # spatial_kdtree is copied to the global scope of each subprocess
    nbrhoods = spatial_kdtree.radius_neighbors(pt_stationary.reshape(1, -1), return_distance=False)
    nbrhood = nbrhoods[0]  # nbrhoods is a list of arrays (jagged)

    pts_moving_feat = feat_moving[nbrhood]

    if len(nbrhood) > 1:
        feat_distances = cdist(pt_stationary_feat.reshape(1, -1), pts_moving_feat)[0]
        nearest_two = np.argsort(feat_distances)[:2]
        prominence = feat_distances[nearest_two[0]] / (1e-9 + feat_distances[nearest_two[1]])
        if prominence < prominence_thresh and feat_distances[nearest_two[0]] < max_feat_dist:
            return nbrhood[nearest_two[0]]


def match_pts(pts_stationary, pts_moving, feat_stationary, feat_moving, max_feat_dist, prominence_thresh, max_distance, nb_workers):
    """Find matching moving points with a radius neighborhood of fixed points

    Parameters
    ----------
    pts_stationary : array
        array (N, 3) of fixed points
    pts_moving : array
        array (M, 3) of moving points
    feat_stationary : array
        array (N, 6) of fixed point features
    feat_moving : array
        array (M, 6) of moving point features
    max_feat_dist : float
        float of the maximum allowed feature-space distance
    prominence_thresh : float
        float of the maximum allowed prominence factor
    max_distance : float
        float of the neighborhood search radius
    nb_workers : int
        number of processes to search for matches in parallel

    Returns
    -------
    stationary_matches : array
        indices of found matches in the stationary image
    moving_matches : array
        indices of found matches in the moving image

    """
    print('building kd-tree')
    spatial_nbrs = NearestNeighbors(radius=max_distance, algorithm='kd_tree', n_jobs=-1).fit(pts_moving)

    # Building the kd-tree is fast, but the radius neighbors cannot be held in memory
    # The radius neighbors should be part of the neighboorhood matching within subprocesses

    print('building arguments list')
    args = []
    for i, pt_stationary in tqdm.tqdm(enumerate(pts_stationary), total=len(pts_stationary)):
        args.append((pt_stationary, feat_stationary[i], prominence_thresh, max_feat_dist))

    print('finding neighborhood matches')
    with multiprocessing.Pool(processes=nb_workers, initializer=neighborhood_init, initargs=(spatial_nbrs, feat_moving)) as pool:
        results = list(tqdm.tqdm(pool.imap(neighborhood_matching, args, chunksize=10), total=len(pts_stationary)))

    stationary_matches = [i for i, j in enumerate(results) if j is not None]
    moving_matches = [j for j in results if j is not None]
    print('Found {} matches.'.format(len(moving_matches)))

    return stationary_matches, moving_matches


def match_pts_global(pts_stationary, pts_moving, feat_stationary, feat_moving, max_feat_dist, prominence_thresh, max_distance, nb_workers):
    """Find matching moving points globally (without neighborhood search)

    Parameters
    ----------
    pts_stationary : array
        array (N, 3) of fixed points
    pts_moving : array
        array (M, 3) of moving points
    feat_stationary : array
        array (N, 6) of fixed point features
    feat_moving : array
        array (M, 6) of moving point features
    max_feat_dist : float
        float of the maximum allowed feature-space distance
    prominence_thresh : float
        float of the maximum allowed prominence factor
    max_distance : float
        float of the neighborhood search radius
    nb_workers : int
        number of processes to search for matches in parallel

    Returns
    -------
    stationary_matches : array
        indices of found matches in the stationary image
    moving_matches : array
        indices of found matches in the moving image

    """
    moving_nbrs = NearestNeighbors(n_neighbors=2, algorithm='kd_tree', n_jobs=-1).fit(feat_moving)
    distances, indices = moving_nbrs.kneighbors(feat_stationary)
    stationary_matches = []
    moving_matches = []
    for i, (idxs, dists) in enumerate(zip(indices, distances)):
        prominence = dists[0] / (1e-9 + dists[1])
        if prominence < prominence_thresh and dists[0] < max_feat_dist:
            stationary_matches.append(i)
            moving_matches.append(idxs[0])
    return stationary_matches, moving_matches


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
    nb_pts = fixed_coords.shape[0]
    # fixed_coords = np.reshape(fixed_coords, (1, *fixed_coords.shape))
    dist = scipy.spatial.distance.cdist(fixed_coords, matches_stationary)
    nb_matches = matches_stationary.shape[0]
    T = np.zeros((nb_pts, 4+nb_matches))
    T[:,:4] = np.hstack([np.ones((nb_pts,1)), fixed_coords])
    T[:,4:] = dist + nb_matches*smoothing
    moving_coords = T.dot(c)
    # moving_coords = np.reshape(moving_coords, (1, *moving_coords.shape))
    return moving_coords


def get_nb_chunks(img):
    return tuple(int(np.ceil(img_dim/chunk_dim)) for img_dim, chunk_dim in zip(img.shape, img.chunks))


def get_chunk_index(chunk_shape, chunk_idx):
    return np.array([int(i*dim) for i, dim in zip(chunk_idx, chunk_shape)])


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
