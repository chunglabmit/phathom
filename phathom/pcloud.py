import numpy as np
import scipy
from scipy.stats import multivariate_normal
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors, kneighbors_graph
from sklearn import linear_model
from skimage import filters
import multiprocessing
import tqdm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# Set consistent numpy random state
np.random.seed(123)


def plot_pts(pts1, pts2=None, candid1=None, candid2=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(pts1[:,2], pts1[:,1], pts1[:,0], c='b', marker='o', label='Stationary', alpha=0.01)

    if pts2 is not None:
        ax.scatter(pts2[:,2], pts2[:,1], pts2[:,0], c='r', marker='o', label='Moving', alpha=0.01)

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


def synthesize_pts(nb_pts, mean, sigma):
    covariance = np.diag(sigma**2)
    mn = multivariate_normal(mean=mean, cov=covariance)
    pts = mn.rvs(size=nb_pts)
    return pts


def remove_random_pts(pts, frac):
    nb_pts = pts.shape[0]
    np.random.shuffle(pts)
    nb_missing = int(nb_pts*frac)
    return pts[:nb_pts-nb_missing]


def get_transformation(zdeg, ydeg, xdeg):
    ztheta = zdeg/180*np.pi
    ytheta = ydeg/180*np.pi
    xtheta = xdeg/180*np.pi
    Tz = np.array( [[np.cos(ztheta), -np.sin(ztheta), 0],
                   [np.sin(ztheta),  np.cos(ztheta), 0],
                   [0, 				0,  		   1]])
    Ty = np.array( [[np.cos(ytheta), 0, np.sin(ytheta)],
                   [0,  			1, 0],
                   [-np.sin(ytheta), 0, np.cos(ytheta)]])
    Tx = np.array( [[1, 0, 				0],
                   [0,  np.cos(xtheta), -np.sin(xtheta)],
                   [0, 	np.sin(xtheta), np.cos(xtheta)]])
    T = Tz.dot(Ty).dot(Tx)
    return T


def transform_pts(pts, transform, noise_level=0):
    if noise_level > 0:
        noise = noise_level*(2*np.random.random(size=pts.T.shape)-1)
    else:
        noise = 0
    transformed_pts = np.dot(transform, pts.T) + noise
    return transformed_pts.T


def geometric_hash(center, vectors):
    # Use QR Decomposition
    a = np.vstack([vectors[2]-center, vectors[0]-center, vectors[1]-center]).T
    q, r = np.linalg.qr(a)
    d = np.diag(np.sign(np.diag(r)))
    q = q.dot(d)
    r = d.dot(r)
    # u, s, v = np.linalg.svd(a)
    feat_vector = r[np.triu_indices(3)]
    return feat_vector  # / np.linalg.norm(feat_vector)


def geometric_features(pts, nb_workers):
    # Performance params available: leaf_size and n_jobs
    nbrs = NearestNeighbors(n_neighbors=4, algorithm='kd_tree', n_jobs=-1).fit(pts)

    distances, indices = nbrs.kneighbors(pts)
    # indices is len(pts) by 3, in order of decreasing distance

    args = []
    for i, (center, row) in enumerate(zip(pts, indices)):
        args.append((center, pts[row[1:]]))

    with multiprocessing.Pool(processes=nb_workers) as pool:
        features = pool.starmap(geometric_hash, args)

    # converting to a numpy array is pretty slow
    features = np.array(features)

    return features


def find_similar(feat_stationary, feat_moving):
    feature_nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree', n_jobs=-1).fit(feat_moving)
    return feature_nbrs.kneighbors(feat_stationary)


def neighborhood_init(external_spatial_kdtree, external_feat_moving):
    global spatial_kdtree
    global feat_moving
    spatial_kdtree = external_spatial_kdtree
    feat_moving = external_feat_moving


def neighborhood_matching(args):
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


def match_pts(pts_stationary, pts_moving, feat_stationary, feat_moving, max_feat_dist, prominence_thresh, max_distance, nb_workers, display=False):
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


def consumer(queue, io_lock, c, matches_stationary, smoothing, img_moving, img_registered):
    while True:
        item = queue.get()
        if item is None:
            break
        else:
            fixed_coord = item

        moving_coord = coord_mapping(fixed_coord, c, matches_stationary, smoothing)

        nearest = np.round(moving_coord)[0]

        z = int(nearest[0])
        y = int(nearest[1])
        x = int(nearest[2])
        if z >= img_registered.shape[0]:
            z = img_registered.shape[0]-1
        elif z < 0:
            z = 0
        if y >= img_registered.shape[1]:
            y = img_registered.shape[1]-1
        elif y < 0:
            y = 0
        if x >= img_registered.shape[2]:
            x = img_registered.shape[2]-1
        elif x < 0:
            x = 0

        interpolated_value = img_moving[z, y, x]

        # interpolated_value = map_coordinates(img_moving, moving_coord.T)

        with io_lock:
            print('Nearest', nearest)
            # img_registered[fixed_coord[0], fixed_coord[1], fixed_coord[2]] = interpolated_value


def get_nb_chunks(img):
    return tuple(int(np.ceil(img_dim/chunk_dim)) for img_dim, chunk_dim in zip(img.shape, img.chunks))


def get_chunk_index(chunk_shape, chunk_idx):
    return np.array([int(i*dim) for i, dim in zip(chunk_idx, chunk_shape)])


def do_work(img_fixed, img_moving, img_registered, start, local_coords, batch_size, c, matches_stationary, smoothing):
    chunk_shape = np.array(img_fixed.chunks)
    img_shape = np.array(img_fixed.shape)

    global_coords = start + local_coords

    stop = start + chunk_shape
    if (stop[0]>img_shape[0]):
        stop[0] = img_shape[0]
        z_outbounds = np.where(global_coords[:,0] >= img_fixed.shape[0])[0]
        global_coords = np.delete(global_coords, z_outbounds, axis=0)
    if (stop[1]>img_shape[1]):
        stop[1] = img_shape[1]
        y_outbounds = np.where(global_coords[:,1] >= img_fixed.shape[1])[0]
        global_coords = np.delete(global_coords, y_outbounds, axis=0)
    if (stop[2]>img_shape[2]):
        stop[2] = img_shape[2]
        x_outbounds = np.where(global_coords[:,2] >= img_fixed.shape[2])[0]
        global_coords = np.delete(global_coords, x_outbounds, axis=0)

    output_shape = tuple(t-s for s,t in zip(start, stop))

    moving_coords = np.zeros(global_coords.shape)
    for i in range(int(np.ceil(global_coords.shape[0]/batch_size))):
        idx1 = i*batch_size
        idx2 = min(global_coords.shape[0], idx1+batch_size)
        moving_coords[idx1:idx2] = coord_mapping(global_coords[idx1:idx2], c, matches_stationary, smoothing)
    z_coords_clip = np.clip(moving_coords[:,0], 0, img_moving.shape[0]-1)
    y_coords_clip = np.clip(moving_coords[:,1], 0, img_moving.shape[1]-1)
    x_coords_clip = np.clip(moving_coords[:,2], 0, img_moving.shape[2]-1)

    invalid_mask = np.where(np.logical_or(np.logical_or(moving_coords[:,0]!=z_coords_clip,
                                                        moving_coords[:,1]!=y_coords_clip),
                                            moving_coords[:,2]!=x_coords_clip))[0]

    moving_coords = np.round(np.vstack([z_coords_clip, y_coords_clip, x_coords_clip]).T).astype('uint32')

    interp_values = img_moving.vindex[moving_coords[:,0].tolist(),
                                    moving_coords[:,1].tolist(),
                                    moving_coords[:,2].tolist()]
    interp_values[invalid_mask] = 0

    interp_chunk = np.reshape(interp_values, output_shape)

    img_registered[start[0]:stop[0], start[1]:stop[1], start[2]:stop[2]] = interp_chunk


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
