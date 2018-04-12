import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from scipy.ndimage import affine_transform
from scipy.ndimage import map_coordinates, center_of_mass
from scipy.optimize import minimize
from skimage import filters

# Set consistent numpy random state
np.random.seed(123)

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


def plot_densities(fixed, moving=None, registered=None, z=0, mip=False):
    if mip:
        fixed_img = fixed.max(axis=0)
    else:
        fixed_img = fixed[z]
    plt.imshow(fixed_img)
    plt.show()
    if moving is not None:
        if mip:
            moving_img = moving.max(axis=0)
        else:
            moving_img = moving[z]
        plt.imshow(moving_img)
        plt.show()
    if registered is not None:
        if mip:
            registered_img = registered.max(axis=0)
        else:
            registered_img = registered[z]
        plt.imshow(registered_img)
        plt.show()


def rotation_matrix(thetas):
    rz = np.eye(3)
    rz[1,1] = np.cos(thetas[0])
    rz[2,2] = np.cos(thetas[0])
    rz[1,2] = -np.sin(thetas[0])
    rz[2,1] = np.sin(thetas[0])
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


def augmented_matrix(t, r):
    a = np.eye(len(t)+1)
    a[:-1, :-1] = r
    a[:-1, -1] = t
    return a


def rigid_transformation(t, r, pts):
    return r.dot(pts.T).T + t


def indices_to_um(pts, voxel_dimensions):
    return np.array([d*pts[:, i] for d, i in zip(voxel_dimensions, range(len(voxel_dimensions)))]).T


def um_to_indices(pts_um, voxel_dimensions):
    return np.array([pts_um[:, i]/d for d, i in zip(voxel_dimensions, range(len(voxel_dimensions)))]).T


def shape_to_coordinates(shape):
    indices = np.indices(shape)
    z_idx = indices[0].flatten()
    y_idx = indices[1].flatten()
    x_idx = indices[2].flatten()
    return np.array([z_idx, y_idx, x_idx]).T


def transform_density(moving_density, output_shape, t, r):
    coords = shape_to_coordinates(output_shape)
    moving_coords = rigid_transformation(t, r, coords)
    interp_values = map_coordinates(moving_density, moving_coords.T)
    transformed_density = np.reshape(interp_values, output_shape)
    return transformed_density


def ncc(fixed, registered):
    idx = np.where(registered>0)
    a = fixed[idx]
    b =registered[idx]
    # a = fixed
    # b = registered
    return np.sum((a-a.mean())*(b-b.mean())/((a.size-1)*a.std()*b.std()))


def registration_objective(x, moving_density, fixed_density):
    t = x[:3]
    thetas = x[3:]
    r = rotation_matrix(thetas)
    transformed_density = transform_density(moving_density, fixed_density.shape, t, r)
    return -ncc(fixed_density, transformed_density)


def main():
    # Original image properties
    voxel_dimensions = (2.0, 1.6, 1.6)
    fixed_img_shape = (256, 512, 512)
    moving_img_shape = (256, 512, 512)

    # Synthetic point clouds
    nb_pts = 100
    cov = np.diag(np.array([1,2,1]))
    t = np.array([-1000, 0, 0])
    r = rotation_matrix(np.array([np.pi/2, 0, 0])) # 90deg in plane rotation
    noise_um = 0.0
    missing_frac = 0.00
    # Density estimation
    bin_size_um = 100

    ###################################

    # Create synthetic point clouds
    nb_missing = np.round(nb_pts * missing_frac)
    missing_idx = np.random.choice(range(nb_pts), int(nb_missing), replace=False)
    moving_noise_um = noise_um * np.random.rand(nb_pts, 3) # um
    print('Taking out {} missing points.'.format(nb_missing))

    fixed_pts = np.floor(np.array(fixed_img_shape) * np.random.multivariate_normal([0,0,0], cov, size=nb_pts)) # np.random.rand(nb_pts, 3))
    fixed_pts_um = np.array(voxel_dimensions) * fixed_pts  # um
    moving_pts_um = rigid_transformation(t, r, fixed_pts_um) + moving_noise_um # um
    moving_pts_um = np.delete(moving_pts_um, missing_idx,axis=0)

    plot_pts(fixed_pts_um, moving_pts_um)

    # Find translation
    fixed_centroid_um = fixed_pts_um.mean(axis=0)
    moving_centroid_um = moving_pts_um.mean(axis=0)
    fixed_pts_um_zeroed = fixed_pts_um - fixed_centroid_um
    moving_pts_um_zeroed = moving_pts_um - moving_centroid_um
    t_hat = moving_centroid_um - fixed_centroid_um
    t_err = np.linalg.norm(t-t_hat)
    print('translation: {}'.format(t_hat))
    print('Error in translation vector: {} um'.format(t_err))

    plot_pts(fixed_pts_um_zeroed, moving_pts_um_zeroed)

    # Figure out the nb_bins
    fixed_pts_um_range = np.array((fixed_pts_um_zeroed.min(axis=0), fixed_pts_um_zeroed.max(axis=0)))
    moving_pts_um_range = np.array((moving_pts_um_zeroed.min(axis=0), moving_pts_um_zeroed.max(axis=0)))
    fixed_pts_um_extent = np.diff(fixed_pts_um_range, axis=0)[0]
    moving_pts_um_extent = np.diff(moving_pts_um_range, axis=0)[0]
    fixed_bins = np.ceil(fixed_pts_um_extent / bin_size_um)
    moving_bins = np.ceil(moving_pts_um_extent / bin_size_um)

    # Figure out the bin ranges
    fixed_bins_extent = bin_size_um*fixed_bins
    moving_bins_extent = bin_size_um*moving_bins
    fixed_excess = fixed_bins_extent - fixed_pts_um_extent
    moving_excess = moving_bins_extent - moving_pts_um_extent
    fixed_bin_range = fixed_pts_um_range + np.array((-fixed_excess/2, fixed_excess/2))
    moving_bin_range = moving_pts_um_range + np.array((-moving_excess/2, moving_excess/2))

    # Calculate the point density maps
    fixed_density, fixed_edges = np.histogramdd(fixed_pts_um_zeroed, bins=fixed_bins, range=fixed_bin_range.T)
    moving_density, moving_edges = np.histogramdd(moving_pts_um_zeroed, bins=moving_bins, range=moving_bin_range.T)

    fixed_density = filters.gaussian(fixed_density, sigma=2)
    moving_density = filters.gaussian(moving_density, sigma=2)

    plot_densities(fixed_density, moving_density, mip=True)


    # 3D intensity-based registration
    fixed_density_com = np.array(fixed_density.shape)/2
    moving_density_com = np.array(moving_density.shape)/2
    t0 = moving_density_com - fixed_density_com
    x0 = np.concatenate((t0, np.zeros(3)))
    res = minimize(fun=registration_objective,
                   x0=x0,
                   args=(fixed_density, moving_density),
                   method='Nelder-Mead',
                   bounds=None,
                   tol=1e-6,
                   options={'disp': True})
    print('Optimization status code {} and exit flag {}'.format(res.status, res.success))
    print('Final correlation coefficient: {}'.format(-res.fun))

    t_star = res.x[:3]
    thetas_star = res.x[3:]
    print(t_star, thetas_star)
    r_star = rotation_matrix(thetas_star)
    registered_density = transform_density(moving_density, fixed_density.shape, t_star, r_star)

    plot_densities(fixed_density, registered_density, mip=True)







if __name__ == '__main__':
    main()