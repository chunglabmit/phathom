import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


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


def plot_densities(z, fixed, moving=None, registered=None, mip=False):
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


def img_dimensions(shape, voxel_dimensions):
    return np.array([s*d for s, d in zip(shape, voxel_dimensions)])


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


def rigid_transformation(t, r, pts):
    return r.dot(pts.T).T + t


def indices_to_um(pts, voxel_dimensions):
    return np.array([d*pts[:, i] for d, i in zip(voxel_dimensions, range(len(voxel_dimensions)))]).T


def um_to_indices(pts_um, voxel_dimensions):
    return np.array([pts_um[:, i]/d for d, i in zip(voxel_dimensions, range(len(voxel_dimensions)))]).T


def main():
    # Original image properties
    voxel_dimensions = (2.0, 1.6, 1.6)
    img_shape = (512, 512, 512)
    # Synthetic point clouds
    nb_pts = 1000
    translation = np.array([-500, 0, 0])
    # Density estimation
    bin_size_um = 100

    # Create synthetic point clouds
    fixed_img_dimensions = img_dimensions(img_shape, voxel_dimensions)
    moving_img_dimensions = fixed_img_dimensions

    fixed_pts = np.random.randn(nb_pts, 3)
    fixed_pts_um = fixed_img_dimensions*fixed_pts

    moving_pts_um = fixed_pts_um[:, (0, 2, 1)]
    moving_pts_um[:, 2] *= -1
    moving_pts_um += translation

    plot_pts(fixed_pts_um, moving_pts_um)

    # Estimate translation from centroids
    fixed_centroid_um = fixed_pts_um.mean(axis=0)
    moving_centroid_um = moving_pts_um.mean(axis=0)
    t = moving_centroid_um - fixed_centroid_um
    print('Estimated translation vector: {}'.format(t))

    # # Calculate histograms
    # fixed_bins = np.ceil(fixed_img_dimensions / bin_size_um)
    # moving_bins = np.ceil(moving_img_dimensions / bin_size_um)
    #
    # fixed_range = tuple((0, d) for d in fixed_img_dimensions)
    # moving_range = tuple((0, d) for d in moving_img_dimensions)
    #
    # fixed_density, fixed_edges = np.histogramdd(fixed_pts_um, bins=fixed_bins, range=fixed_range)
    # moving_density, moving_edges = np.histogramdd(moving_pts_um, bins=moving_bins, range=moving_range)
    #
    # plot_densities(3, fixed_density, moving_density, mip=True)



if __name__ == '__main__':
    main()