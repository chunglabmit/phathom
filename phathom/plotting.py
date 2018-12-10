from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from nuggt.utils import ngutils
import neuroglancer


def plot_pts(pts1, pts2=None, alpha=1, candid1=None, candid2=None):
    """ Plot point clouds with correspondences.

    :param pts1: point cloud 1
    :param pts2: point cloud 2
    :param alpha: point cloud transparency
    :param candid1: matches in point cloud 1
    :param candid2: matches in point cloud 2
    """
    fig = plt.figure(figsize=(4, 4))
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
    """ Plot keypoint density maps.

    :param fixed: array of fixed image density
    :param moving: array of moving image density
    :param registered: array of registered density
    :param z: int indicating which z-slice to plot (if mip == False)
    :param mip: boolean indicating whether or not to project max-intensity on z
    :param clim: pseudocoloring lower and upper intensity bounds
    """
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

# For registration

def plot_correspondence(pts1, pts2, alpha=0.1):
    for i in range(pts1.shape[-1]):
        plt.subplot(1, pts1.shape[-1], i+1)
        plt.scatter(pts1[:, i], pts2[:, i], alpha=alpha)
    plt.show()


def plot_hist(data, bins, xlim=None, ylim=None):
    plt.hist(data, bins)
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    plt.show()


def plot_hist_2d(x, y, bins, _range=None, xlim=None, ylim=None):
    plt.hist2d(x, y, bins=bins, range=_range)
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    plt.show()


def plot_overlay(img1, img2, clim1=None, clim2=None, figsize=None):
    plt.figure(figsize=figsize)
    plt.imshow(img1, cmap='Greens', clim=clim1)
    plt.imshow(img2, cmap='Reds', alpha=0.5, clim=clim2)
    plt.show()


def plot_image(img, viewer, layer, shader, voxel_size=(1, 1, 1)):
    with viewer.txn() as txn:
        source = neuroglancer.LocalVolume(img.astype(np.float32), voxel_size=voxel_size)
        txn.layers[layer] = neuroglancer.ImageLayer(source=source, shader=shader)


def plot_fixed(fixed_img, viewer, normalization=1, voxel_size=(1, 1, 1)):
    fixed_shader = ngutils.red_shader % (1 / normalization)
    plot_image(fixed_img, viewer, 'fixed', fixed_shader, voxel_size)


def plot_moving(moving_img, viewer, normalization=1, voxel_size=(1, 1, 1)):
    moving_shader = ngutils.green_shader % (1 / normalization)
    plot_image(moving_img, viewer, 'moving', moving_shader, voxel_size)


def plot_both(fixed_img, moving_img, viewer, normalization=1, fixed_voxel_size=(1, 1, 1), moving_voxel_size=(1, 1, 1)):
    plot_fixed(fixed_img, viewer, normalization, fixed_voxel_size)
    plot_moving(moving_img, viewer, normalization, moving_voxel_size)


def plot_mip(img, axis=0, clim=None):
    plt.imshow(img.max(axis=axis), clim=clim)
    plt.show()


def zprojection(image, centers=None, zlim=None, clim=None):
    if zlim is None:
        zlim = [0, image.shape[0]-1]
    if len(zlim) != 2:
        raise ValueError("Need start and stop zlim")
    projection = image[zlim[0]:zlim[1]].max(axis=0)
    plt.imshow(projection, clim=clim)
    if centers is not None:
        idx = np.where(np.logical_and(centers[:, 0] >= zlim[0], centers[:, 0] < zlim[1]))[0]
        points = centers[idx]
        plt.plot(points[:, 2], points[:, 1], 'r*')
    plt.show()