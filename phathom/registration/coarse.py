"""The coarse module contains functions related to coarse intensity-based registration
"""

import numpy as np
from skimage.measure import block_reduce
from skimage.morphology import convex_hull_image, remove_small_objects
from scipy.optimize import basinhopping
from scipy.ndimage import map_coordinates
from scipy.ndimage.morphology import distance_transform_edt
from scipy.ndimage.measurements import center_of_mass
import tqdm
from phathom.registration import pcloud
from phathom import io
import neuroglancer
from nuggt.utils import ngutils


def ncc(img1, img2, nonzero=False):
    """Calculate the normalized cross-correlation between two images

    Parameters
    ----------
    img1 : ndarray
        first image to be compared
    img2: ndarray
        second image to be compared
    nonzero : bool
        Flag to only consider pixels where `img2 != 0`. Default is `False`.

    Returns
    -------
    ncc : float
        normalized cross-correlation between `img1` and `img2`

    """
    if nonzero:
        idx = np.where(img2)
        a = img1[idx]
        b = img2[idx]
    else:
        a = img1
        b = img2
    return np.sum((a-a.mean())*(b-b.mean())/((a.size-1)*a.std()*b.std()))


def mse(img1, img2):
    """Calculate the mean squared error between two images

    Parameters
    ----------
    img1 : ndarray
        first image to be compared
    img2: ndarray
        second image to be compared

    Returns
    -------
    mse : float
        mean squared error between `img1` and `img2`

    """
    return np.mean((img1-img2)**2)


def downsample_mean(img, factors):
    """Downsample an image by integer factors by averaging blocks

    Parameters
    ----------
    img : ndarray
        image to be downsampled
    factors: tuple
        integers to downsample each axis

    Returns
    -------
    img_downsampled : ndarray
        downsampled image

    """
    return block_reduce(img, factors, np.mean, 0)


def rigid_transformation(t, r, pts, center=None):
    """Apply rotation and translation (rigid transformtion) to a set of points

    Parameters
    ----------
    t : ndarray
        (D,) array representing the translation vector
    r : ndarray
        (D, D) array representing the rotation matrix
    pts : ndarray
        (N, D) array of N D-dimensional points to transform
    center : ndarray, optional
        (D,) array representing the center of rotation. Defaults to the origin

    Returns
    -------
    transformed_pts : ndarray
        (N, D) array of transformed points

    """
    if center is None:
        return r.dot(pts.T).T + t
    else:
        return r.dot((pts-center).T).T + center + t


def rigid_warp(img, t, thetas, center, output_shape):
    """Warp an image using a rigid transformation

    Parameters
    ----------
    img : ndarray
        D-dimensional image to warp
    t : ndarray
        (D,) translation vector in pixels
    thetas : ndarray
        (D,) rotation angle vector in radians
    center : ndarray
        (D,) center of rotation vector in pixels
    output_shape : tuple
        tuple of length D of the output image shape

    Returns
    -------
    warped_img : ndarray
        D-dimensional warped image

    """
    r = pcloud.rotation_matrix(thetas)
    idx = np.indices(output_shape)
    pts = np.reshape(idx, (idx.shape[0], idx.size//idx.shape[0])).T
    warped_pts = rigid_transformation(t, r, pts, center)
    interp_values = map_coordinates(img, warped_pts.T)
    warped_img = np.reshape(interp_values, output_shape)
    return warped_img


def center_mass(img):
    """Calculates the center of mass of an image

    Parameters
    ----------
    img : ndarray
        image of the mass density

    Returns
    -------
    center : ndarray
        center of mass of the input image

    """
    return np.asarray(center_of_mass(img))


def threshold_img(img, t):
    """Binarize input image by intensity thresholding

    Parameters
    ----------
    img : ndarray
        image to threshold
    t : float
        intensity threshold value

    Returns
    -------
    mask : ndarray
        thresholded img (binary)

    """
    return (img > t)


def convex_hull(mask):
    """Calculate the convex hull image of a potentially non-convex input mask

    Parameters
    ----------
    mask : ndarray
        binary mask

    Returns
    -------
    hull : ndarray
        bianry mask of convex hull

    """
    hull = np.zeros_like(mask)
    for i, f in enumerate(tqdm.tqdm(mask)):
        if not np.all(f == 0):
            hull[i] = convex_hull_image(f)
    return hull


def distance_transform(mask):
    """Calculate the distance transform of a binary mask

    Parameters
    ----------
    mask : ndarray
        binary mask

    Returns
    -------
    edt : ndarray
        euclidian distance transform of `mask`

    """
    return distance_transform_edt(mask)


def _registration_objective(x, source, target, center):
    """Objective function for intensity-based registration

    Parameters
    ----------
    x : ndarray
        translation and rotation parameter vector. t = x[:D] and theta = x[D:]
    source : ndarray
        image to be warped
    target : ndarray
        image to try to match
    center : ndarray
        (D,) center of rotation for the source image

    Returns
    -------
    mse : float
       mean squared error between the warped image and target image

    """
    transformed_img = rigid_warp(source,
                                 t=x[:3],
                                 thetas=x[3:],
                                 center=center,
                                 output_shape=target.shape)
    return mse(target, transformed_img)


def optimize(source, target, center=None, t0=None, theta0=None, niter=10):
    """Estimate rigid transformation parameters that align source to target using
    basinghopping (Metropolis-Hastings) with L-BFGS-B minimizer

    Parameters
    ----------
    source : ndarray
        image to be aligned to `target`
    target : ndarray
        image for `source` to match
    center : ndarray
        (D,) center of rotation for the source image
    t0 : ndarray, optional
        (D,) initial guess for the translation vector. Default is center difference.
    theta0 : ndarray, optional
        (D,) initial guess for the rotation angle vector. Default is zeros.
    niter : int, optional
        number of basinhopping iterations to perform. Default is 10.

    Returns
    -------
    t_star : ndarray
        resulting translation vector
    theta_star : ndarray
        resulting rotation angle vector
    center : ndarray
        center of rotation

    """
    if center is None:
        center = center_mass(source)
    if t0 is None:
        target_center = center_mass(target)
        t0 = target_center - center
    if theta0 is None:
        theta0 = np.zeros_like(center)
    bounds = [(-s, s) for s in target.shape] + [(-np.pi, np.pi) for _ in range(3)]
    res = basinhopping(_registration_objective,
                       x0=np.concatenate((t0, theta0)),
                       niter=niter,
                       T=1.0,
                       stepsize=1.0,
                       interval=5,
                       minimizer_kwargs={
                           'method': 'L-BFGS-B',
                           'args': (source,
                                    target,
                                    center),
                           'bounds': bounds,
                           'tol': 0.001,
                           'options': {'disp': False}
                       },
                       disp=True)
    t_star = res.x[:3]
    theta_star = res.x[3:]
    return t_star, theta_star, center


def _scale_rigid_params(t, center, factors):
    """Convert the translation and center of rotation portions of the rigid transformation
    to another image scale (the original image size, for example)

    Parameters
    ----------
    t : ndarray
        translation vector
    center : ndarray
        center of rotation vector
    factors : tuple
        integer scaling factors for each axis

    Returns
    -------
    t_new : ndarray
        scaled translation vector
    center_new : ndarray
        scaled center of rotation vector

    """
    f = np.asarray(factors)
    return t*f, center*f


def coarse_registration(source, target, threshold, opt_kwargs):
    source_mask = threshold_img(source, threshold)
    target_mask = threshold_img(target, threshold)

    source_no_small = remove_small_objects(source_mask, min_size=10)
    target_no_small = remove_small_objects(target_mask, min_size=10)

    source_hull = convex_hull(source_no_small)
    target_hull = convex_hull(target_no_small)

    source_edt = distance_transform(source_hull)
    target_edt = distance_transform(target_hull)

    return optimize(source_edt, target_edt, **opt_kwargs)


def main():
    fixed_zarr_path = '/media/jswaney/Drive/Justin/coregistration/whole_brain/fixed_down.zarr'
    moving_zarr_path = '/media/jswaney/Drive/Justin/coregistration/whole_brain/moving_down.zarr'
    voxel_dims = (20.0, 16.0, 16.0)
    normalization = 2000

    # fixed_down_path = '/media/jswaney/Drive/Justin/coregistration/whole_brain/fixed_down.zarr'
    # moving_down_path = '/media/jswaney/Drive/Justin/coregistration/whole_brain/moving_down.zarr'
    # factors = (10, 10, 10)
    # fixed_img = io.zarr.open(fixed_zarr_path)
    # io.zarr.downsample_zarr(fixed_img, factors, fixed_down_path, nb_workers=12)
    # moving_img = io.zarr.open(moving_zarr_path)
    # io.zarr.downsample_zarr(moving_img, factors, moving_down_path, nb_workers=12)

    fixed_img = np.array(io.zarr.open(fixed_zarr_path))
    moving_img = np.array(io.zarr.open(moving_zarr_path))

    viewer = neuroglancer.Viewer()
    print(viewer)

    with viewer.txn() as txn:
        fixed_img_source = neuroglancer.LocalVolume(fixed_img.astype(np.float32))
        fixed_img_shader = ngutils.green_shader % (1 / normalization)
        txn.layers['fixed'] = neuroglancer.ImageLayer(source=fixed_img_source,
                                                      shader=fixed_img_shader)

        moving_img_source = neuroglancer.LocalVolume(moving_img.astype(np.float32))
        moving_img_shader = ngutils.red_shader % (1 / normalization)
        txn.layers['moving'] = neuroglancer.ImageLayer(source=moving_img_source,
                                                       shader=moving_img_shader)

    input('Press any key to continue...')


if __name__ == '__main__':
    main()