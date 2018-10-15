from skimage import img_as_float32
from skimage.segmentation import (morphological_geodesic_active_contour,
                                  morphological_chan_vese,
                                  chan_vese,
                                  inverse_gaussian_gradient,
                                  circle_level_set)


def igg(image, alpha=100.0, sigma=5.0):
    return inverse_gaussian_gradient(image, alpha, sigma)


def circle_ls(shape, center=None, radius=None):
    return circle_level_set(shape, center, radius)


def mgac(gimage, niter, **kwargs):
    return morphological_geodesic_active_contour(gimage, niter, **kwargs)


def mcv(image, niter, **kwargs):
    return morphological_chan_vese(image, niter, **kwargs)


def cv(image, **kwargs):
    return chan_vese(image, **kwargs)


def ventricle_2d(image, center, niter):
    # gimage = igg(img_as_float32(image))
    init_level_set = circle_ls(image.shape, center, radius=100)
    return cv(img_as_float32(image), init_level_set=init_level_set)