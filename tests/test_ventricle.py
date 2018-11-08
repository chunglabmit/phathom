import unittest
import os
import numpy as np
from phathom import io
from phathom import synthetic
from phathom.segmentation import ventricle
from phathom.segmentation.segmentation import find_volumes

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm

from phathom.plotting import plot_mip, zprojection
from skimage.viewer import CollectionViewer
import tifffile

from skimage import img_as_float32
from skimage.filters import gaussian

from skimage.segmentation import slic, watershed, felzenszwalb, quickshift

from skimage.filters import sobel
from skimage.future import graph


seed = 295
test_images = False

working_dir = '/media/jswaney/Drive/Justin/organoid_etango/test_ventricle'
syto16_img = io.tiff.imread(os.path.join(working_dir, 'syto16_clahe_8x.tif'))
sox2_img = io.tiff.imread(os.path.join(working_dir, 'sox2_clahe_8x.tif'))

cmap = mcolors.ListedColormap(np.random.rand(4096, 3))


def weight_boundary(graph, src, dst, n):
    """
    Handle merging of nodes of a region boundary region adjacency graph.

    This function computes the `"weight"` and the count `"count"`
    attributes of the edge between `n` and the node formed after
    merging `src` and `dst`.


    Parameters
    ----------
    graph : RAG
        The graph under consideration.
    src, dst : int
        The vertices in `graph` to be merged.
    n : int
        A neighbor of `src` or `dst` or both.

    Returns
    -------
    data : dict
        A dictionary with the "weight" and "count" attributes to be
        assigned for the merged node.

    """
    default = {'weight': 0.0, 'count': 0}

    count_src = graph[src].get(n, default)['count']
    count_dst = graph[dst].get(n, default)['count']

    weight_src = graph[src].get(n, default)['weight']
    weight_dst = graph[dst].get(n, default)['weight']

    count = count_src + count_dst
    return {
        'count': count,
        'weight': (count_src * weight_src + count_dst * weight_dst)/count
    }


def merge_boundary(graph, src, dst):
    """Call back called before merging 2 nodes.

    In this case we don't need to do any computation here.
    """
    pass


class TestMGAC(unittest.TestCase):

    @unittest.skip
    def test_random_walker(self):
        threshold = 600
        w = 10


        binary = (syto16_img < threshold).astype(np.float32)
        dilated = binary_dilation(binary, np.ones((w, w))).astype(np.float32)
        unlabeled = dilated - binary
        labels = binary + 1
        labels *= (1 - unlabeled.astype(np.float32))

        mask = ventricle.rw(img_as_float32(syto16_img), labels)

        plt.imshow(syto16_img, clim=[0, 4095])
        plt.imshow(mask, alpha=0.5, clim=[0, 2])
        plt.show()

    @unittest.skip
    def test_felzenszwalb(self):
        mask = felzenszwalb(syto16_img, scale=5, sigma=3, min_size=20)
        plt.imshow(mask, alpha=1)
        plt.show()

    @unittest.skip
    def test_quickshift(self):
        k = 20
        sigma = 0.8
        beta = 0.9  # beta -> 0 corresponds to local maxima
        gamma = 1  # intensity importance relative to spatial proximity

        g = ventricle.smooth(sox2_img, sigma)
        labels = ventricle.quickshift_segmentation(g, k, beta, gamma)

        thresh = 0.9

        edges = sobel(g)

        rag = graph.rag_boundary(labels, edges)
        merged = graph.cut_normalized(labels, rag, thresh)

        # plt.imshow(sox2_img)
        plt.imshow(labels, alpha=0.5)
        plt.show()

        plt.imshow(merged, alpha=0.5)
        plt.show()

    @unittest.skip
    def test_graphcuts(self):
        back_mu = 600
        obj_mu = 1200
        w_const = 10
        sigma = 1

        g = ventricle.smooth(sox2_img, sigma).astype(sox2_img.dtype)
        labels = ventricle.gc(g, back_mu, obj_mu, w_const)

        plt.imshow(syto16_img)
        plt.imshow(labels, alpha=0.5, cmap='tab20b')
        plt.show()

    @unittest.skip
    def test_slic(self):
        n_segments = 300
        compactness = 0.005
        sigma = 2

        img = syto16_img

        labels = slic(img, n_segments=n_segments, compactness=compactness, sigma=sigma)

        g = ventricle.smooth(img, sigma)
        rag = graph.rag_mean_color(g, labels)
        # edges = sobel(g)
        # rag = graph.rag_boundary(labels, edges)

        merged = graph.cut_threshold(labels, rag, thresh=300)
        # merged = graph.merge_hierarchical(labels,
        #                                   rag,
        #                                   thresh=100,
        #                                   rag_copy=False,
        #                                   in_place_merge=True,
        #                                   merge_func=merge_boundary,
        #                                   weight_func=weight_boundary)  # requires rag_boundary
        # merged = graph.cut_normalized(labels, rag, thresh=0.002)

        plt.subplot(121)
        plt.imshow(g, cmap='gray')
        cmap = mcolors.ListedColormap(np.random.rand(256, 3))
        plt.imshow(labels, alpha=0.5, cmap=cmap)
        plt.subplot(122)
        plt.imshow(g, cmap='gray')
        cmap = mcolors.ListedColormap(np.random.rand(256, 3))
        plt.imshow(merged, alpha=0.5, cmap=cmap)
        plt.show()


    @unittest.skip
    def test_compact_watershed(self):
        g = ventricle.smooth(syto16_img, sigma=1)
        gradient = sobel(g)
        labels = watershed(gradient, markers=200, compactness=0.0001)

        # plt.imshow(labels, cmap='tab20b')
        plt.imshow(sox2_img, cmap='gray', alpha=0.5)
        plt.show()

    @unittest.skip
    def test_all_felzenszwalb(self):
        labels = felzenszwalb(syto16_img, scale=5, sigma=2, min_size=10)
        labels2 = felzenszwalb(sox2_img, scale=5, sigma=2, min_size=10)

        g = ventricle.smooth(syto16_img, sigma=2)
        gradient = sobel(g)
        g2 = ventricle.smooth(sox2_img, sigma=2)
        gradient2 = sobel(g2)

        rag = graph.rag_mean_color(g, labels)
        merged = graph.cut_threshold(labels, rag, thresh=200)

        edges = sobel(g)
        rag = graph.rag_boundary(labels, edges)
        merged_sobel = graph.cut_threshold(labels, rag, thresh=50)

        rag = graph.rag_mean_color(g2, labels2)
        merged2 = graph.cut_threshold(labels2, rag, thresh=200)

        edges = sobel(g2)
        rag = graph.rag_boundary(labels2, edges)
        merged_sobel2 = graph.cut_threshold(labels2, rag, thresh=50)

        plt.subplot(231)
        plt.title('Superpixels')
        plt.imshow(syto16_img, cmap='gray')
        cmap = mcolors.ListedColormap(np.random.rand(256, 3))
        plt.imshow(labels, cmap=cmap, alpha=0.5)
        plt.subplot(232)
        plt.title('MFI difference')
        plt.imshow(syto16_img, cmap='gray')
        cmap = mcolors.ListedColormap(np.random.rand(256, 3))
        plt.imshow(merged, alpha=0.3, cmap=cmap)
        plt.subplot(233)
        plt.title('Sobel Weights')
        plt.imshow(syto16_img, cmap='gray')
        cmap = mcolors.ListedColormap(np.random.rand(256, 3))
        plt.imshow(merged_sobel, alpha=0.3, cmap=cmap)
        plt.subplot(234)
        plt.title('Superpixels')
        plt.imshow(sox2_img, cmap='gray')
        cmap = mcolors.ListedColormap(np.random.rand(256, 3))
        plt.imshow(labels, cmap=cmap, alpha=0.5)
        plt.subplot(235)
        plt.title('MFI difference')
        plt.imshow(sox2_img, cmap='gray')
        cmap = mcolors.ListedColormap(np.random.rand(256, 3))
        plt.imshow(merged2, alpha=0.3, cmap=cmap)
        plt.subplot(236)
        plt.title('Sobel Weights')
        plt.imshow(sox2_img, cmap='gray')

        plt.imshow(merged_sobel2, alpha=0.3, cmap=cmap)
        plt.show()

    @unittest.skip
    def test_region_mfi(self):
        labels = felzenszwalb(syto16_img, scale=5, sigma=2, min_size=10)
        g = ventricle.smooth(syto16_img, sigma=2)
        rag = graph.rag_mean_color(g, labels)
        merged = graph.cut_threshold(labels, rag, thresh=200)

        plt.imshow(syto16_img)
        plt.imshow(merged, alpha=0.4, cmap=cmap)
        plt.show()

        mfi = ventricle.calculate_region_mfi(sox2_img, merged)
        volumes = ventricle.calculate_volume(merged)

        selem = np.ones((10, 10))
        sox2_in, sox2_out, sox2_ratio = ventricle.calculate_relative_sox2(merged,
                                                                          sox2_img,
                                                                          selem)

        ratio_thresh = 1.4
        out_thresh = 900
        idx = np.where(np.logical_and(sox2_ratio > ratio_thresh, sox2_out > out_thresh))[0]
        ventricle_lbls = idx + 1

        filtered = ventricle.filter_labels(merged, ventricle_lbls)

        plt.subplot(311)
        plt.plot(sox2_in)
        plt.subplot(312)
        plt.plot(sox2_out)
        plt.plot([0, len(sox2_out)], [out_thresh, out_thresh])
        plt.subplot(313)
        plt.plot(sox2_ratio)
        plt.plot([0, len(sox2_ratio)], [ratio_thresh, ratio_thresh])
        plt.plot(idx, sox2_ratio[idx], 'r*')
        plt.show()

        plt.imshow(sox2_img, cmap='gray')
        plt.imshow(filtered, alpha=0.5, cmap=cmap)
        plt.show()

    @unittest.skip
    def test_slic_3d(self):
        n_segments = 1000
        compactness = 0.005
        sigma = 2

        labels = slic(syto16_img,
                      n_segments=n_segments,
                      compactness=compactness,
                      sigma=sigma,
                      multichannel=False)

        g = ventricle.smooth(syto16_img, sigma)
        rag = graph.rag_mean_color(g, labels)
        merged = graph.cut_threshold(labels, rag, thresh=300)

        plt.subplot(311)
        plt.imshow(syto16_img[60])
        plt.imshow(merged[60], cmap=cmap, clim=[0, 4095], alpha=0.4)
        plt.subplot(312)
        plt.imshow(syto16_img[70])
        plt.imshow(merged[70], cmap=cmap, clim=[0, 4095], alpha=0.4)
        plt.subplot(313)
        plt.imshow(syto16_img[80])
        plt.imshow(merged[80], cmap=cmap, clim=[0, 4095], alpha=0.4)
        plt.show()


class TestGraphCuts(unittest.TestCase):

    @unittest.skip
    def test_sox_gc(self):
        back_mu = 100
        obj_mu = 1600
        w_const = 10

        sigma = 1
        g = ventricle.smooth(syto16_img, sigma).astype(syto16_img.dtype)

        seg = ventricle.gc(g, back_mu, obj_mu, w_const)

        plt.subplot(311)
        plt.imshow(syto16_img[60], cmap='gray')
        plt.imshow(seg[60], cmap=cmap, clim=[0, 4095], alpha=0.4)
        plt.subplot(312)
        plt.imshow(syto16_img[70], cmap='gray')
        plt.imshow(seg[70], cmap=cmap, clim=[0, 4095], alpha=0.4)
        plt.subplot(313)
        plt.imshow(syto16_img[80], cmap='gray')
        plt.imshow(seg[80], cmap=cmap, clim=[0, 4095], alpha=0.4)
        plt.show()

        tifffile.imsave(os.path.join(working_dir, 'seg.tif'), seg)

    # @unittest.skip
    def test_filter_ventricles(self):
        seg = tifffile.imread(os.path.join(working_dir, 'seg.tif'))
        seg = 1-seg

        labels, nb_labels = ventricle.label(seg)

        labels, mask_largest = ventricle.remove_largest_region(labels)

        mask_ventricles = (labels > 0)

        tifffile.imsave(os.path.join(working_dir, 'ventricles.tif'), mask_ventricles)
        tifffile.imsave(os.path.join(working_dir, 'background.tif'), mask_largest)

