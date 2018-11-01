import unittest
import os
import numpy as np
from phathom import io
from phathom import synthetic
from phathom.phenotype import celltype

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm

from phathom.plotting import plot_mip, zprojection
from skimage.viewer import CollectionViewer
import tifffile

seed = 295

try:
    working_dir = '/media/jswaney/Drive/Justin/organoid_etango_small/test_celltype'
    syto16_img = io.tiff.imread(os.path.join(working_dir, 'syto16.tif'))
    sox2_img = io.tiff.imread(os.path.join(working_dir, 'sox2.tif'))
    tbr1_img = io.tiff.imread(os.path.join(working_dir, 'tbr1.tif'))
    test_images = True
except FileNotFoundError:
    test_images = False

reason = "Only testing synthetic data, not real images"


@unittest.skipIf(test_images == False, reason)
class TestImages(unittest.TestCase):

    def setUp(self):
        self.images = [syto16_img, sox2_img, tbr1_img]

    def test_shapes(self):
        for img in self.images:
            self.assertEqual(img.shape, (32, 256, 256))

    def test_dtypes(self):
        for img in self.images:
            self.assertTrue(img.dtype == np.uint8)


class TestNegativeCurvatureProduct(unittest.TestCase):

    def test_blobs(self):
        np.random.seed(seed)
        sigma = 2
        blobs = synthetic.generate_blobs(10, (32, 128, 128), sigma=sigma)
        ncp = celltype.negative_curvature_product(blobs, sigma)
        # plot_mip(-ncp)

        blob_loc = np.where(blobs > 0.5)
        no_blob_loc = np.where(blobs < 0.01)

        self.assertTrue(np.all(np.abs(ncp[blob_loc])) > 1e-5)
        self.assertTrue(np.all(np.abs(ncp[no_blob_loc])) < 1e-5)

    @unittest.skipIf(test_images == False, reason)
    def test_image(self):
        sigma = (1.2, 3, 3)

        ncp = celltype.negative_curvature_product(sox2_img, sigma)

        plt.imshow(-ncp[0])
        plt.show()


class TestCalculateNucleusProbabilty(unittest.TestCase):

    def test_blobs(self):
        np.random.seed(seed)
        sigma = 2
        steepness = 500
        offset = 0.0005
        blobs = synthetic.generate_blobs(10, (32, 128, 128), sigma=sigma)
        prob = celltype.nucleus_probability(blobs, sigma, steepness, offset)
        # plot_mip(prob)
        blob_loc = np.where(blobs > 0.5)
        no_blob_loc = np.where(blobs < 0.01)

        self.assertTrue(np.all(prob[blob_loc]) > 0.5)
        self.assertTrue(np.all(prob[no_blob_loc] < 0.5))

    @unittest.skipIf(test_images == False, reason)
    def test_image(self):
        sigma = (1.2, 3, 3)
        steepness = 500
        offset = 0.0005

        prob = celltype.nucleus_probability(syto16_img, sigma, steepness, offset)

        viewer = CollectionViewer(prob)
        viewer.show()


class TestNucleiCentersNCP(unittest.TestCase):

    def test_blobs(self):
        np.random.seed(seed)

        nb_pts = 10
        sigma = 2
        min_distance = 1
        threshold_abs = 1e-5
        threshold_rel = 0.1

        blobs = synthetic.generate_blobs(nb_pts, (32, 128, 128), sigma=sigma)

        centers = celltype.nuclei_centers_ncp(blobs,
                                              sigma,
                                              min_distance=min_distance,
                                              threshold_abs=threshold_abs,
                                              threshold_rel=threshold_rel)

        self.assertEqual(centers.shape[0], nb_pts)

    @unittest.skipIf(test_images == False, reason)
    def test_images(self):
        sigma = (1.2, 3.0, 3.0)
        min_distance = 2
        threshold_abs = 1e-9
        threshold_rel = 0

        centers = celltype.nuclei_centers_ncp(syto16_img,
                                                 sigma,
                                                 min_distance=min_distance,
                                                 threshold_abs=threshold_abs,
                                                 threshold_rel=threshold_rel)

        # zprojection(syto16_img, centers, zlim=[8, 12])


class TestNucleiCentersProbability(unittest.TestCase):

    def test_blobs(self):
        np.random.seed(seed)

        nb_pts = 10
        sigma = 2
        steepness = 500
        offset = 0.005
        h = 0.001
        threshold = 0.1

        blobs = synthetic.generate_blobs(nb_pts, (32, 128, 128), sigma=sigma)
        prob = celltype.nucleus_probability(blobs, sigma, steepness, offset)
        centers = celltype.nuclei_centers_probability(prob, threshold, h)

        self.assertEqual(centers.shape[0], nb_pts)

    @unittest.skipIf(test_images == False, reason)
    def test_image(self):
        sigma = (1.2, 3, 3)
        steepness = 500
        offset = 0.0005
        threshold = 0.05
        h = 0.001

        prob = celltype.nucleus_probability(syto16_img, sigma, steepness, offset)
        centers = celltype.nuclei_centers_probability(prob, threshold, h)

        prob[tuple(centers.T)] = 1
        viewer = CollectionViewer(prob)
        viewer.show()


class TestNucleiCenteredIntensities(unittest.TestCase):

    @unittest.skipIf(test_images == False, reason)
    def test_image(self):
        sigma = (1.2, 3.0, 3.0)
        steepness = 500
        offset = 0.0005
        h = 0.001
        threshold = 0.1
        radius = 1

        prob = celltype.nucleus_probability(syto16_img, sigma, steepness, offset)
        centers = celltype.nuclei_centers_probability(prob, threshold, h)
        sox2 = celltype.nuclei_centered_intensities(sox2_img, sigma, centers, radius)
        tbr1 = celltype.nuclei_centered_intensities(tbr1_img, sigma, centers, radius)

        sox2_mfi = np.asarray([x.mean() for x in sox2])
        tbr1_mfi = np.asarray([x.mean() for x in tbr1])

        sox2_std = np.asarray([x.std() for x in sox2])
        tbr1_std = np.asarray([x.std() for x in tbr1])

        plt.xlabel('SOX2 MFI')
        plt.ylabel('TBR1 MFI')
        plt.hist2d(sox2_mfi, tbr1_mfi, bins=50, norm=mcolors.PowerNorm(0.8))
        plt.show()

        idx = np.where(sox2_mfi < 35)[0]
        plt.hist(tbr1_mfi[idx], bins=64)
        plt.xlabel('TBR1 MFI in SOX2 negatives')
        plt.showsox2_img()

        plt.subplot(121)
        plt.plot(sox2_mfi, sox2_std, '*')
        plt.ylabel('Standard Deviation')
        plt.xlabel('MFI')
        plt.subplot(122)
        plt.plot(tbr1_mfi, tbr1_std, '*')
        plt.xlabel('MFI')
        plt.show()


# class TestThresholdMFI(unittest.TestCase):

    # @unittest.skipIf(test_images == False, reason)
    # def test_images(self):
    #     sigma = (1.8, 3.0, 3.0)
    #     steepness = 500
    #     offset = 0.0005
    #     h = 0.001
    #     threshold = 0.1
    #     radius = 1
    #     sox2_mfi_threshold = 35
    #     tbr1_mfi_threshold = 30
    #
    #     prob = celltype.nucleus_probability(syto16_img, sigma, steepness, offset)
    #     centers = celltype.nuclei_centers_probability(prob, threshold, h)
    #     sox2 = celltype.nuclei_centered_intensities(sox2_img, sigma, centers, radius)
    #     tbr1 = celltype.nuclei_centered_intensities(tbr1_img, sigma, centers, radius)
    #
    #     sox2_mfi = celltype.calculate_mfi(sox2)
    #     tbr1_mfi = celltype.calculate_mfi(tbr1)
    #
    #     sox2_labels = celltype.threshold_mfi(sox2_mfi, sox2_mfi_threshold)
    #     tbr1_labels = celltype.threshold_mfi(tbr1_mfi, tbr1_mfi_threshold)
    #
    #     plt.hist(sox2_mfi[np.where(sox2_labels == 0)], bins=64, range=[0, 100])
    #     plt.hist(sox2_mfi[np.where(sox2_labels == 1)], bins=64, range=[0, 100])
    #     plt.show()
    #
    #     sox2_img[tuple(centers[np.where(sox2_labels == 1)[0]].T)] = 255
    #     viewer = CollectionViewer(sox2_img)
    #     viewer.show()
    #
    #     plt.hist(tbr1_mfi[np.where(tbr1_labels == 0)], bins=64, range=[0, 100])
    #     plt.hist(tbr1_mfi[np.where(tbr1_labels == 1)], bins=64, range=[0, 100])
    #     plt.show()
    #
    #     tbr1_img[tuple(centers[np.where(tbr1_labels == 1)[0]].T)] = 255
    #     viewer = CollectionViewer(tbr1_img)
    #     viewer.show()
    #
    #     sox2_pts = np.zeros(sox2_img.shape, np.uint8)
    #     sox2_pts[tuple(centers[np.where(sox2_labels == 1)[0]].T)] = 255
    #
    #     tbr1_pts = np.zeros(tbr1_img.shape, np.uint8)
    #     tbr1_pts[tuple(centers[np.where(tbr1_labels == 1)[0]].T)] = 255
    #
    #     cell_table = np.column_stack((centers, sox2_labels, tbr1_labels))
    #     np.save(os.path.join(working_dir, 'cell_table.npy'), cell_table)

#
# class TestQueryNeighbors(unittest.TestCase):
#
#     def test(self):
#         cell_table = np.load(os.path.join(working_dir, 'cell_table.npy'))
#         pts = cell_table[:, 0:3]
#         sox2_labels = cell_table[:, -2]
#         tbr1_labels = cell_table[:, -1]
#
#         distances, indices = celltype.query_neighbors(pts, n_neighbors=10)
#         print(indices)


@unittest.skipIf(test_images == False, reason)
class TestLocalDensities(unittest.TestCase):

    def test(self):
        cell_table = np.load(os.path.join(working_dir, 'cell_table.npy'))
        pts = cell_table[:, 0:3]
        sox2_labels = cell_table[:, -2]
        tbr1_labels = cell_table[:, -1]

        n_neighbors = 50

        distances, indices = celltype.query_neighbors(pts, n_neighbors=n_neighbors)
        features = celltype.local_densities(distances, indices, sox2_labels, tbr1_labels)

        plt.plot(features[:, 0], features[:, 1], '*')
        plt.show()

        # celltype.cluster_dendrogram(features)

        connectivity = celltype.connectivity(pts, n_neighbors=n_neighbors)
        print(connectivity)
        region_labels = celltype.cluster(features, n_clusters=2, connectivity=connectivity)

        # idx0 = np.where(region_labels == 0)[0]
        # idx1 = np.where(region_labels == 1)[0]

        # print(len(idx0), len(idx1))
        #
        # region1 = np.zeros(syto16_img.shape)
        # region1[tuple(pts[idx0].T)] = 1
        #
        # region2 = np.zeros(syto16_img.shape)
        # region2[tuple(pts[idx1].T)] = 1
        #
        # viewer = CollectionViewer(region1)
        # viewer.show()

        # r0_sox2 = sox2_labels[idx0].sum()
        # r0_tbr1 = tbr1_labels[idx0].sum()
        #
        # r1_sox2 = sox2_labels[idx1].sum()
        # r1_tbr1 = tbr1_labels[idx1].sum()
        #
        # print('Region 0 has {} sox2+ and {} tbr1+'.format(r0_sox2/len(idx0), r0_tbr1/len(idx0)))
        # print('Region 1 has {} sox2+ and {} tbr1+'.format(r1_sox2 / len(idx1), r1_tbr1 / len(idx1)))
        #
        # vor = celltype.voronoi(np.column_stack((pts[:, 2], -pts[:, 1])))
        # celltype.voronoi_plot(vor)
        #
        # vor3d = celltype.voronoi(pts)

        img = celltype.rasterize_regions(pts, region_labels, syto16_img.shape)

        tifffile.imsave(os.path.join(working_dir, 'seg.tif'), img)
