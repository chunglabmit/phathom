import unittest
import numpy as np
from scipy.ndimage.morphology import distance_transform_edt
import phathom.segmentation.graphcuts as graphcuts
from phathom import utils
import zarr
import tempfile
import os


class TestPoissonPdf(unittest.TestCase):

    def test_zero(self):
        p = graphcuts.poisson_pdf(0, 1)
        self.assertAlmostEqual(p, np.exp(-1))

    def test_large(self):
        p = graphcuts.poisson_pdf(1e6, 1)
        self.assertAlmostEqual(p, 0.0)


class TestInformationGain(unittest.TestCase):

    def test_zero(self):
        ig = graphcuts.information_gain(0)
        self.assertAlmostEqual(ig, np.exp(1))

    def test_one(self):
        ig = graphcuts.information_gain(1)
        self.assertAlmostEqual(ig, 1)


class TestHist(unittest.TestCase):

    def test_single_value(self):
        n = 5
        data = np.ones(n)
        left_edges, freq, width = graphcuts.hist(data, bins=256, _range=(0, 256))
        self.assertEqual(freq[1], n)
        self.assertEqual(freq[0], 0)

    def test_arange(self):
        data = np.arange(256)
        left_edges, freq, width = graphcuts.hist(data, bins=256, _range=(0, 256))
        self.assertTrue(np.all(freq == 1))


class TestFitPoissonMixture(unittest.TestCase):

    def test_simple(self):
        data = np.array([100, 200])
        bins = 256
        _range = (0, bins)
        T = 150

        hist_output = graphcuts.hist(data, bins, _range)
        back_mu, obj_mu = graphcuts.fit_poisson_mixture(hist_output, T)

        self.assertAlmostEqual(back_mu, 100, 0)
        self.assertAlmostEqual(obj_mu, 200, 0)


class TestEntropy(unittest.TestCase):

    def test_better_threshold(self):
        h1 = graphcuts.entropy(25, 75, 50, 100)
        h2 = graphcuts.entropy(25, 75, 35, 100)
        self.assertGreater(h1, h2)


class TestMaxEntropy(unittest.TestCase):

    def test_error(self):
        data = np.ones(3)
        hist_output = graphcuts.hist(data, 256, (0, 256))
        with self.assertRaises(ValueError):
            graphcuts.max_entropy(hist_output, np.float32)

    def test(self):
        data = np.arange(256, dtype='uint8')
        hist_output = graphcuts.hist(data, 256, (0, 256))
        T_star, back_mu_star, obj_mu_star = graphcuts.max_entropy(hist_output,
                                                                  data.dtype)
        self.assertAlmostEqual(T_star, 159)
        self.assertAlmostEqual(back_mu_star, 79.5)
        self.assertAlmostEqual(obj_mu_star, 207.5)


class TestGraphCuts(unittest.TestCase):

    def setUp(self):
        sz = [10, 100, 100]
        dist = 5
        noise_intensity = 1
        noise_std = 1
        signal_intensity = 100

        segmentation = np.zeros(sz)
        segmentation[5, 50, 50] = 1
        segmentation = distance_transform_edt(1 - segmentation)
        segmentation = (segmentation < dist).astype(np.uint8)

        data = np.abs(np.random.normal(size=sz, loc=noise_intensity, scale=noise_std))
        data += segmentation * signal_intensity
        data = data.astype(np.uint8)

        self.segmentation = segmentation
        self.data = data

    def test(self):
        hist_output = graphcuts.hist(self.data, bins=256, _range=(0, 256))
        T, back_mu, obj_mu = graphcuts.max_entropy(hist_output, self.data.dtype)
        seg = graphcuts.graph_cuts(self.data, back_mu, obj_mu, w_const=0.8, w_grad=0.5)

        difference = np.linalg.norm(seg.astype(np.uint8) - self.segmentation)

        self.assertAlmostEqual(difference, 0, places=0)


class TestParallelGraphCuts(unittest.TestCase):

    def setUp(self):
        sz = (10, 100, 100)
        dist = 5
        noise_intensity = 1
        noise_std = 1
        signal_intensity = 100

        segmentation = np.zeros(sz)
        segmentation[5, 50, 50] = 1
        segmentation = distance_transform_edt(1 - segmentation)
        segmentation = (segmentation < dist).astype(np.uint8)

        data = np.abs(np.random.normal(size=sz, loc=noise_intensity, scale=noise_std))
        data += segmentation * signal_intensity
        data = data.astype(np.uint8)

        self.segmentation = segmentation
        self.data = zarr.empty(sz, chunks=(10, 10, 10), dtype=np.uint8)
        self.data[:] = data

        filename = os.path.join(tempfile.gettempdir(), 'pgc.zarr')
        self.out = zarr.open(filename, mode='w', shape=sz, chunks=(10, 10, 10), dtype=np.uint8)

        self.out_shm = utils.SharedMemory(sz, np.uint8)

    def test_zarr(self):
        hist_output = graphcuts.hist(self.data, bins=256, _range=(0, 256))
        T, back_mu, obj_mu = graphcuts.max_entropy(hist_output, self.data.dtype)
        gc_args = {'back_mu': back_mu,
                   'obj_mu': obj_mu,
                   'w_const': 1,
                   'w_grad': 0}
        graphcuts.parallel_graph_cuts(self.data,
                                      self.out,
                                      overlap=2,
                                      chunks=(10, 20, 20),
                                      nb_workers=4,
                                      **gc_args)
        difference = np.linalg.norm(self.out - self.segmentation)
        self.assertAlmostEqual(difference, 0, places=0)

    def test_shm(self):
        hist_output = graphcuts.hist(self.data, bins=256, _range=(0, 256))
        T, back_mu, obj_mu = graphcuts.max_entropy(hist_output, self.data.dtype)
        gc_args = {'back_mu': back_mu,
                   'obj_mu': obj_mu,
                   'w_const': 1,
                   'w_grad': 0}

        graphcuts.parallel_graph_cuts(self.data,
                                      self.out_shm,
                                      overlap=2,
                                      chunks=(10, 20, 20),
                                      nb_workers=1,
                                      **gc_args)

        with self.out_shm.txn() as a:
            difference = np.linalg.norm(a - self.segmentation)

        self.assertAlmostEqual(difference, 0, places=0)

