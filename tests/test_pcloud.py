import unittest
import numpy as np
from phathom import synthetic
from phathom.registration import pcloud


class TestRotationMatrix(unittest.TestCase):
    def test_90deg(self):
        thetas = [np.pi/2, 0, 0]
        target = np.array([[1, 0, 0],
                           [0, 0, 1],
                           [0, -1, 0]])
        r = pcloud.rotation_matrix(thetas)
        self.assertAlmostEqual(np.linalg.norm(r-target), 0)

        thetas = [0, np.pi/2, 0]
        target = np.array([[0, 0, 1],
                           [0, 1, 0],
                           [-1, 0, 0]])
        r = pcloud.rotation_matrix(thetas)
        self.assertAlmostEqual(np.linalg.norm(r-target), 0)

        thetas = [0, 0, np.pi/2]
        target = np.array([[0, -1, 0],
                           [1, 0, 0],
                           [0, 0, 1]])
        r = pcloud.rotation_matrix(thetas)
        self.assertAlmostEqual(np.linalg.norm(r-target), 0)

    def test_360deg(self):
        thetas = [2*np.pi, 2*np.pi, 2*np.pi]
        target = np.array([[1, 0, 0],
                           [0, 1, 0],
                           [0, 0, 1]])
        r = pcloud.rotation_matrix(thetas)
        self.assertAlmostEqual(np.linalg.norm(r - target), 0)


class TestRotation(unittest.TestCase):
    def test_without_center(self):
        n = 10
        shape = (128, 128, 128)
        thetas = [np.pi/2, 0, 0]
        points = synthetic.random_points(n, shape)
        rotated = pcloud.rotate(points, thetas)
        self.assertTrue(np.allclose(rotated[0], points[0]))
        self.assertTrue(np.allclose(rotated[1], points[2]))
        self.assertTrue(np.allclose(rotated[2], -points[1]))

    def test_with_center(self):
        thetas = [np.pi/4, -np.pi, np.pi/3]
        center = [75.2, 64.0, 42.8]
        points = np.array([[c, 0] for c in center])
        rotated = pcloud.rotate(points, thetas, center)
        self.assertTrue(np.allclose(np.asarray(rotated)[:, 0], np.asarray(points)[:, 0]))
        self.assertFalse(np.allclose(np.asarray(rotated)[:, 1], np.asarray(points)[:, 1]))

    def test_single_point(self):
        pt = [1.2, 5.7, 9.2]
        points = tuple(np.array([p]) for p in pt)

        thetas = [np.pi/2, 0, 0]
        rotated = pcloud.rotate(points, thetas)
        self.assertTrue(np.allclose(rotated[0], points[0]))
        self.assertTrue(np.allclose(rotated[1], points[2]))
        self.assertTrue(np.allclose(rotated[2], -points[1]))

        thetas = [0, np.pi/2, 0]
        rotated = pcloud.rotate(points, thetas)
        self.assertTrue(np.allclose(rotated[0], points[2]))
        self.assertTrue(np.allclose(rotated[1], points[1]))
        self.assertTrue(np.allclose(rotated[2], -points[0]))

        thetas = [0, 0, np.pi/2]
        rotated = pcloud.rotate(points, thetas)
        self.assertTrue(np.allclose(rotated[0], -points[1]))
        self.assertTrue(np.allclose(rotated[1], points[0]))
        self.assertTrue(np.allclose(rotated[2], points[2]))


class TestGeometricHash(unittest.TestCase):
    def test_simple(self):
        center = np.array([1, 2, 3])
        vectors = np.array([[2, 2, 3],
                            [2, 3, 4],
                            [1, 2, 7]])
        target = np.array([4, 0, 1, 1, 1, 1])
        features = pcloud.geometric_hash(center, vectors)
        self.assertTrue(np.allclose(features, target))

    def test_similar_pts(self):
        center = np.array([1, 2, 3])
        vectors1 = np.array([[2, 2, 3],
                             [2, 3, 4],
                             [1, 2, 7]])
        vectors2 = np.array([[2, 2, 3],
                             [2, 3, -4],
                             [1, 2, 7]])
        features1 = pcloud.geometric_hash(center, vectors1)
        features2 = pcloud.geometric_hash(center, vectors2)
        self.assertFalse(np.allclose(features1, features2))

    def test_rotated_pts(self):
        center = np.array([0, 0, 0])
        vectors1 = np.array([[2, 2, 3],
                             [2, 3, 4],
                             [1, 2, 7]])
        vectors2 = np.array([[2, 3, -2],
                             [2, 4, -3],
                             [1, 7, -2]])
        features1 = pcloud.geometric_hash(center, vectors1)
        features2 = pcloud.geometric_hash(center, vectors2)
        self.assertTrue(np.allclose(features1, features2))


class TestGeometricFeatures(unittest.TestCase):
    def test_parallel_same(self):
        nb_pts = 20
        shape = (128, 128, 128)
        points = synthetic.random_points(nb_pts, shape)

        nb_workers = [2, 4, 8]
        features_1 = pcloud.geometric_features(np.asarray(points).T, 1)
        for n in nb_workers:
            features_n = pcloud.geometric_features(np.asarray(points).T, n)
            self.assertTrue(np.allclose(features_1, features_n))


class TestFindSimilar(unittest.TestCase):
    def test(self):
        feat_stationary = np.array([[1, 2, 3]])
        feat_moving = np.array([[1, 4, 3],
                                [2, 2, 3],
                                [1, 2, 6]])
        dist, idx = pcloud.find_similar(feat_stationary, feat_moving)
        self.assertEqual(idx[0, 0], 1)
        self.assertAlmostEqual(dist[0, 0], 1)


class TestCheckDistance(unittest.TestCase):
    def test_simple3d(self):
        dists = np.array([1, 2, 3, 0])
        max_dists = [4, 3, 2, 1, 0]
        true_idx = [[0, 1, 2, 3],
                    [0, 1, 3],
                    [0, 3],
                    [3],
                    None]
        for max_dist, true in zip(max_dists, true_idx):
            close = pcloud.check_distance(dists, max_dist)
            if close is None:
                self.assertEqual(close, true)
            else:
                self.assertTrue(np.allclose(close[0], np.asarray(true)))


class TestProminence(unittest.TestCase):
    def test(self):
        d1 = np.array([1, 2, 3])
        d2 = np.array([2, 2.5, 3])
        true = np.array([0.5, 0.8, 1])
        prom = pcloud.prominence(d1, d2)
        self.assertTrue(np.allclose(prom, true))

    def test_clipping(self):
        d1 = np.array([0, 0])
        d2 = np.array([0, 1])
        true = np.array([0, 0])
        prom = pcloud.prominence(d1, d2)
        self.assertTrue(np.allclose(prom, true))


class TestCheckProminence(unittest.TestCase):
    def test(self):
        prom = np.array([1, 0.05, 0.2, 0.1, 0.3, 0.9])
        thresholds = [0.5, 0.2, 0.1, 0]
        true_idx = [[1, 2, 3, 4],
                    [1, 3],
                    [1],
                    None]
        for threshold, true in zip(thresholds, true_idx):
            close = pcloud.check_prominence(prom, threshold)
            if close is None:
                self.assertEqual(close, true)
            else:
                self.assertTrue(np.allclose(close[0], np.asarray(true)))


class TestGlobalMatching(unittest.TestCase):

    def setUp(self):
        self.points_fixed = np.array([[0, 0, 0],
                                      [1, 0, 0],
                                      [0, 0, 3],
                                      [0, 2, 0]])
        self.points_moving = np.array([[1, 0, 0],
                                       [0, 0, 0],
                                       [0, 0, -2],
                                       [0, 3, 0]])  # rotated 90 degrees
        self.points_moving_noise = np.array([[1.1, 0, 0],
                                             [0, 0.2, 0],
                                             [0, 0, -2.3],
                                             [0, 3.4, 0]])  # rotated 90 degrees
        self.feat_fixed = pcloud.geometric_features(self.points_fixed, nb_workers=1)
        self.feat_moving = pcloud.geometric_features(self.points_moving, nb_workers=1)
        self.feat_moving_noise = pcloud.geometric_features(self.points_moving_noise, nb_workers=1)

    def test_exact(self):
        idx_fixed, idx_moving = pcloud.global_matching(self.feat_fixed, self.feat_moving)
        self.assertTrue(np.all(idx_fixed == np.array([0, 1, 2, 3])))
        self.assertTrue(np.all(idx_moving == np.array([1, 0, 3, 2])))

    def test_noisy(self):

        idx_fixed, idx_moving = pcloud.global_matching(self.feat_fixed, self.feat_moving_noise)
        self.assertTrue(np.all(idx_fixed == np.array([0, 1, 2, 3])))
        self.assertTrue(np.all(idx_moving == np.array([1, 0, 3, 2])))

    def test_max_fdist(self):
        idx_fixed, idx_moving = pcloud.global_matching(self.feat_fixed,
                                                       self.feat_moving_noise,
                                                       max_fdist=0.6)
        self.assertTrue(np.all(idx_fixed == np.array([0, 1])))
        self.assertTrue(np.all(idx_moving == np.array([1, 0])))

        idx_fixed, idx_moving = pcloud.global_matching(self.feat_fixed,
                                                       self.feat_moving_noise,
                                                       max_fdist=0.5)
        self.assertTrue(np.all(idx_fixed == np.array([0])))
        self.assertTrue(np.all(idx_moving == np.array([1])))
    
    def test_prom_thresh(self):
        # prominences = [0.339, 0.4517, 0.400, 0.3742]
        idx_fixed, idx_moving = pcloud.global_matching(self.feat_fixed,
                                                       self.feat_moving_noise,
                                                       prom_thresh=0.41)
        self.assertTrue(np.all(idx_fixed == np.array([0, 2, 3])))
        self.assertTrue(np.all(idx_moving == np.array([1, 3, 2])))

    def test_both_filters(self):
        idx_fixed, idx_moving = pcloud.global_matching(self.feat_fixed,
                                                       self.feat_moving_noise,
                                                       max_fdist=0.6,
                                                       prom_thresh=0.41)
        self.assertTrue(np.all(idx_fixed == np.array([0])))
        self.assertTrue(np.all(idx_moving == np.array([1])))


class TestNeighborhoodMatching(unittest.TestCase):
    def test(self):
        pass
