import unittest
import numpy as np
from phathom import synthetic
from phathom.registration import pcloud


class TestRotationMatrix(unittest.TestCase):
    def test_90deg(self):
        thetas = [np.pi/2, 0, 0]
        target = np.array([[1, 0, 0],
                           [0, 0, -1],
                           [0, 1, 0]])
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
        self.assertTrue(np.allclose(rotated[1], -points[2]))
        self.assertTrue(np.allclose(rotated[2], points[1]))

    def test_with_center(self):
        thetas = [np.pi/4, -np.pi, np.pi/3]
        center = [75.2, 64.0, 42.8]
        points = np.array([[c, 0] for c in center])
        rotated = pcloud.rotate(points, thetas, center)
        self.assertTrue(np.allclose(np.asarray(rotated)[:, 0], np.asarray(points)[:, 0]))
        self.assertFalse(np.allclose(np.asarray(rotated)[:, 1], np.asarray(points)[:, 1]))