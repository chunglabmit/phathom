import unittest
import numpy as np
from phathom import synthetic


class TestGeneratePoints(unittest.TestCase):
    def test(self):
        n = 100
        shape = (64, 64, 64)
        points = synthetic.random_points(n=n, shape=shape)
        self.assertEqual(len(points), len(shape))
        for dim, lim in zip(points, shape):
            self.assertGreaterEqual(dim.min(), 0)
            self.assertLess(dim.max(), lim)


class TestPointsToBinary(unittest.TestCase):
    def test(self):
        n = 200
        shape = (128, 128, 128)
        points = synthetic.random_points(n=n, shape=shape)
        binary = synthetic.points_to_binary(points=points, shape=shape)
        self.assertEqual(binary.shape, shape)
        self.assertTrue(np.all(binary[points] == 255))
        self.assertTrue(binary.min() == 0)

