import unittest
import numpy as np
import phathom.segmentation.segmentation as pss


class TestAdaptiveThreshold(unittest.TestCase):

    def test_zeros(self):
        t = pss.adaptive_threshold(np.zeros((100, 100, 100)))
        np.testing.assert_array_almost_equal(t, 0)

    def test_bimodal(self):
        a = np.zeros(((100, 100, 100)))
        z, y, x = np.mgrid[:100, :100, :100]
        odd = ((x + y + z) & 1) == 1
        a[odd] = 1
        t = pss.adaptive_threshold(a)
        self.assertTrue(np.all(t > 0))
        self.assertTrue(np.all(t < 1))

    def test_different(self):
        a = np.zeros((100, 100, 100))
        z, y, x = np.mgrid[:100, :100, :100]
        odd = ((x + y + z) & 1) == 1
        a[odd] = 1
        a[:50, :50, :50] += 1
        t = pss.adaptive_threshold(a, sigma=0)
        self.assertGreater(t[25, 25, 25], 1)
        self.assertLess(t[25, 25, 25], 2)
        self.assertGreater(t[25, 25, 75], 0)
        self.assertLess(t[25, 25, 75], 1)
        self.assertAlmostEqual(t[50, 25, 25],
                               (t[25, 25, 25] +  t[75, 25, 25]) / 2,
                               delta=.1)

    def test_low(self):
        a = np.zeros((100, 100, 100))
        z, y, x = np.mgrid[:100, :100, :100]
        odd = ((x + y + z) & 1) == 1
        a[odd] = 1
        t = pss.adaptive_threshold(a, low_threshold=.6)
        np.testing.assert_array_less(.6, t)

    def test_high(self):
        a = np.zeros((100, 100, 100))
        z, y, x = np.mgrid[:100, :100, :100]
        odd = ((x + y + z) & 1) == 1
        a[odd] = 1
        t = pss.adaptive_threshold(a, high_threshold=.4)
        np.testing.assert_array_less(t, .4)

