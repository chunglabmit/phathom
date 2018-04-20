import numpy as np
import unittest
import zarr

from phathom.registration import registration


class TestRegistration(unittest.TestCase):
    def test_empty_detect_blobs_parallel(self):
        z = zarr.zeros((100, 100, 100),
                       chunks=(25, 25, 25),
                       dtype=np.uint16)
        blobs = registration.detect_blobs_parallel(
            z, 1.0, 5, .5, 5)
        self.assertEqual(len(blobs), 0)

    def test_one_detect_blobs_parallel(self):
        z = zarr.zeros((100, 100, 100),
                       chunks=(25, 25, 25),
                       dtype=np.uint16)
        #
        # A blob at 10, 11, 12 of 7 voxels
        #
        is_blob = np.sqrt(np.sum(np.mgrid[-12:13, -11:14, -10:15]**2, 0)) < 7
        z[:25, :25, :25] = is_blob * np.uint16(200)
        blobs = registration.detect_blobs_parallel(z, 1.0, 5, .5, 5)
        self.assertEqual(len(blobs), 1)
        self.assertAlmostEqual(blobs[0, 0], 12, 1)
        self.assertAlmostEqual(blobs[0, 1], 11, 1)
        self.assertAlmostEqual(blobs[0, 2], 10, 1)

if __name__ == '__main__':
    unittest.main()
