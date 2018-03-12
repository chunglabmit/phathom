import phathom
import numpy as np
import unittest
import os
import tempfile

class TestImageIO(unittest.TestCase):
	def test_imread(self):
		filename = os.path.join(os.path.split(__file__)[0], 'example.tif')
		data = phathom.imageio.imread(filename)
		self.assertEqual(data.shape, (64, 128, 128))
		self.assertEqual(data.dtype, 'uint16')

	def test_imsave(self):
		arr = np.random.random((32, 32, 32))
		filename = os.path.join(tempfile.gettempdir(), "imsave_test.tif")
		phathom.imageio.imsave(filename, arr)
		tmp = phathom.imageio.imread(filename)
		assert np.all(arr == tmp)
		assert arr.dtype == tmp.dtype
