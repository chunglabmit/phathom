import phathom
import numpy as np


def test_imread():
	filename = 'example.tif'
	data = phathom.imageio.imread(filename)
	assert data.shape == (64, 128, 128)
	assert data.dtype == 'uint16'

def test_imsave():
	arr = np.random.random((32, 32, 32))
	phathom.imageio.imsave('imsave_test.tif', arr)
	tmp = phathom.imageio.imread('imsave_test.tif')
	assert np.all(arr == tmp)
	assert arr.dtype == tmp.dtype
