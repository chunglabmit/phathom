import unittest
import os
import numpy as np
from phathom import io
from phathom import synthetic
from phathom.segmentation import ventricle

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm

from phathom.plotting import plot_mip, zprojection
from skimage.viewer import CollectionViewer
import tifffile


seed = 295

try:
    working_dir = '/media/jswaney/Drive/Justin/organoid_etango/test_ventricle'
    syto16_img = io.tiff.imread(os.path.join(working_dir, 'syto16_slice.tif'))
    test_images = True
except FileNotFoundError:
    test_images = False

class TestMGAC(unittest.TestCase):

    @unittest.skipIf(not test_images)
    def test(self):
        center = (2200, 2100)
        niter = 10
        mask = ventricle.ventricle_2d(syto16_img, center, niter)

        plt.imshow(syto16_img, clim=[0, 4095])
        plt.imshow(mask, alpha=0.5)
        plt.show()