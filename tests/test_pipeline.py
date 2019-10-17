import contextlib
import json
import numpy as np
import os
import shutil
import tempfile
import tifffile
import unittest
import zarr
try:
    from blockfs.directory import Directory
    haz_blockfs = True
except ImportError:
    haz_blockfs = False

from phathom.pipeline.preprocess_cmd import main as preprocess_main
from phathom.pipeline.geometric_features_cmd import main as gf_main

@contextlib.contextmanager
def make_stack(shape):
    path = tempfile.mkdtemp()
    outpath = tempfile.mkdtemp()
    stack = np.random.RandomState(1234).randint(0, 65535, size=shape)
    for i, plane in enumerate(stack):
        tifffile.imsave(os.path.join(path, "img_%04d.tiff" % i), plane)
    yield((path, outpath))
    shutil.rmtree(path)
    shutil.rmtree(outpath)


class TestPreprocess(unittest.TestCase):
    def test_tiff(self):
        with make_stack((10, 20, 20)) as (inpath, outpath):
            preprocess_main([
                "--input", inpath,
                "--output", outpath,
                "--output-format", "tiff"
            ])
            output_paths = sorted(
                [os.path.join(outpath, _) for _ in os.listdir(outpath)])
            self.assertEqual(len(output_paths), 10)
            for filename in output_paths:
                self.assertSequenceEqual(tifffile.imread(filename).shape,
                                         (20, 20))

    def test_zarr(self):
        with make_stack((10, 20, 20)) as (inpath, outpath):
            preprocess_main([
                "--input", inpath,
                "--output", outpath,
                "--output-format", "zarr"
            ])
            z = zarr.open(outpath, "r")
            self.assertSequenceEqual(z.shape, (10, 20, 20))

    if haz_blockfs:
        def test_blockfs(self):
            with make_stack((10, 20, 20)) as (inpath, outpath):
                blockfs_path = os.path.join(outpath, "test.blockfs")
                preprocess_main([
                    "--input", inpath,
                    "--output", blockfs_path,
                    "--output-format", "blockfs"
                ])
                d = Directory.open(blockfs_path)
                self.assertSequenceEqual(d.read_block(0, 0, 0).shape,
                                         (10, 20, 20))


@contextlib.contextmanager
def make_points_case(points:np.ndarray):
    infile = tempfile.mktemp(".json")
    outfile = tempfile.mktemp(".npy")
    with open(infile, "w") as fd:
        json.dump(points.tolist(), fd)
    yield infile, outfile
    os.remove(infile)
    if os.path.exists(outfile):
        os.remove(outfile)

class TestGeometricFeatures(unittest.TestCase):

    def test_01_simple(self):
        inpoints = np.random.RandomState(1234).uniform(size=(5, 3))
        with make_points_case(inpoints) as (infile, outfile):
            gf_main([
                "--input", infile,
                "--output", outfile
            ])
            self.assertTrue(os.path.exists(outfile))
            features = np.load(outfile)
            self.assertEqual(len(features), 5)




if __name__ == '__main__':
    unittest.main()
