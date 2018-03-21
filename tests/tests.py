import phathom
import phathom.conversion
import phathom.utils
import numpy as np
import unittest
import os
import tempfile


class TestConversion(unittest.TestCase):
    """
    Unit tests for conversion module
    """
    def test_imread(self):
        """
        Load example.tif using imread
        """
        filename = os.path.join(os.path.split(__file__)[0], 'example.tif')
        data = phathom.conversion.imread(filename)
        self.assertEqual(data.shape, (64, 128, 128), msg='loaded array has the wrong shape')
        self.assertEqual(data.dtype, 'uint16', msg='loaded array has the wrong data type')

    def test_imsave(self):
        """
        Save and read a random tif and check for consistency
        """
        arr = np.random.random((32, 32, 32))
        filename = os.path.join(tempfile.gettempdir(), "imsave_test.tif")
        phathom.conversion.imsave(filename, arr)
        tmp = phathom.conversion.imread(filename)
        self.assertTrue(np.all(arr == tmp), msg='saved and loaded array values are not equal')
        self.assertEqual(arr.dtype, tmp.dtype, msg='saved and loaded array do not have same data type')


class TestUtils(unittest.TestCase):
    """
    Unit tests for utils module
    """
    def test_make_dir(self):
        """
        Create a directory using make_dir
        """
        test_dir = 'tests/make_dir_test/'
        if os.path.isdir(test_dir):
            os.rmdir(test_dir)
        phathom.utils.make_dir(test_dir)
        self.assertTrue(os.path.isdir(test_dir), msg='test_dir does not exist after running make_dir')
        if os.path.isdir(test_dir): # cleanup
            os.rmdir(test_dir)

    def test_files_in_dir(self):
        """
        Scan for all files within tests/file_tests/
        """
        file_test_dir = os.path.join(os.path.dirname(__file__),
                                     'file_tests')
        expected_files = ['file1.txt', 'file2.tif', 'file3.tif']
        found_files = phathom.utils.files_in_dir(file_test_dir)
        self.assertEqual(found_files, expected_files, msg='found incorrect files')

    def test_tifs_in_dir(self):
        """
        Scan for all tifs within tests/file_tests/
        """
        file_test_dir = os.path.join(os.path.dirname(__file__),
                                     'file_tests')
        expected_files = ['file2.tif', 'file3.tif']

        abs_path = os.path.abspath(file_test_dir)
        expected_paths = [os.path.join(abs_path, fname) for fname in expected_files]

        found_paths, found_files = phathom.utils.tifs_in_dir(file_test_dir)
        self.assertEqual(found_files, expected_files, msg='found incorrect tif filenames')
        self.assertEqual(found_paths, expected_paths, msg='found incorrect tif paths')

    def test_pickle_save_load(self):
        """
        Save and load a pickled dictionary
        """
        true_dict = {'chunks': (8, 16, 32), 'shape': (100, 1000, 1000)}
        tmp_file = 'tests/tmp.pkl'
        phathom.utils.pickle_save(tmp_file, true_dict)
        read_dict = phathom.utils.pickle_load(tmp_file)
        self.assertEqual(read_dict, true_dict, msg='saved and read dict do not match')
        os.remove(tmp_file)  # cleanup


class TestSegmentation(unittest.TestCase):
    pass
