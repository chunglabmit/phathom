import contextlib
import itertools
import unittest
import multiprocessing
import numpy as np
import os
import shutil
import tempfile
import zarr
from functools import partial
import phathom.utils as utils


def double_elements(arr, start_coord, chunks):
    stop = np.minimum(start_coord + chunks, arr.shape)
    data = utils.extract_box(arr, start_coord, stop)
    return 2 * data


shm = utils.SharedMemory((25, 25), np.float32)


def double_elements_shm(arr, start_coord, chunks):
    stop = np.minimum(start_coord + chunks, arr.shape)
    with shm.txn() as arr:
        data = utils.extract_box(arr, start_coord, stop)
    return 2 * data


class TestPmapChunks(unittest.TestCase):

    def setUp(self):
        self.z = zarr.zeros((25, 25), chunks=(5, 5), dtype=np.float32)
        self.z[:] = np.arange(25**2).reshape((25, 25))
        self.f_zarr = double_elements

        self.shm = shm
        with shm.txn() as arr:
            arr[:] = self.z[:]
        self.f_shm = double_elements_shm

    def test_zarr(self):
        results = utils.pmap_chunks(self.f_zarr, self.z, nb_workers=2)
        self.assertEqual(np.asarray(results).sum(), 2 * self.z[:].sum())
        # using other shape chunks should still work
        results = utils.pmap_chunks(self.f_zarr, self.z, chunks=self.z.shape, nb_workers=2)
        self.assertEqual(np.asarray(results).sum(), 2 * self.z[:].sum())

    def test_shm(self):
        results = utils.pmap_chunks(self.f_shm, self.z, nb_workers=2)  # one chunk should still work
        self.assertEqual(np.asarray(results).sum(), 2 * self.z[:].sum())
        results = utils.pmap_chunks(self.f_shm, self.z, chunks=(5, 5), nb_workers=2)
        self.assertEqual(np.asarray(results).sum(), 2 * self.z[:].sum())


@contextlib.contextmanager
def make_new_zarr(shape, chunks, dtype):
    path = tempfile.mktemp(".zarr")
    z = zarr.open(path, mode="w", shape=shape, chunks=chunks, dtype=dtype)
    yield z
    shutil.rmtree(path)


class TestMemoryToZarr(unittest.TestCase):

    def test_copy(self):
        with make_new_zarr(shape=(25, 35, 45), chunks=(5, 5, 5),
                           dtype=np.float32) as z:
            shared_memory = utils.SharedMemory((25, 35, 45), np.float32)
            with shared_memory.txn() as mem:
                mem[:] = np.random.RandomState(1234).uniform(size=(25, 35, 45))\
                    .astype(np.float32)
            with multiprocessing.Pool() as pool:
                utils.shared_memory_to_zarr(shared_memory, z, pool, (0, 0, 0))
            with shared_memory.txn() as mem:
                np.testing.assert_array_equal(z, mem)

    def test_copy_offset(self):
        with make_new_zarr(shape=(25, 35, 45), chunks=(5, 5, 5),
                           dtype=np.float32) as z:
            shared_memory = utils.SharedMemory((10, 10, 10), np.float32)
            with shared_memory.txn() as mem:
                mem[:] = np.random.RandomState(1234).uniform(size=(10, 10, 10))\
                    .astype(np.float32)
            with multiprocessing.Pool() as pool:
                utils.shared_memory_to_zarr(shared_memory, z, pool, (5, 10, 15))
            with shared_memory.txn() as mem:
                np.testing.assert_array_equal(z[5:15, 10:20, 15:25], mem)


try:
    from blockfs.directory import Directory

    @contextlib.contextmanager
    def make_new_blockfs(shape, chunks, dtype):
        path = tempfile.mkdtemp()
        blockfs_path = os.path.join(path, "test.blockfs")
        directory = Directory(shape[2], shape[1], shape[0],
                              dtype, blockfs_path,
                              x_block_size=chunks[2],
                              y_block_size=chunks[1],
                              z_block_size=chunks[0])
        directory.create()
        directory.start_writer_processes()
        yield directory
        shutil.rmtree(path)

    class TestMemoryToBlockfs(unittest.TestCase):

        def test_copy(self):
            with make_new_blockfs(shape=(25, 35, 45), chunks=(5, 5, 5),
                               dtype=np.float32) as z:
                mem = np.random.RandomState(1234).uniform(size=(25, 35, 45)) \
                    .astype(np.float32)
                utils.memory_to_blockfs(mem, z, (0, 0, 0))
                z.close()
                directory = Directory.open(z.directory_filename)
                for x, y, z in itertools.product(range(0, 45, 5),
                                                 range(0, 35, 5),
                                                 range(0, 25, 5)):
                    np.testing.assert_array_equal(
                        directory.read_block(x, y, z), mem[z:z+5, y:y+5, x:x+5])

        def test_copy_offset(self):
            with make_new_blockfs(shape=(25, 35, 45), chunks=(5, 5, 5),
                               dtype=np.float32) as z:
                mem = np.random.RandomState(1234).uniform(size=(10, 10, 10)) \
                        .astype(np.float32)
                utils.memory_to_blockfs(mem, z, (5, 10, 15))
                z.close()
                directory = Directory.open(z.directory_filename)
                for x, y, z in itertools.product(range(15, 25, 5),
                                                 range(10, 20, 5),
                                                 range(5, 15, 5)):
                    np.testing.assert_array_equal(
                        directory.read_block(x, y, z),
                        mem[z-5:z, y-10:y-5, x-15:x-10])
except:
    pass