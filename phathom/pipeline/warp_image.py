import argparse
import multiprocessing
import numpy as np
import os

import tifffile

from blockfs.directory import Directory
from precomputed_tif.blockfs_stack import BlockfsStack
from precomputed_tif.client import read_chunk, get_info, ArrayReader
from phathom.registration.registration import chunk_coordinates
from phathom.registration.torch_reg import register
from phathom.utils import pickle_load
from scipy.ndimage import map_coordinates
import sys
import tqdm

INPUT_URLS = []
INPUT_FORMATS = []
INPUT_SHAPES = []
OUTPUT_STACKS = []
OUTPUT_BLOCKFS_DIRS = []
GRID_Z, GRID_Y, GRID_X = np.mgrid[0:64, 0:64, 0:64]
INTERPOLATOR = None
GRID_VALUES = None


def parse_args(args=sys.argv[1:]):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--interpolator",
        help="The interpolator output by phathom-fit-nonrigid-transform",
        required=True)
    parser.add_argument(
        "--url",
        help="The neuroglancer URL of the moving image. May be specified "
             "multiple times.",
        action="append")
    parser.add_argument(
        "--input-file",
        help="A 3D tif file as an alternative image input to Neuroglancer"
    )
    parser.add_argument(
        "--url-format",
        help="The format of the URL if a file URL. Must be specified once "
             "per URL if specified at all. Valid values are \"tiff\", \"zarr\" "
             "and \"blockfs\". Default is blockfs",
        action="append")
    parser.add_argument(
        "--output",
        help="The location for the Neuroglancer data source for the warped "
             "image. Must be specified once per input URL.",
        action="append")
    parser.add_argument(
        "--n-workers",
        help="The number of workers devoted to transforming coordinates",
        default=os.cpu_count())
    parser.add_argument(
        "--n-writers",
        help="The number of worker processes devoted to writing output data",
        type=int,
        default=min(12, os.cpu_count())
    )
    parser.add_argument(
        "--n-levels",
        help="The number of levels in each output volume",
        type=int,
        default=7)
    parser.add_argument(
        "--output-shape",
        help="Output volume shape in x,y,z format. If not specified, it "
             "will be the same as the shape of the first input volume."
    )
    parser.add_argument(
        "--silent",
        help="Do not print progress bars",
        action="store_true"
    )
    parser.add_argument(
        "--use-gpu",
        help="Use a GPU to perform the warping computation",
        action="store_true"
    )
    return parser.parse_args(args)


def write_level_1(opts):
    xe = ye = ze = 0
    for stack in OUTPUT_STACKS:
        xe = max(xe, stack.x_extent)
        ye = max(ye, stack.y_extent)
        ze = max(xe, stack.z_extent)
    starts = chunk_coordinates((ze, ye, xe), (64, 64, 64))
    with multiprocessing.Pool(opts.n_workers) as pool:
        futures = []
        for start in starts:
            futures.append(pool.apply_async(do_chunk, (start,)))
        for future in tqdm.tqdm(futures, disable=opts.silent):
            future.get()


#
# Register reads the fixed image to determine whether it is partially
# in-bounds. This returns all zeros if the key is out of bounds and
# returns a nonzero value otherwise
#
class MockFixedImg:
    def __init__(self, input_shape):
        self.input_shape = input_shape

    def __getitem__(self, key):
        for i in range(3):
            if key[i].start >= self.input_shape[i]:
                break
            if key[i].stop <= 0:
                break
        else:
            return np.array([1], np.uint8)
        return np.array([0], np.uint8)


def write_level_1_gpu(opts):
    for input_url, input_format, input_shape, output_dir in zip(
            INPUT_URLS, INPUT_FORMATS, INPUT_SHAPES, OUTPUT_BLOCKFS_DIRS):

        ardr = ArrayReader(input_url, input_format)


        chunks = (output_dir.z_block_size,
                  output_dir.y_block_size,
                  output_dir.x_block_size)
        grid_values = np.array(GRID_VALUES).reshape(
            3, 1, 1, *GRID_VALUES[0].shape)
        register(ardr, MockFixedImg(GRID_SHAPE), output_dir,
                 grid_values,
                 chunks, opts.n_workers)


def do_chunk(start):
    grid_in_x = GRID_X + start[2]
    grid_in_y = GRID_Y + start[1]
    grid_in_z = GRID_Z + start[0]
    fgrid_out_z, fgrid_out_y, fgrid_out_x = INTERPOLATOR(
        np.column_stack(
            [_.flatten() for _ in (grid_in_z, grid_in_y, grid_in_x)])) \
        .transpose()
    grid_out_z = fgrid_out_z.reshape(*grid_in_z.shape)
    grid_out_y = fgrid_out_y.reshape(*grid_in_y.shape)
    grid_out_x = fgrid_out_x.reshape(*grid_in_x.shape)
    for input_url, input_format, input_shape, output_dir in zip(
            INPUT_URLS, INPUT_FORMATS, INPUT_SHAPES, OUTPUT_BLOCKFS_DIRS):
        if start[0] >= output_dir.z_extent or \
                start[1] >= output_dir.y_extent or \
                start[2] >= output_dir.x_extent:
            continue
        shape_out = output_dir.get_block_size(start[2], start[1], start[0])
        goz, goy, gox = [a[:shape_out[0], :shape_out[1], :shape_out[2]]
                         for a in (grid_out_z, grid_out_y, grid_out_x)]
        x_min = max(0, int(np.min(gox)))
        x_max = min(int(np.max(gox)) + 1, input_shape[2])
        y_min = max(0, int(np.min(goy)))
        y_max = min(int(np.max(goy)) + 1, input_shape[1])
        z_min = max(0, int(np.min(goz)))
        z_max = min(int(np.max(goz)) + 1, input_shape[0])
        if x_min >= x_max or \
                y_min >= y_max or \
                z_min >= z_max:
            output_dir.write_block(np.zeros(shape_out, output_dir.dtype),
                                   start[2], start[1], start[0])
            continue
        block = read_chunk(input_url, x_min, x_max, y_min, y_max, z_min, z_max,
                           format=input_format)
        pts = ((
            (goz.flatten() - z_min),
            (goy.flatten() - y_min),
            (gox.flatten() - x_min)))
        block_out = map_coordinates(block, pts).reshape(gox.shape)
        output_dir.write_block(block_out, start[2], start[1], start[0])


def write_level_n(opts, level):
    for stack in OUTPUT_STACKS:
        stack.write_level_n(level, silent=opts.silent, n_cores=opts.n_writers)


def main(args=sys.argv[1:]):
    opts = parse_args(args)
    if opts.input_file is not None:
        do_tiff(opts)
        return
    prepare(opts)
    if opts.use_gpu:
        write_level_1_gpu(opts)
    else:
        write_level_1(opts)
    for directory in OUTPUT_BLOCKFS_DIRS:
        directory.close()
    for level in range(2, opts.n_levels + 1):
        write_level_n(opts, level)


def do_tiff(opts):
    moving_img = tifffile.imread(opts.input_file)
    output_img = np.zeros_like(moving_img)
    d = pickle_load(opts.interpolator)
    grid_values =np.array(d["grid_values"]).reshape(
        3, 1, 1, *d["grid_values"][0].shape)

    img_shape = d["grid_shape"][::-1]
    chunks = 3 * (128, )
    fixed_img = MockFixedImg(img_shape)
    register(moving_img, fixed_img, output_img, grid_values, chunks,
             opts.n_workers)
    tifffile.imsave(opts.output[0], output_img.astype(moving_img.dtype),
                    compress=3)

def prepare(opts):
    global INTERPOLATOR, GRID_VALUES, GRID_SHAPE
    d = pickle_load(opts.interpolator)
    INTERPOLATOR = d["interpolator"]
    GRID_VALUES = d["grid_values"]
    GRID_SHAPE = d["grid_shape"]
    if opts.output_shape is not None:
        output_shape = [int(_) for _ in opts.output_shape.split(",")[::-1]]
    else:
        output_shape = GRID_SHAPE
    for i in range(len(opts.url)):
        INPUT_URLS.append(opts.url[i])
        info = get_info(opts.url[i])
        xe, ye, ze = info.get_scale(1).shape
        INPUT_SHAPES.append((ze, ye, xe))
        if opts.url_format is None or len(opts.url_format) <= i:
            INPUT_FORMATS.append("blockfs")
        else:
            INPUT_FORMATS.append(opts.url_format[i])
        stack = BlockfsStack(output_shape, opts.output[i])
        stack.write_info_file(opts.n_levels)
        OUTPUT_STACKS.append(stack)
        level_1_path = os.path.join(opts.output[i], "1_1_1",
                                    BlockfsStack.DIRECTORY_FILENAME)
        for dirname in (os.path.dirname(os.path.dirname(level_1_path)),
                        os.path.dirname(level_1_path)):
            if not os.path.isdir(dirname):
                os.mkdir(dirname)
        directory = Directory(output_shape[2], output_shape[1], output_shape[0],
                              info.data_type, level_1_path,
                              n_filenames=opts.n_writers)
        directory.create()
        directory.start_writer_processes()
        OUTPUT_BLOCKFS_DIRS.append(directory)


if __name__ == "__main__":
    main()
