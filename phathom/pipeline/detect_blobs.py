import argparse
import json
import multiprocessing
import numpy as np
import tqdm
from precomputed_tif.client import ArrayReader
import sys

from phathom.segmentation.segmentation import detect_blobs
from phathom.utils import chunk_coordinates


def parse_args(args=sys.argv[1:]):
    parser = argparse.ArgumentParser()
    parser.add_argument("--url",
                        help="Neuroglancer URL of the datasource",
                        required=True)
    parser.add_argument("--url-format",
                        help="Format of the URL. Defaults to blockfs",
                        default="blockfs")
    parser.add_argument("--output",
                        help="Path to the output .json file",
                        required=True)
    parser.add_argument("--voxel-size",
                        help="Voxel size in microns in x,y,z format. "
                        "Default is 1.8,1.8,2.0",
                        default="1.8,1.8,2.0")
    parser.add_argument("--sigma-low",
                        type=float,
                        help="Foreground sigma in microns",
                        default=3.0)
    parser.add_argument("--sigma-high",
                        type=float,
                        help="Background sigma in microns",
                        default=30.0)
    parser.add_argument("--pad-x",
                        type=int,
                        help="Array padding in the x direction",
                        default=10)
    parser.add_argument("--pad-y",
                        type=int,
                        help="Array padding in the y direction",
                        default=10)
    parser.add_argument("--pad-z",
                        type=int,
                        help="Array padding in the z direction",
                        default=10)
    parser.add_argument("--min-distance",
                        type=float,
                        help="Minimum distance between maxima in microns",
                        default=20)
    parser.add_argument("--threshold",
                        type=float,
                        help="Cutoff threshold for maxima intensity",
                        default=10)
    parser.add_argument("--block-size-x",
                        type=int,
                        help="Size of a processing block in the x direction",
                        default=256)
    parser.add_argument("--block-size-y",
                        type=int,
                        help="Size of a processing block in the y direction",
                        default=256)
    parser.add_argument("--block-size-z",
                        type=int,
                        help="Size of a processing block in the z direction",
                        default=256)
    parser.add_argument("--n-workers",
                        type=int,
                        help="# of processes to use when parallelizing",
                        default=multiprocessing.cpu_count())
    return parser.parse_args(args)

def main(args=sys.argv[1:]):
    opts = parse_args(args)
    ar = ArrayReader(opts.url, format=opts.url_format)
    block_size = (opts.block_size_z, opts.block_size_y, opts.block_size_x)
    start_coords = chunk_coordinates(
        ar.shape,
        block_size)
    max_coords = np.array(ar.shape).reshape(1, 3)
    block_size = np.array(block_size).reshape(1, 3)
    end_coords = np.minimum(max_coords, start_coords + block_size)
    voxel_size = \
        np.array([float(_) for _ in reversed(opts.voxel_size.split(","))])
    pad = (opts.pad_z, opts.pad_y, opts.pad_x)
    sigma_low = opts.sigma_low / voxel_size
    sigma_high = opts.sigma_high / voxel_size
    min_distance = opts.min_distance / voxel_size
    futures = []
    with multiprocessing.Pool(opts.n_workers) as pool:
        for start, end in zip(start_coords, end_coords):
            futures.append(pool.apply_async(
                detect_blobs,
                (ar, start, end, pad, sigma_low, sigma_high, opts.threshold,
                 min_distance)))
        coords = []
        for future in tqdm.tqdm(futures):
            coords.append(future.get())
    coords = np.concatenate(coords, 0).tolist()
    with open(opts.output, "w") as fd:
        json.dump(coords, fd)


if __name__ == "__main__":
    main()
