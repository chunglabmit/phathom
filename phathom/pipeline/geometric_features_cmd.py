import argparse
import json
import numpy as np
import os
import sys

from phathom.registration.pcloud import permuted_geometric_features


def parse_args(args=sys.argv[1:]):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        help="Path to a json file of blob coordinates, e.g. as output by "
        "detect-blobs.",
        required=True)
    parser.add_argument(
        "--output",
        help="The path to the features file to be written.",
        required=True)
    parser.add_argument(
        "--voxel-size",
        help="The size of a voxel in microns, three comma-separated values "
        "in x, y, z order e.g. \"1.8,1.8,2.0",
        default="1.8,1.8,2.0")
    parser.add_argument(
        "--n-neighbors",
        help="The number of neighbors to consider. All combinations of 3 among"
        " the neighbors will be generated. The minimum number is 3.",
        default=3,
        type=int
    )
    parser.add_argument(
        "--n-workers",
        help="# of worker processes to use",
        default = os.cpu_count(),
        type=int
    )
    return parser.parse_args(args)


def main(args=sys.argv[1:]):
    opts = parse_args(args)
    coords = np.array(json.load(open(opts.input)))[:, ::-1]
    try:
        voxel_size = np.array(
            [[float(_) for _ in opts.voxel_size.split(",")]])
    except ValueError:
        print(("--voxel-size of %s must be in the format, "
               "\"nnn.nnn,nnn.nnn,nnn.nnn\"") % opts.voxel_size)
        raise
    micron_coords = coords * voxel_size
    features = permuted_geometric_features(micron_coords, opts.n_workers,
                                           opts.n_neighbors)
    np.save(opts.output, features)


if __name__=="__main__":
    main()