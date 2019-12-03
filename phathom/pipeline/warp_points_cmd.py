import argparse
import json
import multiprocessing
import numpy as np
import os
from functools import partial
from precomputed_tif.client import get_info
from phathom.registration.registration import fit_rbf, rbf_transform
from phathom.utils import pickle_load
import sys
import tqdm


def parse_args(args = sys.argv[1:]):
    parser = argparse.ArgumentParser(
        description="Translate points from the fixed frame of reference "
    "to the moving frame")
    parser.add_argument(
        "--interpolator",
        help="The interpolator output by phathom-fit-nonrigid-transform",
        required=True)
    parser.add_argument(
        "--input",
        help="The .json file of coordinates in the fixed frame of reference",
        required=True)
    parser.add_argument(
        "--output",
        help="The .json file of coordinates as translated to the moving frame"
    )
    parser.add_argument(
        "--n-workers",
        help="Number of worker processes",
        default = os.cpu_count(),
        type=int
    )
    parser.add_argument(
        "--batch-size",
        help="Number of coordinates in a batch",
        default=100000,
        type=int
    )
    return parser.parse_args(args)


INTERPOLATOR = None


def run_interpolator(coords):
    return INTERPOLATOR(coords)


def main(args = sys.argv[1:]):
    global INTERPOLATOR
    opts = parse_args(args)
    pickle = pickle_load(opts.interpolator)
    INTERPOLATOR = pickle["interpolator"]
    with open(opts.input) as fd:
        input_coords = np.array(json.load(fd))[:, ::-1]
    with multiprocessing.Pool(opts.n_workers) as pool:
        output_coords = []
        futures = []
        for idx in range(0, len(input_coords), opts.batch_size):
            idx_end = min(idx + opts.batch_size, len(input_coords))
            futures.append(pool.apply_async(
                run_interpolator, (input_coords[idx:idx_end],)))
        for future in tqdm.tqdm(futures):
            output_coords.append(future.get()[:, ::-1])
    output_coords = np.concatenate(output_coords, 0)
    with open(opts.output, "w")  as fd:
        json.dump(output_coords.tolist(), fd)


if __name__=="__main__":
    main()