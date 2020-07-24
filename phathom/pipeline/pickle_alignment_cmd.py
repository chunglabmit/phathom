import pickle
from functools import partial

import argparse
import json
import numpy as np
import sys
from nuggt.utils.warp import Warper
from phathom.registration.registration import warp_regular_grid, fit_map_interpolator, interpolator

DESCRIPTION="""pickle-alignment is a program that creates a pickled 
alignment function from a points file from nuggt-align. This alignment
file can then be used in commands such as phathom-warp-image or
phathom-non-rigid-registration.
"""

def parse_args(args=sys.argv[1:]):
    parser = argparse.ArgumentParser(
        description=DESCRIPTION)
    parser.add_argument(
        "--input",
        help="The points file, e.g. from nuggt-align",
        required=True)
    parser.add_argument(
        "--output",
        help="The pickle function that is output",
        required=True)
    parser.add_argument(
        "--invert",
        help="Create a mapping from fixed to moving instead of moving "
             "to fixed. Specify this flag if using the function in "
             "phathom-warp-image.",
        action="store_true"
    )
    parser.add_argument(
        "--image-size",
        help="The 3D size of the image volume. The format is x,y,z. "
        "For example, \"2048,2048,1800\" for a z-stack of 1800 images "
        "of size, 2048 by 2048"
    )
    parser.add_argument(
        "--grid-size",
        help="Number of elements in the grid in each direction",
        type=int,
        default=25
    )
    return parser.parse_args(args)


def main(args=sys.argv[1:]):
    opts = parse_args(args)
    with open(opts.input, "r") as fd:
        d = json.load(fd)
    moving_points = np.array(d["moving"])
    fixed_points = np.array(d["reference"])
    if opts.invert:
        fixed_points, moving_points = moving_points, fixed_points
    warper = Warper(moving_points, fixed_points)
    if opts.image_size is None:
        image_size = np.max(fixed_points, 0)
    else:
        image_size = [int(_) for _ in reversed(opts.image_size.split(","))]
    xs = np.linspace(0, image_size[2], opts.grid_size)
    ys = np.linspace(0, image_size[1], opts.grid_size)
    zs = np.linspace(0, image_size[0], opts.grid_size)
    grid_values = warp_regular_grid(opts.grid_size, zs, ys, xs, warper)
    map_interp = fit_map_interpolator(
        grid_values, image_size, order=1)
    map_interpolator = partial(interpolator,
                                interp=map_interp)

    with open(opts.output, "wb") as fd:
        pickle.dump(
            dict(interpolator=map_interpolator,
                 grid_values=grid_values,
                grid_shape=image_size), fd)


if __name__=="__main__":
    main()