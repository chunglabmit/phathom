import argparse
import pickle
import multiprocessing
import numpy as np
import json
import tqdm
import sys
import os

from scipy.spatial import KDTree

from precomputed_tif.client import ArrayReader
from phathom.registration.fine import follow_gradient
from phathom.pipeline.find_neighbors_cmd import FindNeighborsData

def parse_arguments(args=sys.argv[1:]):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fixed-coords",
        help="Path to the blobs found in the fixed volume",
        required=True
    )
    parser.add_argument(
        "--fixed-url",
        help="URL of the fixed neuroglancer volume",
        required=True
    )
    parser.add_argument(
        "--moving-url",
        help="URL of the moving neuroglancer volume",
        required=True
    )
    parser.add_argument(
        "--transform",
        help="Path to the transform .pkl file",
        required=True
    )
    parser.add_argument(
        "--output",
        help="Name of the output file - a JSON dictionary of results",
        required=True
    )
    parser.add_argument(
        "--x-grid",
        help="Dimensions of the point grid in the X direction",
        type=int,
        default=50
    )
    parser.add_argument(
        "--y-grid",
        help="Dimensions of the point grid in the Y direction",
        type=int,
        default=50
    )
    parser.add_argument(
        "--z-grid",
        help="Dimensions of the point grid in the Z direction",
        type=int,
        default=50
    )
    parser.add_argument(
        "--sigma-x",
        help="Smoothing sigma in the X direction",
        type=float,
        default=3.0
    )
    parser.add_argument(
        "--sigma-y",
        help="Smoothing sigma in the Y direction",
        type=float,
        default=3.0
    )
    parser.add_argument(
        "--sigma-z",
        help="Smoothing sigma in the Z direction",
        type=float,
        default=3
    )
    parser.add_argument(
        "--half-window-x",
        help="The half-window size in the x direction. The actual window "
        "will be half-window-x * 2 + 1",
        type=int,
        default=20
    )
    parser.add_argument(
        "--half-window-y",
        help="The half-window size in the y direction.",
        type=int,
        default=20
    )
    parser.add_argument(
        "--half-window-z",
        help="The half-window size in the z direction",
        type=int,
        default=20
    )
    parser.add_argument(
        "--pad-x",
        help="Amount of padding to add to moving array in the x direction "
        "when fetching.",
        type=int,
        default=20
    )
    parser.add_argument(
        "--pad-y",
        help="Amount of padding to add to moving array in the y direction "
        "when fetching.",
        type=int,
        default=20
    )
    parser.add_argument(
        "--pad-z",
        help="Amount of padding to add to moving array in the z direction "
        "when fetching.",
        type=int,
        default=20
    )
    parser.add_argument(
        "--radius",
        help="Points are excluded from the grid if the nearest is more than "
        "the radius from the target grid point. Radius is in microns.",
        type=float,
        default=25
    )
    parser.add_argument(
        "--max-rounds",
        help="Maximum number of steps to take when tracking gradient",
        type=int,
        default=100
    )
    parser.add_argument(
        "--voxel-size",
        help="The size of a voxel in microns, three comma-separated values "
        "in x, y, z order e.g. \"1.8,1.8,2.0",
        default="1.8,1.8,2.0")
    parser.add_argument(
        "--n-cores",
        help="Number of processors to use",
        type=int,
        default=os.cpu_count()
    )
    parser.add_argument(
        "--min-correlation",
        help="Discard matches if correlation is less than this",
        type=float,
        default=.90
    )
    return parser.parse_args(args)


def choose_points(points_fixed, x_grid, y_grid, z_grid, shape, radius,
                  voxel_size, n_cores):
    xs = np.linspace(0, shape[2] * voxel_size[2], x_grid)
    ys = np.linspace(0, shape[1] * voxel_size[1], y_grid)
    zs = np.linspace(0, shape[0] * voxel_size[0], z_grid)
    grid_z, grid_y, grid_x = np.meshgrid(zs, ys, xs)
    grid = np.column_stack(
        [grid_z.flatten(), grid_y.flatten(), grid_x.flatten()])

    kdtree = KDTree(points_fixed * np.array([voxel_size]))
    nearest_d, nearest_idx = np.array(kdtree.query(grid)).transpose()
    mask = nearest_d <= radius
    unique_idx = np.unique(nearest_idx[mask])
    return points_fixed[nearest_idx]



def main(args=sys.argv[1:]):
    opts = parse_arguments(args)
    with open(opts.transform, "rb") as fd:
        interpolator = pickle.load(fd)["interpolator"]
    with open(opts.fixed_coords) as fd:
        points_fixed = np.array(json.load(fd))[:, ::-1]
    afixed = ArrayReader(opts.fixed_url, format="blockfs")
    amoving = ArrayReader(opts.moving_url, format="blockfs")
    voxel_size = \
        np.array([float(_) for _ in opts.voxel_size.split(",")][::-1])

    chosen_points = choose_points(points_fixed,
                                  opts.x_grid,
                                  opts.y_grid,
                                  opts.z_grid,
                                  afixed.shape,
                                  opts.radius,
                                  voxel_size,
                                  opts.n_cores)
    with multiprocessing.Pool(opts.n_cores) as pool:
        futures = []
        for pt_fixed in chosen_points:
            args = (afixed, amoving, interpolator, pt_fixed,
                    opts.half_window_x, opts.half_window_y, opts.half_window_z,
                    opts.pad_x, opts.pad_y, opts.pad_z,
                    opts.max_rounds,
                    [opts.sigma_z, opts.sigma_y, opts.sigma_x])
            futures.append(
                (pt_fixed,
                 pool.apply_async(follow_gradient, args))
            )
        matches = []
        for pt_fixed, future in tqdm.tqdm(futures):
            pt_moving, corr = future.get()
            if corr >= opts.min_correlation:
                matches.append((pt_fixed, pt_moving))
    fixed_coords = np.stack([pt_fixed for pt_fixed, pt_moving in matches])
    moving_coords = np.stack([pt_moving for pt_fixed, pt_moving in matches])
    idx = np.arange(len(fixed_coords))
    fnd = FindNeighborsData(
        fixed_coords,
        moving_coords,
        np.zeros_like(idx, dtype=np.float32),
        np.zeros_like(idx, dtype=np.float32),
        voxel_size,
        idx, idx, 0, 0, 0, None
    )
    fnd.write(opts.output)


if __name__ == "__main__":
    main()