import argparse
import logging
import matplotlib.backends.backend_pdf
from matplotlib import pyplot
import numpy as np
from precomputed_tif.client import get_info, read_chunk
from functools import partial
from skimage.external import tifffile
from phathom.registration import registration as reg
from phathom.pipeline.filter_matches_cmd import FilterMatchesData
from phathom.plotting import plot_residuals
from phathom.utils import pickle_save
from scipy.ndimage import map_coordinates
import sys


PDF = None


def parse_args(args=sys.argv[1:]):
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",
                        help="Input file from phathom-filter-matches",
                        required=True)
    parser.add_argument("--output",
                        help="Pickled nonrigid transform and grid values",
                        required=True)
    parser.add_argument(
        "--visualization-file",
        help="The path to the PDF file output by this program. "
        "This file contains helpful visualizations that document the "
        "program's progress.")
    parser.add_argument(
        "--interactive",
        help="If supplied, the program will display each of the visualizations "
        "as they are created. Only supply if you have a display.",
        action="store_true")
    parser.add_argument(
        "--show-warping-illustration",
        help="Add a 2d image of the moving image, warped to the fixed image "
        "to the visualization PDF. This takes time and memory to create but "
        "can be useful to determine the quality of the final warping before "
        "warping the entire moving volume.",
        action="store_true")
    parser.add_argument(
        "--max-samples",
        help="The maximum number of samples to be used when constructing "
        "the thin-plate spline",
        type=int,
        default=10000)
    parser.add_argument(
        "--smoothing",
        help="The smoothing for the thin-plate spline",
        type=float,
        default=10)
    parser.add_argument(
        "--grid-points",
        help="The number of grid points in the X, Y and Z when creating a "
        "bspline approximation",
        default=100,
        type=int)
    parser.add_argument(
        "--fixed-url",
        help="The Neuroglancer URL of the fixed volume, e.g. "
        "\"https://my-server.org/precomputed/fixed\".",
        required=True)
    parser.add_argument(
        "--moving-url",
        help="The Neuroglancer URL of the moving volume, e.g. "
             "\"https://my-server.org/precomputed/fixed\".",
        required=True)
    parser.add_argument(
        "--fixed-url-format",
        help="The data format of the fixed URL if it is a file URL. "
        "Valid values are \"blockfs\", \"tiff\" or \"zarr\".",
        default="blockfs")
    parser.add_argument(
        "--moving-url-format",
        help="The data format of the moving URL if it is a file URL. "
        "Valid values are \"blockfs\", \"tiff\" or \"zarr\".",
        default="blockfs")
    parser.add_argument(
        "--log-level",
        help="The log verbosity level. Default is WARNING, options are "
        "DEBUG, INFO, WARNING and ERROR",
        default="WARNING")
    return parser.parse_args(args)


def main(args=sys.argv[1:]):
    global PDF
    opts = parse_args(args)
    log_level = getattr(logging, opts.log_level)
    logging.basicConfig(level=log_level)
    if opts.visualization_file is not None:
        matplotlib.interactive(opts.interactive)
        PDF = matplotlib.backends.backend_pdf.PdfPages(opts.visualization_file)

    input_data = FilterMatchesData.read(opts.input)
    voxel_size = np.array([input_data.voxel_size])
    n_coords = len(input_data.fixed_coords)
    logging.info("%d points in input" % n_coords)

    random = np.random.RandomState(
        input_data.fixed_coords.astype(int).flatten())
    if n_coords <= opts.max_samples:
        fixed_sample = input_data.fixed_coords
        moving_sample = input_data.moving_coords
        affine_sample = input_data.affine_coords
    else:
        sample_idx = random.choice(n_coords, opts.max_samples, replace=False)
        fixed_sample = input_data.fixed_coords[sample_idx]
        moving_sample = input_data.moving_coords[sample_idx]
        affine_sample = input_data.affine_coords[sample_idx]
    affine_sample_vox = affine_sample / voxel_size
    moving_sample_vox = moving_sample / voxel_size
    rbf_z, rbf_y, rbf_x = reg.fit_rbf(affine_sample_vox,
                                      moving_sample_vox,
                                      opts.smoothing)
    tps_transform = partial(reg.rbf_transform,
                            rbf_z=rbf_z,
                            rbf_y=rbf_y,
                            rbf_x=rbf_x)
    nonrigid_keypoints_vox = tps_transform(
        input_data.affine_coords / voxel_size)
    nonrigid_residuals = reg.match_distance(
        nonrigid_keypoints_vox * voxel_size,
        input_data.moving_coords)
    logging.info("Average residual after non-rigid registration: %.1f um" %
                 np.mean(nonrigid_residuals))
    if PDF is not None:
        figure = pyplot.figure(figsize=(6, 6))
        plot_residuals(input_data.moving_coords, nonrigid_residuals)
        figure.suptitle("Non-rigid registration residuals")
        PDF.savefig(figure)
        figure = pyplot.figure(figsize=(6, 3))
        pyplot.hist(nonrigid_residuals, bins=128)
        figure.suptitle("Residuals after non-rigid registration")
        PDF.savefig(figure)
    nonrigid_transform = partial(
        reg.nonrigid_transform,
        affine_transform=input_data.affine_transform_fn,
        rbf_z=rbf_z,
        rbf_y=rbf_y,
        rbf_x=rbf_x)

    fixed_shape = get_info(opts.fixed_url).get_scale(1).shape[::-1]
    moving_shape = get_info(opts.moving_url).get_scale(1).shape[::-1]
    nb_pts = opts.grid_points

    z = np.linspace(0, fixed_shape[0], nb_pts)
    y = np.linspace(0, fixed_shape[1], nb_pts)
    x = np.linspace(0, fixed_shape[2], nb_pts)
    grid_values = reg.warp_regular_grid(nb_pts, z, y, x, nonrigid_transform)
    if PDF is not None:
        figure = pyplot.figure(figsize=(6, 6))
        #
        # Plot the vector displacement at each of the grid points
        #
        half = nb_pts // 2
        figure.add_subplot(2, 2, 1).quiver(
            grid_values[2][half] - x.reshape(1, nb_pts),
            grid_values[1][half] - y.reshape(nb_pts, 1))
        figure.add_subplot(2, 2, 2).quiver(
            grid_values[2][:, half] - x.reshape(1, nb_pts),
            grid_values[0][:, half] - z.reshape(nb_pts, 1))
        figure.add_subplot(2, 2, 4).quiver(
            grid_values[1][:, :, half] - y.reshape(1, nb_pts),
            grid_values[0][:, :, half] - z.reshape(nb_pts, 1))
        figure.suptitle("Warping grid vector displacement")
        PDF.savefig(figure)
    grid_interp = reg.fit_grid_interpolator(z, y, x, grid_values)
    map_interp = reg.fit_map_interpolator(grid_values, fixed_shape, order=1)
    grid_interpolator = partial(reg.interpolator, interp=grid_interp)
    map_interpolator = partial(reg.interpolator, interp=map_interp)
    interp_keypoints_vox = map_interpolator(
        pts=input_data.fixed_coords / voxel_size)
    interp_keypoints = interp_keypoints_vox * voxel_size
    interp_residuals = reg.match_distance(interp_keypoints,
                                          input_data.moving_coords)
    logging.info("Interpolator residual average: %.1f" %
                 np.mean(interp_residuals))
    if PDF is not None:
        figure = pyplot.figure(figsize=(6, 6))
        plot_residuals(input_data.moving_coords, interp_residuals)
        figure.suptitle("Interpolator residuals per point")
        PDF.savefig(figure)
        figure = pyplot.figure(figsize=(6, 3))
        pyplot.hist(interp_residuals, bins=128)
        figure.suptitle("Interpolator residuals")
        PDF.savefig(figure)
    #
    # Do the XY slice
    #
    center = np.array(fixed_shape) // 2
    if opts.show_warping_illustration:
        fxyz, fxyy, fxyx = np.mgrid[center[0]:center[0]+1,
                                    0:fixed_shape[1],
                                    0:fixed_shape[2]]
        fpts_flat = np.column_stack([_.flatten() for _ in (fxyz, fxyy, fxyx)])
        mxyz_flat, mxyy_flat, mxyx_flat = \
            map_interpolator(pts=fpts_flat).transpose()
        mxyz, mxyy, mxyx = [_.reshape(fxyz.shape) for _ in
                            (mxyz_flat, mxyy_flat, mxyx_flat)]
        min_z = max(0, int(np.min(mxyz)))
        min_y = max(0, int(np.min(mxyy)))
        min_x = max(0, int(np.min(mxyx)))
        max_z = min(moving_shape[0], int(np.ceil(np.max(mxyz))))
        max_y = min(moving_shape[1], int(np.ceil(np.max(mxyy))))
        max_x = min(moving_shape[2], int(np.ceil(np.max(mxyx))))
        moving = read_chunk(opts.moving_url,
                            min_x, max_x, min_y, max_y, min_z, max_z,
                            format=opts.moving_url_format)
        moving_img = map_coordinates(
            moving, [mxyz - min_z, mxyy - min_y, mxyx - min_x])[0]
        fixed_img = read_chunk(opts.fixed_url,
                               0, fixed_shape[2], 0, fixed_shape[1],
                               center[0], center[0]+1,
                               format=opts.fixed_url_format)[0]
        if PDF is not None:
            figure = pyplot.figure(figsize=(6, 6))
            cimg = np.column_stack((fixed_img.flatten() / fixed_img.max(),
                                    moving_img.flatten() / moving_img.max(),
                                    np.zeros(np.prod(fixed_img.shape)))) \
            .reshape(fixed_img.shape[0], fixed_img.shape[1], 3)
            pyplot.imshow(cimg)
            PDF.savefig(figure)
    pickle_save(opts.output,
                dict(interpolator=map_interpolator,
                     grid_values=grid_values,
                     grid_shape=fixed_shape))
    if PDF is not None:
        PDF.close()


if __name__ == "__main__":
    main()
