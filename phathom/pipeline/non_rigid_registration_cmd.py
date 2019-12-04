import argparse
import functools
import itertools
import json
import numpy as np

from phathom.registration.pcloud import rotation_matrix
from precomputed_tif.client import get_info, read_chunk
import os
from phathom.registration import registration as reg
from phathom.registration.coarse import rigid_transformation, rigid_warp
import pickle
import tempfile
import tifffile
import shutil
import subprocess
import sys


def parse_args(args=sys.argv[1:]):
    parser = argparse.ArgumentParser(
        description="Rough non-rigid registration using Elastix and sitk-align"
    )
    parser.add_argument(
        "--fixed-url",
        help="Neuroglancer URL of the fixed volume",
        required=True)
    parser.add_argument(
        "--fixed-url-format",
        help="The data format of the fixed URL if it is a file URL. "
        "Valid values are \"blockfs\", \"tiff\" or \"zarr\".",
        default="blockfs")

    parser.add_argument(
        "--moving-url",
        help="Neuroglancer URL of the moving volume",
        required=True
    )
    parser.add_argument(
        "--moving-url-format",
        help="The data format of the moving URL if it is a file URL. "
        "Valid values are \"blockfs\", \"tiff\" or \"zarr\".",
        default="blockfs")
    parser.add_argument(
        "--output",
        help="The pickle file holding the interpolator",
        required=True)
    parser.add_argument(
        "--mipmap-level",
        help="The mipmap level of the downsampling, e.g. 32 or 64",
        default=16,
        type=int)

    parser.add_argument(
        "--grid-points",
        help="The number of grid points across the image in each direction",
        default=25,
        type=int
    )
    parser.add_argument(
        "--initial-rotation",
        help="The initial rotation of the moving image along the X, Y and Z "
        "axes as 3 comma-separated values in degrees.",
        default="0,0,0"
    )
    parser.add_argument(
        "--rotation-center",
        help="The rotation center from rigid-rotate, the X,Y,Z values "
        "separated by commas. Default is the image center."
    )
    parser.add_argument(
        "--initial-translation",
        help="The initial translation of the moving image along the X, Y and Z "
        "axes as 3 comma-separated values. If first is negative, specify as "
        "--initial-translation=-xxx,yyy,zzz",
        default="0,0,0"
    )
    return parser.parse_args(args)


def get_rotate_parameters(opts, shape):
    theta_x, theta_y, theta_z = [
        float(_) / 180 * np.pi for _ in opts.initial_rotation.split(",")
    ]
    theta = np.array([theta_z, theta_y, theta_x])
    if opts.rotation_center is None:
        center_z, center_y, center_x = shape / 2
    else:
        center_x, center_y, center_z = \
            [float(_) for _ in opts.rotation_center.split(",")]
    center = np.array([center_z, center_y, center_x])
    offset_x, offset_y, offset_z = \
        [-float(_) for _ in opts.initial_translation.split(",")]
    offset = np.array([offset_z, offset_y, offset_x])
    #
    # t = translation
    # s = scale
    #
    return dict(t=offset, thetas=theta, center=center, s=1), \
           dict(t=offset / opts.mipmap_level,
                thetas=theta,
                center=center / opts.mipmap_level,
                s=1)


def main(args=sys.argv[1:]):
    opts = parse_args(args)
    fixed_scale = get_info(opts.fixed_url).get_scale(opts.mipmap_level)
    fixed_shape = np.array(fixed_scale.shape[::-1])
    moving_scale = get_info(opts.moving_url).get_scale(opts.mipmap_level)
    moving_shape = np.array(moving_scale.shape[::-1])
    full_rotate_params, rotate_params =\
        get_rotate_parameters(opts, moving_shape * opts.mipmap_level)
    tempdir = tempfile.mkdtemp()
    try:
        fixed_img = read_chunk(opts.fixed_url,
                               0, fixed_shape[2],
                               0, fixed_shape[1],
                               0, fixed_shape[0],
                               level = opts.mipmap_level,
                               format = opts.fixed_url_format)
        fixed_img = rigid_warp(fixed_img,
                                output_shape=fixed_shape,
                                **rotate_params)
        fixed_path = os.path.join(tempdir, "fixed.tiff")
        moving_path = os.path.join(tempdir, "moving.tiff")
        moving_points_path = os.path.join(tempdir, "moving.json")
        alignment_path = os.path.join(tempdir, "alignment.json")
        tifffile.imsave(fixed_path, fixed_img)
        del fixed_img
        moving_img = read_chunk(opts.moving_url,
                                0, moving_shape[2],
                                0, moving_shape[1],
                                0, moving_shape[0],
                                level = opts.mipmap_level,
                                format = opts.moving_url_format)
        tifffile.imsave(moving_path, moving_img)
        del moving_img
        moving_points = []
        nb_pts = opts.grid_points
        zs = np.linspace(0, moving_shape[0], nb_pts)
        ys = np.linspace(0, moving_shape[1], nb_pts)
        xs = np.linspace(0, moving_shape[2], nb_pts)
        for z, y, x in itertools.product(zs, ys, xs):
            moving_points.append((x, y, z))
        with open(moving_points_path, "w") as fd:
            json.dump(moving_points, fd, indent=2)
        #
        # The transformation we want has the opposite polarity as sitk-align
        # so fixed becomes moving and moving, fixed
        #
        subprocess.check_call(
            [
                "sitk-align",
                "--fixed-file",
                moving_path,
                "--moving-file",
                fixed_path,
                "--fixed-point-file",
                moving_points_path,
                "--xyz",
                "--transform-parameters-folder",
                tempdir,
                "--alignment-point-file",
                alignment_path
            ]
        )
        with open(alignment_path, "r") as fd:
            d = json.load(fd)
        xs, ys, zs = [_ * opts.mipmap_level for _ in (xs, ys, zs)]
        moving_points = \
            np.array(d["reference"]).reshape((nb_pts, nb_pts, nb_pts, 3)) * \
            opts.mipmap_level
        fixed_points = np.array(d["moving"]) * opts.mipmap_level
        r = rotation_matrix(full_rotate_params["thetas"])
        fixed_points = rigid_transformation(
            full_rotate_params["t"],
            r,
            fixed_points,
            full_rotate_params["center"])
        fixed_points = fixed_points.reshape((nb_pts, nb_pts, nb_pts, 3))
        grid_values = fixed_points.transpose(3, 0, 1, 2)
        moving_shape_full = np.asanyarray(moving_shape) *  opts.mipmap_level
        map_interp = reg.fit_map_interpolator(
            grid_values, moving_shape_full, order=1)
        map_interpolator = \
            functools.partial(reg.interpolator, interp=map_interp)
        output = dict(
            nb_pts = nb_pts,
            fixed_points=fixed_points,
            moving_points=moving_points,
            interpolator=map_interpolator)
        with open(opts.output, "wb") as fd:
            pickle.dump(output, fd)
    finally:
        shutil.rmtree(tempdir)


if __name__=="__main__":
    main()