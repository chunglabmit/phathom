import argparse
import logging
import json
import matplotlib.backends.backend_pdf
from matplotlib import pyplot
import numpy as np
import os
import sys

from phathom.utils import pickle_load
from phathom.registration.coarse import rigid_transformation
from phathom.registration.pcloud import radius_matching, rotation_matrix


PDF = None


class FindNeighborsData:
    """
The output of phathom-find-neighbors is a JSON dictionary
with the following keys:

fixed-coords: coordinates of blobs in the fixed volume in microns in x, y, z
format
moving-coords: coordinates of blobs in the moving volume in microns
               after correction using the rigid registration (in x, y, z fmt)
fixed_features: a feature vector per blob for the fixed volume
moving_features: a feature vector per blob for the moving volume
voxel-size: size of a voxel in microns in x, y, z format
fixed-idx: index of match pair member in the fixed volume
moving-idx: index of corresponding match pair member in the moving volume
radius: search radius used
max-fdist: maximum allowed feature distance for a match
prom-thresh: prominence threshold used for match
    """
    def __init__(self,
                 fixed_coords,
                 moving_coords,
                 fixed_features,
                 moving_features,
                 voxel_size,
                 fixed_idx,
                 moving_idx,
                 radius,
                 max_fdist,
                 prom_thresh,
                 rigid_transform_params):
        self.fixed_coords = fixed_coords
        self.moving_coords = moving_coords
        self.fixed_features = fixed_features
        self.moving_features = moving_features
        self.voxel_size = voxel_size
        self.fixed_idx = fixed_idx
        self.moving_idx = moving_idx
        self.radius = radius
        self.max_fdist = max_fdist
        self.prom_thresh = prom_thresh
        self.rigid_transform_params = rigid_transform_params

    @staticmethod
    def read(path):
        """
        Read the neighbors data from a JSON file

        :param path: the file to read from
        :return: The data in the file
        """
        with open(path) as fd:
            d = json.load(fd)
            fixed_coords = np.array(d["fixed-coords"])[:, ::-1]
            moving_coords = np.array(d["moving-coords"])[:, ::-1]
            fixed_features = np.array(d["fixed-features"])
            moving_features = np.array(d["moving-features"])
            voxel_size = d["voxel-size"][::-1]
            fixed_idx = np.array(d["fixed-idx"])
            moving_idx = np.array(d["moving-idx"])
            radius = d["radius"]
            max_fdist = d["max-fdist"]
            prom_thresh = d["prom-thresh"]
            rigid_transform_params = d["rigid-transform-params"]
            return FindNeighborsData(
                fixed_coords=fixed_coords,
                moving_coords=moving_coords,
                fixed_features=fixed_features,
                moving_features=moving_features,
                voxel_size=voxel_size,
                fixed_idx=fixed_idx,
                moving_idx=moving_idx,
                radius=radius,
                max_fdist=max_fdist,
                prom_thresh=prom_thresh,
                rigid_transform_params=rigid_transform_params)

    def write(self, path):
        """
        Write the data out to a JSON file

        :param path: the path to the file.
        """
        d = {
            "fixed-coords": self.fixed_coords[:, ::-1].astype(float).tolist(),
            "moving-coords": self.moving_coords[:, ::-1].astype(float).tolist(),
            "fixed-features": self.fixed_features.astype(float).tolist(),
            "moving-features": self.moving_features.astype(float).tolist(),
            "voxel-size": self.voxel_size[::-1],
            "fixed-idx": self.fixed_idx.astype(int).tolist(),
            "moving-idx": self.moving_idx.astype(int).tolist(),
            "radius": self.radius,
            "max-fdist": self.max_fdist,
            "prom-thresh": self.prom_thresh,
            "rigid-transform-params": self.rigid_transform_params
        }
        with open(path, "w") as fd:
            json.dump(d, fd)


def parse_args(args=sys.argv[1:]):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fixed-coords",
        help="The path to the blob coordinate file for the fixed volume. "
        "This is a .json file in x, y, z format e.g. as made by detect-blobs.",
        required=True)
    parser.add_argument(
        "--moving-coords",
        help="The path to the blob coordinate file for the moving volume. "
        "This is a .json file in x, y, z format e.g. as made by detect-blobs.",
        required=True)
    parser.add_argument(
        "--fixed-features",
        help="The path to the features file for the fixed volume, e.g. "
        "as produced by phathom-geometric-features",
        required=True)
    parser.add_argument(
        "--moving-features",
        help="The path to the features file for the moving volume, e.g. "
        "as produced by phathom-geometric-features",
        required=True)
    parser.add_argument(
        "--rigid-transformation",
        help="The rigid transformation to convert the moving coordinates "
        "to an approximation of the fixed volume space, e.g. as produced "
        "by phathom-rigid-registration.",
        required=True)
    parser.add_argument(
        "--output",
        help="The output of this program, a JSON dictionary with the "
        "intermediate results to this stage.",
        required=True)
    parser.add_argument(
        "--voxel-size",
        help="The size of a voxel in microns, three comma-separated values "
        "in x, y, z order e.g. \"1.8,1.8,2.0",
        default="1.8,1.8,2.0")
    parser.add_argument(
        "--radius",
        help="The search radius for matches, in microns",
        default=150.0,
        type=float)
    parser.add_argument(
        "--max-fdist",
        help="The maximum allowed feature distance for a match",
        default=2.0,
        type=float)
    parser.add_argument(
        "--prom-thresh",
        help="The prominence threshold for a match. All competing matches must "
        "have a feature distance that is less than this fraction of the "
        "match being considered",
        default=0.3,
        type=float)
    parser.add_argument(
        "--n-workers",
        help="The number of worker processes to use during computation",
        default=os.cpu_count(),
        type=int)
    parser.add_argument(
        "--batch-size",
        help="The number of fixed points to process per worker invocation",
        default=500000,
        type=int)
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
        "--log-level",
        help="The log verbosity level. Default is WARNING, options are "
        "DEBUG, INFO, WARNING and ERROR",
        default="WARNING")
    return parser.parse_args(args)


def plot_points(fixed_coords, moving_coords, center):
    figure = pyplot.figure(figsize=(6, 6))
    ax = figure.add_subplot(2, 2, 1)
    plot_axis(ax, center, fixed_coords, moving_coords, 0)
    ax = figure.add_subplot(2, 2, 2)
    plot_axis(ax, center, fixed_coords, moving_coords, 1)
    ax = figure.add_subplot(2, 2, 4)
    plot_axis(ax, center, fixed_coords, moving_coords, 2)
    return figure


def cull_pts(coords, center, axis, count):
    """
    Return the "count" closest points to the center in the given dimension

    :param coords: coordinates to cull
    :param center: the center of the volume
    :param axis: the axis that is orthogonal to the plane being displayed
    :param count: the maximum number of coordinates to return
    :return: a count x 2 array of the coordinates to display
    """
    idx = np.argsort(np.abs(center[axis] - coords[:, axis]))
    if len(idx) > count:
        coords = coords[idx[:count]]
    return np.column_stack([coords[:, i] for i in range(3) if i != axis])


def plot_axis(ax, center, fixed_coords, moving_coords, axis):
    f = cull_pts(fixed_coords, center, axis, 2000)
    m = cull_pts(moving_coords, center, axis, 2000)
    ax.plot(f[:, 1], f[:, 0], "ro", label="fixed", markersize=1)
    ax.plot(m[:, 1], m[:, 0], "go", label="moving", markersize=1)


def main(args=sys.argv[1:]):
    global PDF
    opts = parse_args(args)
    matplotlib.interactive(opts.interactive)
    log_level = getattr(logging, opts.log_level)
    logging.basicConfig(level=log_level)
    if opts.visualization_file is not None:
        matplotlib.interactive(opts.interactive)
        PDF = matplotlib.backends.backend_pdf.PdfPages(opts.visualization_file)
    try:
        voxel_size = \
            np.array([[float(_) for _ in opts.voxel_size.split(",")]])[::-1]
    except ValueError:
        print("--voxel-size=%s must be in nnn.nnn,nnn.nnn,nnn.nnn format" %
              opts.voxel_size)
        raise
    logging.info("Opening fixed coordinate file %s" % opts.fixed_coords)
    with open(opts.fixed_coords) as fd:
        fixed_coords = np.array(json.load(fd))[:, ::-1]
    logging.info("Opening moving coordinate file %s" % opts.moving_coords)
    with open(opts.moving_coords) as fd:
        moving_coords = np.array(json.load(fd))[:, ::-1]
    transformation_dict = pickle_load(opts.rigid_transformation)
    fixed_features = np.load(opts.fixed_features)
    moving_features = np.load(opts.moving_features)
    logging.info("Transformation parameters:")
    t = transformation_dict['t']
    logging.info("    Offset: %s" % str(t))
    center = transformation_dict['center']
    logging.info("    Center: %s" % str(center))
    theta = np.pi * 2 - transformation_dict['theta']
    logging.info("    Thetas: %s" % str(theta * 180 / np.pi))
    s = transformation_dict['s']
    logging.info("    Scale: %s" % str(s))
    r = rotation_matrix(theta)

    logging.info("Applying rigid transformation to moving coordinates")
    xformed_moving_coords = rigid_transformation(
        pts=moving_coords, t=t, r=r, center=center, s=s)
    if PDF is not None:
        figure = plot_points(fixed_coords, xformed_moving_coords, center)
        figure.suptitle("Fixed and moving points after rigid transformation")
        PDF.savefig(figure)
    xformed_moving_coords_um = xformed_moving_coords * voxel_size
    fixed_coords_um = fixed_coords * voxel_size

    logging.info("Matching fixed to moving coords")
    idx_fixed, idx_moving = radius_matching(
        fixed_coords_um,
        xformed_moving_coords_um,
        fixed_features,
        moving_features,
        opts.radius,
        opts.n_workers,
        opts.batch_size,
        dict(max_fdist=opts.max_fdist,
             prom_thresh=opts.prom_thresh))
    logging.info("%d matches found" % len(idx_fixed))

    if PDF is not None:
        figure = plot_points(fixed_coords[idx_fixed],
                             xformed_moving_coords[idx_moving],
                             center)
        figure.suptitle("Matched points")
        PDF.savefig(figure)
        d =  {
            "Fixed pt. count": len(fixed_coords),
            "Moving pt. count": len(moving_coords),
            "Matches:": len(idx_fixed),
        }
        key_text = "\n".join(d.keys())
        value_text = "\n".join([str(_) for _ in d.values()])
        figure = pyplot.figure(figsize=(8, 6))
        figure.text(0, 1, key_text, fontsize=12)
        figure.text(.2, 1, value_text, fontsize=12)
    logging.info("Writing data")
    data = FindNeighborsData(
        fixed_coords = fixed_coords_um,
        moving_coords = xformed_moving_coords_um,
        fixed_features=fixed_features,
        moving_features=moving_features,
        voxel_size=voxel_size[0].astype(float).tolist(),
        fixed_idx=idx_fixed,
        moving_idx=idx_moving,
        radius=opts.radius,
        max_fdist=opts.max_fdist,
        prom_thresh=opts.prom_thresh,
        rigid_transform_params=dict(
            t=t.astype(float).tolist(),
            center=center.astype(float).tolist(),
            theta=theta.astype(float).tolist(),
            s=float(s)
        )
    )
    data.write(opts.output)
    if PDF is not None:
        PDF.close()


if __name__=="__main__":
    main()