import argparse
import base64
import json
import logging
import matplotlib.backends.backend_pdf
from matplotlib import pyplot
import os
from functools import partial
import numpy as np
import pandas
import pickle
import sys
import typing
from phathom import plotting
from phathom.registration import registration as reg
from phathom.registration.pcloud import estimate_affine, register_pts
from .find_neighbors_cmd import FindNeighborsData, plot_points


PDF = None


class FilterMatchesData:
    def __init__(self,
                 fixed_coords: np.ndarray,
                 moving_coords: np.ndarray,
                 affine_coords: np.ndarray,
                 fixed_idxs: np.ndarray,
                 moving_idxs: np.ndarray,
                 voxel_size: typing.Sequence[float],
                 affine_transform_fn: typing.Callable[[np.ndarray], np.ndarray]):
        """
        The coordinates after coherence filtering

        :param fixed_coords: Fixed coordinates surviving filtering
        :param moving_coords: Moving coordinates surviving filtering
        :param affine_coords: Moving coordinates after affine transform into
        fixed coordinate space
        :param fixed_idxs: the indices of the chosen fixed coordinates in the
        original fixed coordinates array
        :param moving_idxs: the indices of the chosen moving coordinates in the
        original moving coordinates array
        :param voxel_size: The size of a voxel - a 3 tuple.
        :param affine_transform_fn: A function for converting from the moving
        to the fixed frame.
        """
        self.fixed_coords = fixed_coords
        self.moving_coords = moving_coords
        self.affine_coords = affine_coords
        self.fixed_idxs = fixed_idxs
        self.moving_idxs = moving_idxs
        self.voxel_size = voxel_size
        self.affine_transform_fn = affine_transform_fn

    def write(self, path: str) -> None:
        """
        Write the contents of the FilterMatchData to a json file
        :param path: Path to the file
        """
        d = {
            "fixed-coords": self.fixed_coords.astype(float)[:, ::-1].tolist(),
            "moving-coords": self.moving_coords.astype(float)[:, ::-1].tolist(),
            "affine-coords": self.affine_coords.astype(float)[:, ::-1].tolist(),
            "fixed-idxs": self.fixed_idxs.astype(int).tolist(),
            "moving-idxs": self.moving_idxs.astype(int).tolist(),
            "voxel-size": self.voxel_size[::-1],
            "affine-transform-fn": base64.b64encode(
                pickle.dumps(self.affine_transform_fn)).decode("utf8")
        }
        with open(path, "w") as fd:
            json.dump(d, fd)

    @staticmethod
    def read(path: str):
        """
        Read the FilterMatchData from a file

        :param path: path to the file
        :return: a Filter MatchData
        """
        with open(path) as fd:
            d = json.load(fd)
        affine_transform_fn = pickle.loads(
            base64.b64decode(d["affine-transform-fn"]))
        return FilterMatchesData(
            np.array(d["fixed-coords"])[:, ::-1],
            np.array(d["moving-coords"])[:, ::-1],
            np.array(d["affine-coords"])[:, ::-1],
            np.array(d["fixed-idxs"]),
            np.array(d["moving-idxs"]),
            d["voxel-size"][::-1],
            affine_transform_fn)


def parse_args(args=sys.argv[1:]):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        help="The output from the phathom-geometric-features command",
        required=True)
    parser.add_argument(
        "--output",
        help="The output of this program - a json dictionary with the "
        "coherent matches",
        required=True)
    parser.add_argument(
        "--min-samples",
        help="Minimum number of samples for RANSAC",
        default=30,
        type=int)
    parser.add_argument(
        "--max-distance",
        help="Maximum distance in um between matches after applying affine "
             "transform",
        default=200,
        type=float)
    parser.add_argument(
        "--n-neighbors",
        help="Number of neighbors for displacement coherence",
        default=3,
        type=int)
    parser.add_argument(
        "--min-coherence",
        help="Minimum coherence for displacement coherence filter",
        default = .9,
        type=float)
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
        "--residuals-file",
        help="Residuals .csv file (starting residual error vs final)")
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

    data =  FindNeighborsData.read(opts.input)
    voxel_size = np.array([data.voxel_size])
    fixed_pts = data.fixed_coords
    moving_pts = data.moving_coords
    logging.info("Loaded %d fixed points, %d moving points" %
                 (len(fixed_pts), len(moving_pts)))
    fixed_idx = data.fixed_idx
    moving_idx = data.moving_idx
    logging.info("Loaded %d matches" % len(fixed_idx))

    fixed_keypoints = fixed_pts[fixed_idx]
    moving_keypoints = moving_pts[moving_idx]
    center = np.array(data.rigid_transform_params["center"]) * voxel_size[0]
    if PDF is not None:
        figure = plot_points(fixed_keypoints, moving_keypoints, center)
        figure.suptitle("Initial matches")
        PDF.savefig(figure)
    starting_residuals = reg.match_distance(fixed_keypoints, moving_keypoints)
    if PDF is not None:
        figure = pyplot.figure(figsize=(6, 6))
        plotting.plot_residuals(fixed_keypoints, starting_residuals)
        figure.suptitle("Starting residuals")
        PDF.savefig(figure)
        figure = pyplot.figure(figsize=(6, 3))
        pyplot.hist(starting_residuals, bins=128)
        figure.suptitle("Starting ave. distance [um] = %.1f" %
                        np.mean(starting_residuals))
        PDF.savefig(figure)
    logging.info("Starting average distance [um] = %.1f",
                 np.mean(starting_residuals))
    ransac, ransac_inliers = estimate_affine(
        fixed_keypoints,
        moving_keypoints,
        "ransac",
        min_samples=opts.min_samples)
    ransac_keypoints = register_pts(fixed_keypoints, ransac)
    if PDF is not None:
        figure = plot_points(ransac_keypoints, moving_keypoints, center)
        figure.suptitle("Matches after applying estimated affine xform")
        PDF.savefig(figure)
    ransac_residuals = reg.match_distance(ransac_keypoints, moving_keypoints)
    logging.info("Average residual after applying affine xform: %.1f" %
                 np.mean(ransac_residuals))
    if PDF is not None:
        figure = pyplot.figure(figsize=(6, 6))
        plotting.plot_residuals(ransac_keypoints, ransac_residuals)
        figure.suptitle("Residuals after ransac affine")
        PDF.savefig(figure)
        figure = pyplot.figure(figsize=(6, 3))
        pyplot.hist(ransac_residuals, bins=128)
        figure.suptitle("Residuals after ransac affine")
        PDF.savefig(figure)

    if opts.residuals_file is not None:
        df = pandas.DataFrame(dict(
            coarse=starting_residuals,
            affine=ransac_residuals))
        df.to_csv(opts.residuals_file)
    inlier_index = np.where(ransac_residuals < opts.max_distance)
    fixed_inlier_idx = fixed_idx[inlier_index]
    moving_inlier_idx = moving_idx[inlier_index]
    fixed_keypoints_dist = fixed_keypoints[inlier_index]
    moving_keypoints_dist = moving_keypoints[inlier_index]
    #
    # Calculate the affine transform in the raw coordinate space
    #
    degree = 1
    um2voxel = 1 / voxel_size
    model = reg.fit_polynomial_transform(
        fixed_keypoints_dist * um2voxel,
        moving_keypoints_dist * um2voxel,
        degree)
    model_z, model_y, model_x = model
    affine_transformation = partial(reg.polynomial_transform,
                                    degree=degree,
                                    model_z=model_z,
                                    model_y=model_y,
                                    model_x=model_x)
    affine_keypoints_vox = affine_transformation(pts=fixed_keypoints * um2voxel)
    affine_keypoints_vox_dist = affine_keypoints_vox[inlier_index]
    affine_keypoints = affine_keypoints_vox / um2voxel
    affine_keypoints_dist = affine_keypoints[inlier_index]
    if PDF is not None:
        figure = plot_points(affine_keypoints_dist, moving_keypoints_dist,
                             center)
        figure.suptitle("Affine transformed keypoints")
        PDF.savefig(figure)
    #
    # Calculate displacement coherence
    #
    coherences = reg.coherence(opts.n_neighbors,
                               affine_keypoints_dist,
                               moving_keypoints_dist)
    if PDF is not None:
        figure = pyplot.figure(figsize=(6, 3))
        pyplot.hist(coherences, bins=128)
        figure.suptitle("Distribution of displacement coherences")
        PDF.savefig((figure))
    logging.info("Average coherence: %.2f" % np.mean(coherences))
    #
    # Filter out incoherent matches
    #
    coherent_index = np.where(coherences > opts.min_coherence)[0]
    fixed_coherent_idx = fixed_inlier_idx[coherent_index]
    moving_coherent_idx = moving_inlier_idx[coherent_index]
    logging.info("Found %d outliers" % (len(coherences) - len(coherent_index)))
    fixed_keypoints_coherent = fixed_keypoints_dist[coherent_index]
    fixed_keypoints_coherent_vox = fixed_keypoints_coherent * um2voxel
    affine_keypoints_coherent = affine_keypoints_dist[coherent_index]
    affine_keypoints_coherent_vox = affine_keypoints_coherent * um2voxel
    moving_keypoints_coherent = moving_keypoints_dist[coherent_index]
    coherent_residuals = reg.match_distance(affine_keypoints_coherent,
                                            moving_keypoints_coherent)
    if PDF is not None:
        figure = pyplot.figure(figsize=(6, 6))
        plotting.plot_residuals(affine_keypoints_coherent, coherent_residuals)
        figure.suptitle("Points by residual after coherence filtering")
        PDF.savefig(figure)
        figure = plot_points(affine_keypoints_coherent,
                             moving_keypoints_coherent, center)
        figure.suptitle("Coherent affine-transformed points")
        PDF.savefig(figure)
        figure = pyplot.figure(figsize=(6, 3))
        pyplot.hist(coherent_residuals, bins=128)
        figure.suptitle("Residuals after coherence filtering")
        PDF.savefig(figure)
    logging.info("Average residual after coherence filtering: %.1f" %
                 np.mean(coherent_residuals))
    data = FilterMatchesData(
        fixed_keypoints_coherent,
        moving_keypoints_coherent,
        affine_keypoints_coherent,
        fixed_coherent_idx,
        moving_coherent_idx,
        voxel_size[0].tolist(),
        affine_transformation)
    data.write(opts.output)
    if PDF is not None:
        PDF.close()


if __name__ == "__main__":
    main()
