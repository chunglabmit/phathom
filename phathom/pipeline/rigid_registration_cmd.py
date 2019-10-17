import argparse
import numpy as np
import matplotlib.backends.backend_pdf
import matplotlib.pyplot as pylab
from precomputed_tif.client import get_info, read_chunk
from skimage.filters import threshold_otsu
import sys

from phathom.utils import pickle_save
from phathom.registration.coarse import \
    coarse_registration, rigid_warp, _scale_rigid_params


def parse_args(args=sys.argv[1:]):
    parser = argparse.ArgumentParser()
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
        "--output",
        help="The filename of the pickled alignment function that is "
        "output by this program",
        required=True)
    parser.add_argument(
        "--mipmap-level",
        help="The mipmap level of the downsampling, e.g. 32 or 64",
        default=16,
        type=int)
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
        "--visualization-file",
        help="The path to the PDF file output by this program. "
        "This file contains helpful visualizations that document the "
        "program's progress.")
    parser.add_argument(
        "--n-iter",
        help="the number of basin hopping iterations to perform",
        default=10,
        type=int)
    parser.add_argument(
        "--t0",
        help="the initial guess for the translational displacement. Default "
        "is \"0,0,0\". The guess should be entered as three comma-separated "
        "values in the original space in the order, x, y, z.",
        default="0,0,0")
    parser.add_argument(
        "--theta0",
        help="the initial guess for the rotational component of the "
             "registration. Default is \"0,0,0\". The guess should "
             "be entered as "
             "three comma-separated floating point numbers, in radians.",
        default="0,0,0")
    parser.add_argument(
        "--s0",
        help="the coordinates of the rotational center in the moving frame "
        "of reference. The default is the image centroid.")
    parser.add_argument(
        "--use-hull",
        help="Use the convex hull of the thresholded volumes instead of "
        "the raw thresholded volumes when doing the coarse registration.",
        action="store_true")
    parser.add_argument(
        "--interactive",
        help="If supplied, the program will display each of the visualizations "
        "as they are created. Only supply if you have a display.",
        action="store_true")
    return parser.parse_args(args)


PDF = None


def read_url(url, url_format, mipmap_level):
    z1, y1, x1 = get_url_shape(url, mipmap_level)
    return read_chunk(url, 0, x1, 0, y1, 0, z1,
                      level=mipmap_level, format=url_format)


def get_url_shape(url, mipmap_level):
    info = get_info(url)
    scale = info.get_scale(mipmap_level)
    x1, y1, z1 = scale.shape
    return z1, y1, x1


def main(args=sys.argv[1:]):
    global PDF
    args = parse_args(args)
    matplotlib.interactive(args.interactive)
    if args.visualization_file is not None:
        PDF = matplotlib.backends.backend_pdf.PdfPages(args.visualization_file)
    try:
        t0x, t0y, t0z = [float(_) for _ in args.t0.split(",")]
    except ValueError:
        print("%s must be in the form, \"nnn.nnn,nnn.nnn,nnn.nnn\"" % args.t0)
        raise
    t0 = (t0z, t0y, t0x)
    try:
        theta0x, theta0y, theta0z = [float(_) for _ in args.theta0.split(",")]
    except ValueError:
        print("%s must be in the form, \"nnn.nnn,nnn.nnn,nnn.nnn\"" %
              args.theta0)
        raise
    theta0 = (theta0z, theta0y, theta0x)
    moving_down = read_url(args.moving_url,
                           args.moving_url_format,
                           args.mipmap_level)
    if args.s0 == None:
        s0z, s0y, s0x = np.array(moving_down.shape) / 2
    else:
        try:
            s0x, s0y, s0z = [float(_) for _ in args.s0.split(",")]
        except ValueError:
            print("%s must be in the form, \"nnn.nnn,nnn.nnn,nnn.nnn\"" %
                  args.s0)
            raise
    s0 = (s0z, s0y, s0x)
    fixed_down = read_url(args.fixed_url,
                          args.fixed_url_format,
                          args.mipmap_level)
    if PDF is not None:
        figure = plot_alignment(fixed_down, moving_down, s0, 1, t0, theta0)
        figure.suptitle("Initial alignment")
        PDF.savefig(figure)

    tmoving = threshold_otsu(moving_down)
    tfixed = threshold_otsu(fixed_down)
    if PDF is not None:
        figure = plot_threshold(fixed_down, moving_down, tfixed, tmoving)
        figure.suptitle("Thresholds used")
        PDF.savefig(figure)
        figure = plot_alignment((fixed_down > tfixed).astype(np.float32),
                                (moving_down > tmoving).astype(np.float32),
                                s0, 1, t0, theta0)
        figure.suptitle("Initial alignment of threshold masks")
        PDF.savefig(figure)
    optim_kwargs = {'niter': args.n_iter,
                    't0': t0,
                    'theta0': theta0,
                    's0': 1}
    threshold = [tmoving, tfixed]

    t_down, theta, center_down, s = coarse_registration(
        moving_down,
        fixed_down,
        threshold,
        optim_kwargs,
        use_hull=args.use_hull)
    if PDF is not None:
        figure = plot_alignment(fixed_down, moving_down, center_down,
                                s, t0, theta)
        figure.suptitle("Final alignment")
        PDF.savefig(figure)
    full_shape = get_url_shape(args.fixed_url, 1)
    true_factors = full_shape / np.array(fixed_down.shape)
    t, center = _scale_rigid_params(t_down,
                                    center_down,
                                    true_factors)
    pickle_save(
        args.output,
        dict(t=t, center=center, s=s, theta=theta))
    if PDF is not None:
        PDF.close()


def plot_alignment(fixed, moving, center, s, t0, theta0):
    figure = pylab.figure(figsize=(6, 6))
    wmoving = rigid_warp(moving, t0, theta0, s, center, fixed.shape)
    cimg = np.column_stack([
        fixed.flatten(),
        wmoving.flatten(),
        np.zeros(np.prod(fixed.shape), fixed.dtype)]) \
        .reshape(fixed.shape[0],
                 fixed.shape[1],
                 fixed.shape[2], 3)
    figure.add_subplot(2, 2, 1).imshow(cimg[cimg.shape[0] // 2])
    figure.add_subplot(2, 2, 2).imshow(cimg[:, cimg.shape[1] // 2])
    figure.add_subplot(2, 2, 4).imshow(cimg[:, :, cimg.shape[2] // 2])
    return figure


def plot_threshold(fixed, moving, tfixed, tmoving):
    figure = pylab.figure(figsize=(4, 6))
    ax = figure.gca()
    yf, xf, _ = ax.hist(fixed.flatten(), bins=128, alpha=.5, label="fixed")
    ym, xm, _ = ax.hist(moving.flatten(), bins=128, alpha=.5, label="moving")
    #
    # We make the Y height of the figure 2x the highest peak to the right
    # of the threshold. xf and xm are the start and end of the bin region
    # so we take the average to get the midpoint.
    #
    xf = (xf[:-1] + xf[1:]) / 2
    xm = (xm[:-1] + xm[1:]) / 2
    yh = max(np.max(yf[xf > tfixed]), np.max(ym[xm > tmoving])) * 2
    xw = max(tfixed * 3, tmoving * 3)
    ax.plot([tfixed, tfixed], [0, yh], "r-")
    ax.plot([tmoving, tmoving], [0, yh], "r-")
    ax.set_xlim(0, xw)
    ax.set_ylim(0, yh)
    return figure


if __name__ == "__main__":
    main()