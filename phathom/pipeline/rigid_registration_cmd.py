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
        "--fixed-threshold",
        help="The threshold to use for the fixed image. If not supplied, "
        "one will be automatically calculated",
        type=float)
    parser.add_argument(
        "--moving-threshold",
        help="The threshold to use for the moving image. If not supplied, "
        "one will be automatically calculated",
        type=float)
    parser.add_argument(
        "--threshold-multiplier",
        help="Multiply the thresholds by this amount to adjust them.",
        type=float,
        default=1.0)
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
    opts = parse_args(args)
    if opts.visualization_file is not None:
        matplotlib.interactive(opts.interactive)
        PDF = matplotlib.backends.backend_pdf.PdfPages(opts.visualization_file)
    try:
        t0x, t0y, t0z = [float(_) for _ in opts.t0.split(",")]
    except ValueError:
        print("%s must be in the form, \"nnn.nnn,nnn.nnn,nnn.nnn\"" % opts.t0)
        raise
    t0 = (t0z, t0y, t0x)
    try:
        theta0x, theta0y, theta0z = [float(_) for _ in opts.theta0.split(",")]
    except ValueError:
        print("%s must be in the form, \"nnn.nnn,nnn.nnn,nnn.nnn\"" %
              opts.theta0)
        raise
    theta0 = (theta0z, theta0y, theta0x)
    moving_down = read_url(opts.moving_url,
                           opts.moving_url_format,
                           opts.mipmap_level)
    if opts.s0 == None:
        s0z, s0y, s0x = np.array(moving_down.shape) / 2
    else:
        try:
            s0x, s0y, s0z = [float(_) for _ in opts.s0.split(",")]
        except ValueError:
            print("%s must be in the form, \"nnn.nnn,nnn.nnn,nnn.nnn\"" %
                  opts.s0)
            raise
    s0 = (s0z, s0y, s0x)
    fixed_down = read_url(opts.fixed_url,
                          opts.fixed_url_format,
                          opts.mipmap_level)
    if PDF is not None:
        figure = plot_alignment(fixed_down, moving_down, s0, 1, t0, theta0)
        figure.suptitle("Initial alignment")
        PDF.savefig(figure)
        if opts.interactive:
            figure.show()
    #
    # This is a Cellprofiler trick, do Otsu of the log of the image. It
    # seems to pull out the first peak. An alternative may be to use a 3-class
    # otsu of the log (see centrosome.otsu from CellProfiler)
    #
    if opts.moving_threshold is None:
        tmoving = np.exp(threshold_otsu(np.log(moving_down + 1))) - 1
    else:
        tmoving = opts.moving_threshold
    if opts.fixed_threshold is None:
        tfixed = np.exp(threshold_otsu(np.log(fixed_down + 1))) - 1
    else:
        tfixed = opts.fixed_threshold
    tmoving, tfixed = [_ * opts.threshold_multiplier for _ in (tmoving, tfixed)]
    if PDF is not None:
        figure = plot_threshold(fixed_down, moving_down, tfixed, tmoving)
        figure.suptitle("Thresholds used")
        PDF.savefig(figure)
        figure = plot_alignment((fixed_down > tfixed).astype(np.float32),
                                (moving_down > tmoving).astype(np.float32),
                                s0, 1, t0, theta0)
        figure.suptitle("Initial alignment of threshold masks")
        PDF.savefig(figure)
        if opts.interactive:
            figure.show()
    optim_kwargs = {'niter': opts.n_iter,
                    't0': t0,
                    'theta0': theta0,
                    's0': 1}
    threshold = [tmoving, tfixed]

    t_down, theta, center_down, s = coarse_registration(
        moving_down,
        fixed_down,
        threshold,
        optim_kwargs,
        use_hull=opts.use_hull)
    if PDF is not None:
        figure = plot_alignment(fixed_down, moving_down, center_down,
                                s, t_down, theta)
        figure.suptitle("Final alignment")
        PDF.savefig(figure)
        if opts.interactive:
            figure.show()
    full_shape = get_url_shape(opts.fixed_url, 1)
    true_factors = full_shape / np.array(fixed_down.shape)
    t, center = _scale_rigid_params(t_down,
                                    center_down,
                                    true_factors)
    if PDF is not None:
        figure = pylab.figure(figsize=(6, 3))
        text = "Offset: x=%.1f, y=%.1f, z=%.1f\n" % (t[2], t[1], t[0])
        text += "Angles: x=%.1f°, y=%.1f°, z=%.1f°\n" % tuple(
            [_ * 180 / np.pi for _ in theta])
        text += "Center: x=%.1f, y=%.1f, z=%.1f\n" % (
            center[2], center[1], center[0])
        text += "Scale: %.2f\n" % s
        pylab.text(0, 0.1, text, fontsize=12)
        figure.gca().axis("off")
        PDF.savefig(figure)

    pickle_save(
        opts.output,
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
    cimg = cimg.astype(float) / cimg.max()
    figure.add_subplot(2, 2, 1).imshow(cimg[cimg.shape[0] // 2])
    figure.add_subplot(2, 2, 2).imshow(cimg[:, cimg.shape[1] // 2])
    figure.add_subplot(2, 2, 4).imshow(cimg[:, :, cimg.shape[2] // 2])
    return figure


def plot_threshold(fixed, moving, tfixed, tmoving):
    figure = pylab.figure(figsize=(6, 4))
    ax = figure.gca()
    xw = max(tfixed * 3, tmoving * 3)
    clipped_fixed = fixed[fixed < xw]
    clipped_moving = moving[moving < xw]
    yf, xf, _ = ax.hist(clipped_fixed, bins=128, alpha=.5, label="fixed",
                        log=True)
    ym, xm, _ = ax.hist(clipped_moving, bins=128, alpha=.5, label="moving",
                        log=True)
    #
    # We make the Y height of the figure 2x the highest peak to the right
    # of the threshold. xf and xm are the start and end of the bin region
    # so we take the average to get the midpoint.
    #
    xf = (xf[:-1] + xf[1:]) / 2
    xm = (xm[:-1] + xm[1:]) / 2
    yh = min(np.max(yf),
             max(np.max(yf[xf > tfixed]), np.max(ym[xm > tmoving])) * 100)
    ax.plot([tfixed, tfixed], [0, yh], "r-", label="fixed-threshold")
    ax.plot([tmoving, tmoving], [0, yh], "b-", label="moving-threshold")
    ax.set_xlim(0, xw)
    #ax.set_ylim(0, yh)
    ax.legend()
    return figure


if __name__ == "__main__":
    main()