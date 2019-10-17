# Phathom pipeline

This package has commands for running chunks of Phathom so
it can be executed as a pipeline from the command-line.

## Co-registration workflow

The co-registration workflow takes similarly-stained images from a
fixed and moving volume, derives an elastic transformation between
fixed and moving and then applies that transformation to images in
the moving frame to warp them to the fixed frame.

The steps (see below for command details):

* *phathom-preprocess*: apply CLAHE
(http://tog.acm.org/resources/GraphicsGems/) to both the fixed and
 moving stacks.
* *precomputed-tif*: Make Neuroglancer volumes of the data for
inspection and downsampling (see [precomputed-tif](https://github.com/chunglabmit/precomputed-tif))
* *phathom-rigid-registration*: Perform a coarse rigid registration
using a downsampled version of the stack.

## Commands

### *phathom-preprocess* - apply CLAHE to a TIFF stack

```bash
phathom-preprocess \
   --input <input-path> \
   --output <output-path> \
   [--output-format <output-format> ] \
   [--threshold <threshold> ] \
   [--kernel-size <kernel-size> ] \
   [--clip-limit <clip-limit>] \
   [--n-workers <n-workers>]
```
where

* **input-path** is the path to the input TIFF files
* **output-path** is the path to the directory for the output volume
* **output-format** is the format for the output volume. Currently
  the default and only allowable value is zarr
* **threshold** is the binary threshold to use to mask the image.
Portions of the image less than the threshold will be rendered as 0.
* **kernel-size** is the size of the adaptation kernel used in CLAHE
* **clip-limit** is the CLAHE clip-limit (see [equalize_adapthist](https://scikit-image.org/docs/dev/api/skimage.exposure.html#skimage.exposure.equalize_adapthist))
* **n-workers** is the number of worker processes when running
in parallel. This defaults to 1.

### *phathom-rigid-registration* - perform a rigid registration

```bash
phathom-rigid-registration \
   --fixed-url <fixed-url> \
   --moving-url <moving-url> \
   --output <output-pickle> \
   --mipmap-level <mipmap-level> \
   [--fixed-url-format <fixed-url-format>] \
   [--moving-url-format <moving-url-format>] \
   [--visualization-file <visualization-file>] \
   [--n-iter <n-iter>] \
   [--t0 <t0>] \
   [--theta0 <theta0>] \
   [--s0 <s0>]
   [--use-hull]
   [--interactive]
```

where:
* **fixed-url** is the Neuroglancer URL of the fixed volume, e.g.
"https://my-server.org/precomputed/fixed"
* **moving-url** is the Neuroglancer URL of the moving volume
* **output-pickle** is the path name for the pickled transformation
function produced by this program
* **mipmap-level** is the mipmap level for the reduced resolution, e.g.
"32" for a 32x reduction in size. Typical values are 1, 2, 4, 16 and 32
* **fixed-url-format** for a file URL, the format of the precomputed
fixed volume, "tiff", "zarr" or "blockfs". Defaults to blockfs.
* **moving-url-format** for a file URL, the format of the precomputed
moving volume, "tiff", "zarr" or "blockfs". Defaults to blockfs.
* **visualization-file** the path to a .pdf file that is output by
the program, containing informative visualizations of the various
processing steps.
* **n-iter** the number of basin hopping iterations to perform
* **t0** the initial guess for the translational displacement. Default
is "0,0,0". The guess should be entered as three comma-separated values
in the original space.
* **theta0** the initial guess for the rotational component of the
registration. Default is "0,0,0". The guess should be entered as
three comma-separated floating point numbers, in radians.
* **s0** the coordinates of the rotational center in the moving frame
of reference
* **--use_hull** take the convex hull of the threshold mask instead
of using the mask directly
* **--interactive** show the plots as they are written to the
visualization file