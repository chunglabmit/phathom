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
* OR *phathom-non-rigid-registration* Peform a coarse non-rigid
registration using a downsampled version of the stack.
* *detect-blobs*: Detect blobs for both the fixed and moving stacks.
detect-blobs is part of the [eflash_2018](https://github.com/chunglabmit/eflash_2018)
package
* *phathom-geometric-features*: Calculate geometric features from
the detected blobs from both the moving and fixed stacks.
* *phathom-find-neighbors*: Match neighboring blobs within a search
radius.
* *phathom-filter-matches*: Filter matches based on re-estimation
of affine transform using ransac.
* *phathom-fit-nonrigid-transform*: Create a transform for warping the
moving image onto the fixed one.
* *phathom-warp-image*: Warp moving images onto the fixed reference
frame.

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

### *phathom-non-rigid-registration* - coarse non-rigid registration

*phathom-non-rigid-registration* is an alternative to
*phathom-rigid-registration*. It performs a rough non-rigid registration
using the Elastix toolkit. It is typically initialized using parameters
from the [rigid-rotate](https://github.com/chunglabmit/rigidrotate)
web app - this gives the application an initial starting point and may
be neccessary if there is a large rotation or translation between the
two images.

The result is a pickled dictionary containing the transform
function from the moving coordinate system to the fixed one.

```bash
phathom-non-rigid-registration \
    --fixed-url <fixed-url> \
    --moving-url <moving-url> \
    --output <output> \
    [--fixed-url-format <fixed-url-format>] \
    [--moving-url-format <moving-url-format] \
    [--mipmap-level <mipmap-level>] \
    [--grid-points <grid-points>] \
    [--initial-rotation <initial-rotation>] \
    [--initial-translation <initial-translation>] \
    [--rotation-center <rotation-center>]
```
where
* **fixed-url** is the Neuroglancer URL of the fixed volume
* **fixed-url-format** is the data format of the fixed URL if it is
 a file URL. Valid values are "blockfs", "tiff" or "zarr".
* **moving-url** is the Neuroglancer URL of the moving volume
* **moving-url-format** is the data format of the moving URL if it is
 a file URL. Valid values are "blockfs", "tiff" or "zarr".
* **output** is the pickle file holding the interpolator
* **mipmap-level** is the mipmap level of the downsampling,
 e.g. 32 or 64
* **grid-points** is the number of grid points across the image in
 each direction
* **initial-rotation** is the initial rotation of the moving image
 along the X, Y and Z axes as 3 comma-separated values in degrees.
* **rotation-center** is the rotation center from rigid-rotate,
 the X,Y,Z values separated by commas. Default is the image center.
* **initial-translation** is the initial translation of the moving image
 along the X, Y and Z axes as 3 comma-separated values.

### *phathom-geometric-features* - calculate geometric features

usage:
```bash
phathom-geometric-features \
    --input <input-path> \
    --output <output-path> \
    [--voxel-size <voxel-size>] \
    [--n-workers <n-workers>]
```

where:
* **input-path** is the path to a file of blob coordinates, e.g. as
generated by the **detect-blobs** command (see above).
* **output-path** is the path to the file of geometric features per
blob to be written. This is a pickled Numpy array.
* **voxel-size** the size of one voxel in microns. The format is
three comma-separated numbers for the x, y and z sizes. The default
is "1.8,1.8,2.0" which is the voxel size used in the Chung lab
when imaging at 4x.
* **n-workers** is the number of worker processes used when computing
the geometric features.

### *phathom-find-neighbors*

*phathom-find-neighbors* finds correspondences between points in
the fixed and moving volumes. It applies a transformation to the
moving points to bring them into the fixed frame of reference and
then does the matching based on several criteria:
* radius - after transformation, the points must be within a given
radius of each other
* feature distance - The features from *phathom-geometric-features*
for the corresponding points must match within a certain accuracy.
* prominence threshold - the feature match must be this much better
than the other possible matches in the neighborhood.

Usage:
```bash
phathom-find-neighbors \
    --fixed-coords <fixed-coords> \
    --moving-coords <moving-coords> \
    --fixed-features <fixed-features> \
    --moving-features <moving-features> \
    --output <output> \
    [--rigid-transformation <rigid-transformation>] \
    [--non-rigid-transformation <non-rigid-transformation>] \
    [--voxel-size <voxel-size>] \
    [--radius <radius>] \
    [--max-fdist <max-fdist>] \
    [--prom-thresh <prom-thresh>] \
    [--n-workers <n-workers>] \
    [--batch-size <batch-size>] \
    [--visualization-file <visualization-file>] \
    [--interactive] \
    [--log-level <log-level>]
```

where
* **fixed-coords** is the path to the blob coordinate file for the
fixed volume. This is a .json file in x, y, z format e.g. as made by
detect-blobs.
* **moving-coords** is the path to the blob coordinate file for the
moving volume. This is a .json file in x, y, z format e.g. as made by
detect-blobs.
* **fixed-features** is the path to the features file for the fixed
volume, e.g. as produced by phathom-geometric-features
* **moving-features** is the path to the features file for the moving
volume, e.g. as produced by phathom-geometric-features
* **rigid-transformation** is the rigid transformation to convert the
moving coordinates to an approximation of the fixed volume space, e.g.
as produced by phathom-rigid-registration. Either --rigid-transform
or --non-rigid-transform must be specified.
* **non-rigid-transformation** is the non-rigid rough transformation to
convert moving coordinates into fixed coordinates, e.g. as produced by "
phathom-non-rigid-registration
* **output** is the output of this program, a JSON dictionary with the
intermediate results to this stage.
* **voxel-size** is the size of a voxel in microns, three
comma-separated values in x, y, z order e.g. "1.8,1.8,2.0"
* **radius** is the search radius for matches, in microns
* **max-fdist** is the maximum allowed feature distance for a match
* **prom-thresh** is the prominence threshold for a match. All competing
matches must have a feature distance that is less than this fraction of
the match being considered
* **n-workers** is the number of worker processes to use during
computation
* **batch-size** is the number of fixed points to process per worker
invocation
* **visualization-file** is the path to the PDF file output by this
program. This file contains helpful visualizations that document the
program's progress.
* **interactive** If supplied, the program will display each of the
visualizations as they are created. Only supply if you have a display.
* **log-level** is the log verbosity level. Default is WARNING, options
are DEBUG, INFO, WARNING and ERROR

### *phathom-filter-matches*

*phathom-filter-matches* takes the matches from *find-neighbors* and
uses RANSAC to create an affine transform model. It then filters
the matches further based on the correspondence beteween fixed and
moving pairs after applying the affine transform. The filterered
point set and transform are written as the output.

```bash
phathom-filter-matches \
    --input <input-path> \
    --output <output-path> \
    [--min-samples <min-samples>] \
    [--max-distance <max-distance>] \
    [--n-neighbors <n-neighbors> ] \
    [--min-coherence <min-coherence> ] \
    [--visualization-file <visualization-file> ] \
    [--interactive ] \
    [--residuals-file ] \
    [--log-level ]
```

where:
* **input-path** is the path to the input file, e.g. as produced by
*phathom-find-neighbors*
* **output-path** is the path to the output file from this program,
an intermediate file, stored as a JSON dictionary with keys for the
filtered sets of fixed and moving points and the pickled affine
transform.
* **min-samples** the minimum number of samples selected at random
from the source points when making the model. Default is 30, must be
more than four.
* **max-distance** the maximum distance allowed between fixed and
moving points after applying the affine transform
* **n-neighbors** the number of alternate matches considered when
caluclating the coherence distance. Default is 3
* **min-coherence** the minimum coherence allowed. Default is .9, range
is 0 to 1.
* **visualization-file** the filename of the optional PDF of
visualization plots
* **interactive** to display the plots interactively
* **log-level** The log level for logging, one of DEBUG, INFO,
WARNING or ERROR. Default is WARNING.

### *phathom-fit-nonrigid-transform*

*phathom-fit-nonrigid-transform* creates a pickled point transform
function from the fixed volume space to the moving volume space.

```bash
phathom-fit-nonrigid-transform \
    --input <input-path> \
    --output <output-path> \
    [--visualization-file <visualization-path>] \
    [--interactive]\
    [--max-samples <max-samples>] \
    [--smoothing <smoothing>] \
    [--grid-points <grid-points>] \
    [--fixed-url <fixed-url>] \
    [--moving-url <moving-url>] \
    [--fixed-url-format <fixed-url-format>] \
    [--moving-url-format <moving-url-format>] \
    [--log-level <log-level>]
```
where
* **input-path** is the input file from *phathom-filter-matches*

* **output-path** is the pickled nonrigid transform

* **visualization-file** is the path to the PDF file output by this
  program. This file contains helpful visualizations that document the
  program's progress.

* **--interactive** if supplied, the program will display each of the
  visualizations as they are created. Only supply if you have a display.

* **max-samples** is the maximum number of samples to be used when
  constructing the thin-plate spline

* **smoothing** is the smoothing for the thin-plate spline

* **grid-points** is the number of grid points in the X, Y and Z when
  creating a bspline approximation

* **fixed-url** is the Neuroglancer URL of the fixed volume, e.g.
  "https://my-server.org/precomputed/fixed". This is only used to
  visualize the overlap and is not needed for the basic calculation.

* **moving-url** is the Neuroglancer URL of the moving volume, e.g.
  "https://my-server.org/precomputed/fixed". This is only used to
   visualize the overlap and is not needed for the basic calculation.

* **fixed-url-format** is the data format of the fixed URL if it is a
  file URL. Valid values are "blockfs", "tiff" or "zarr".

* **moving-url-format** is the data format of the moving URL if it is a
  file URL. Valid values are "blockfs", "tiff" or "zarr".

* **log-level** is the log verbosity level. Default is WARNING, options
  are DEBUG, INFO, WARNING and ERROR

### *phathom-warp-points*

*phathom-warp-points* translates points in the fixed reference frame
to the moving reference frame using the interpolator output by
*phathom-fit-nonrigid-transform*

```bash
phathom-warp-points \
    --interpolator <interpolator> \
    --input <input> \
    --output <output> \
    [--n-workers <n-workers>] \
    [--batch-size <batch-size>]
```

where

* **interpolator** is the interpolator pickle output by
*phathom-fit-nonrigid-transform*

* **input** is a .json file of coordinates in the fixed frame of
reference

* **output** is the name of the output file to be written - a .json
file of coordinates in the moving frame of reference

* **n-workers** is the optional number of worker processes to use. If
not specified, one worker is used per core.

* **batch-size** is the number of coordinates per worker invocation

### *phathom-warp-image*

*phathom-warp-image* warps an image from the moving frame of reference
to the fixed frame of reference using the interpolator output by
*phathom-fit-nonrigid-transform*. The image is written as a Neuroglancer
volume.

```bash
phathom-warp-image \
    --interpolator <interpolator> \
    --url <url> \
    --output <output> \
    [--url-format <url-format>] \
    [--n-workers <n-workers>] \
    [--n-writers <n-writers>] \
    [--n-levels <n-levels>] \
    [--output-shape <output-shape>] \
    [--silent] \
    [--use-gpu]
```

where

* **interpolator** is the interpolator pickle file output by
*phathom-fit-nonrigid-transform*

* **url** is the neuroglancer URL of the moving image. May be specified
          multiple times.
* **url-format** is the format of the URL if a file URL. Must be
          specified once per URL if specified at all. Valid values are
          "tiff", "zarr" and "blockfs". Default is blockfs.

* **output** is the location for the Neuroglancer data source for the
          warped image. Must be specified once per input URL.

* **n-workers** The number of workers devoted to transforming
          coordinates (if --use-gpu is not specified)

* **n-writers** is the number of worker processes devoted to writing
          output data

* **n-levels** is the number of levels in each output volume

* **output-shape** is the output volume shape in x,y,z format. If not
           specified, it will be the same as the shape of the first
           input volume.

* **--silent** Do not print progress bars

* **--use-gpu** Use a GPU to perform the warping computation
