# Phathom

[![Travis CI Status](https://travis-ci.org/chunglabmit/phathom.svg?branch=master)](https://travis-ci.org/chunglabmit/phathom)
[![Documentation Status](https://readthedocs.org/projects/phathom/badge/)](http://phathom.readthedocs.io/en/latest/)

Phathom is a Python package for analyzing terabyte-scale biological images.
It relies on distributed computing to scale up image processing tasks that
are commonly encountered when using [Selective Plane Illumination Microscopy (SPIM)][1].
These image processing task include:
  - Non-rigid registration for multi-round imaging
  - Individual nuclei segmentation
  - Cell phenotype classification
  - Brain atlas alignment

## Getting Started
Clone this repository and run `python setup.py install`.
Documentation is hosted on [readthedocs](http://phathom.readthedocs.io/en/latest/).

## Background
The purpose of Phathom is to improve the scalability of computational pipelines used in biological image processing.
Popular open-source tools such as [Scikit-image][2] cannot directly process images that are larger than memory.
To address this, Phathom uses persistent [Zarr][3] arrays to store and access large images on-disk or from a shared file system.
For processing, Phathom uses the [SCOOP][4] library to coordinate a pool of workers, either locally or in a distributed manner.
Phathom supports HPC environments using [SLURM][5].

[//]: # (References)

[1]: https://en.wikipedia.org/wiki/Light_sheet_fluorescence_microscopy
[2]: https://github.com/scikit-image/scikit-image
[3]: https://github.com/zarr-developers/zarr
[4]: https://github.com/soravux/scoop
[5]: https://slurm.schedmd.com/

## Authors
Phathom is maintained by members of the [Kwanghun Chung Lab](http://www.chunglab.org/) at MIT.
Original Author: Justin Swaney
Primary Contack: Lee Kamensky (lkaments@mit.edu)
