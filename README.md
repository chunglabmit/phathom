# Phathom

[![Travis CI Status](https://travis-ci.org/chunglabmit/phathom.svg?branch=master)](https://travis-ci.org/chunglabmit/phathom)

Phathom is a Python package for analyzing terabyte-scale biological images.
It relies on distributed computing to scale up image processing tasks that
are commonly encountered when using selective plane illumination microscopy (SPIM).

## Getting Started
Clone this repository and run `python setup.py install`. 
Documentation will be hosted on readthedocs.

## Background
The purpose of Phathom is to improve the scalability of computational pipelines used in biological image processing.
Popular open-source tools such as Scikit-image cannot directly process images that are larger than memory.
To address this, Phathom uses persistent Zarr arrays to store and access large images on-disk or from a shared file system.
For processing, Phathom uses the SCOOP library to coordinate a pool of workers, either locally or in a distributed manner.
Phathom supports HPC environments using SLURM.