from setuptools import setup, find_packages

version = "0.1.0"

with open("./README.md") as fd:
    long_description = fd.read()

setup(
    name="phathom",
    version=version,
    description=
    "Phenotypic analysis of brain tissue at single-cell resolution",
    long_description=long_description,
    install_requires=[
        "matplotlib",
        "numpy",
        "PyMaxflow",
        "scipy",
        "scikit-image",
        "tifffile",
        "zarr"
    ],
    author="Kwanghun Chung Lab",
    packages=["phathom"],
    entry_points={ 'console_scripts': [
        'phathom-segmentation=phathom.segmentation:main'
    ]},
    url="https://github.com/chunglabmit/phathom",
    license="MIT",
    classifiers=[
        "Development Status :: 3 - Alpha",
        'Programming Language :: Python :: 3.5',
    ]
)