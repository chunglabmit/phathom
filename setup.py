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
        "PyMaxflow",
        "scipy",
        "scikit-image",
        "zarr",
        "numpy",
        "h5py",
        "pyina",
        "tqdm",
        "scikit-learn"
    ],
    author="Kwanghun Chung Lab",
    packages=["phathom"],
    entry_points={ 'console_scripts': [
        'phathom-segmentation=phathom.segmentation:main',
        'phathom-score-centroids=phathom.score.main'
    ]},
    url="https://github.com/chunglabmit/phathom",
    license="MIT",
    classifiers=[
        "Development Status :: 3 - Alpha",
        'Programming Language :: Python :: 3.5',
    ]
)