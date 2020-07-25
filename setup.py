from setuptools import setup, find_packages

version = "0.1.0"

CONSOLE_SCRIPTS = [
    'phathom-segmentation=phathom.segmentation:main',
    'phathom-score-centroids=phathom.score:main',
    'phathom-preprocess=phathom.pipeline.preprocess_cmd:main',
    'phathom-rigid-registration=phathom.pipeline.rigid_registration_cmd:main',
    'phathom-non-rigid-registration='
    'phathom.pipeline.non_rigid_registration_cmd:main',
    'phathom-geometric-features=phathom.pipeline.geometric_features_cmd:main',
    'phathom-find-neighbors=phathom.pipeline.find_neighbors_cmd:main',
    'phathom-filter-matches=phathom.pipeline.filter_matches_cmd:main',
    'phathom-fit-nonrigid-transform='
    'phathom.pipeline.fit_nonrigid_transform_cmd:main',
    'phathom-warp-image=phathom.pipeline.warp_image:main',
    'phathom-warp-points=phathom.pipeline.warp_points_cmd:main',
    'phathom-pickle-alignment=phathom.pipeline.pickle_alignment_cmd:main',
    'phathom-find-corr-neighbors=phathom.pipeline.find_corr_neighbors_cmd:main'
]

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
        "tqdm",
        "scikit-learn",
        "pandas",
        "tifffile",
        "lapsolver",
    ],
    author="Kwanghun Chung Lab",
    packages=["phathom",
              "phathom.atlas",
              "phathom.db",
              "phathom.io",
              "phathom.pipeline",
              "phathom.phenotype",
              "phathom.preprocess",
              "phathom.registration",
              "phathom.segmentation"
              ],
    entry_points={
        'console_scripts': CONSOLE_SCRIPTS
    },
    url="https://github.com/chunglabmit/phathom",
    license="MIT",
    classifiers=[
        "Development Status :: 3 - Alpha",
        'Programming Language :: Python :: 3.5',
    ]
)
