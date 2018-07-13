# Registration Pipeline

1) Convert TIFFs to chunked Zarr array (tifs_to_zarr)
2) Downsample images 100x in each dimension
3) Perform coarse registration (coarse.coarse_registration)
4) Detect blobs to match (registration.detect_blobs_parallel)
5) Extract geometric features (pcloud.geometric_features)
6) Transform fixed points with the coarse transformation
7) Perform neighborhood point matching (pcloud.radius_matching)
8) Filter out incoherent displacements
9) Estimate affine transformation with matches
10) Filter out long-distance matches
11) Re-estimate affine transformation
12) Fit RBF mapping to the affine transformed matches
13) Warp a regular grid for to approximate the deformation
14) Use map_coordinates to register the moving image onto the fixed image
