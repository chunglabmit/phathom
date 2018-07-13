# Preprocessing pipeline

The main goal of preprocessing is to correct illumination variations across images
We can also improve data compression by providing a below-autofluoresence threshold

# Basic
1) Apply CLAHE to all image in the stack
2) Apply mask to remove non-tissue information

# Improved
1) Compute the reference histogram with the center slice
2) Compare to the image below to compute linear histogram coefficients
3) Apply the linear transformation to each image in the stack