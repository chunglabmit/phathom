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

# Using imagej package
- Install JDK and JRE (sudo apt-get install default-jdk/jre)
- Install Cython with pip
- Install pyjnius with pip
- Install imglyb with "conda install -c hanslovsky imglib2-imglyb"
Currently getting legacy error...

