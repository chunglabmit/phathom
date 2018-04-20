import zarr
import numcodecs


def write_subarray(data, z_arr, start):
    """ Write a subarray into a zarr array.

    :param data: An array of data to write
    :param z_arr: A reference to a persistent zarr array (with write access)
    :param start: Array-like containing the 3 indices of the starting coordinate
    """
    stop = tuple(s+c for s,c in zip(start, data.shape))
    assert start[0] < stop[0]
    assert start[1] < stop[1]
    assert start[2] < stop[2]
    z_arr[start[0]:stop[0], start[1]:stop[1], start[2]:stop[2]] = data
