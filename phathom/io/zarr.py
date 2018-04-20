import zarr
import numcodecs


def write_chunk(data, z_arr, start):
    stop = tuple(s+c for s,c in zip(start, data.shape))
    assert start[0] < stop[0]
    assert start[1] < stop[1]
    assert start[2] < stop[2]
    z_arr[start[0]:stop[0], start[1]:stop[1], start[2]:stop[2]] = data
