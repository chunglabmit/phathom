import numpy as np
import pandas

from precomputed_tif.client import ArrayReader
from scipy import ndimage
import pickle
import typing

def corr(blk_fixed:np.ndarray,
         blk_moving:np.ndarray,
         pt:typing.Sequence[int],
         orig_fixed:np.ndarray,
         orig_moving:np.ndarray,
         dx:int, dy:int, dz:int, off_x:int = 0, off_y:int = 0, off_z:int = 0)\
        ->float:
    """
    Calculate the pearson correlation coefficient for a point and its
    offset in the moving coordinate system.

    :param blk_fixed: the fixed block of volume pixels
    :param blk_moving: the moving block of volume pixels
    :param pt: the center point of the correlation
    :param orig_fixed: the z, y and x coordinates of the origin of the
    fixed block
    :param orig_moving: the z, y and x coordinates of the origin of the
    moving block
    :param dx: the half-width of the volume to be measured in the x direction
    :param dy: the half-width of the volume to be measured in the y direction
    :param dz: the half-width of the volume to be measured in the z direction
    :param off_x: the x offset of the center for the moving volume
    :param off_y: the y offset of the center for the moving volume
    :param off_z: the z offset of the center for the moving volume
    :return: the correlation coefficient
    """
    x0f = pt[2] - dx - orig_fixed[2]
    x1f = pt[2] + dx + 1 - orig_fixed[2]
    y0f = pt[1] - dy - orig_fixed[1]
    y1f = pt[1] + dy + 1 - orig_fixed[1]
    z0f = pt[0] - dz - orig_fixed[0]
    z1f = pt[0] + dz + 1  - orig_fixed[0]
    x0m, x1m = [_ + off_x - orig_moving[2] + orig_fixed[2] for _ in (x0f, x1f)]
    y0m, y1m = [_ + off_y - orig_moving[1] + orig_fixed[1] for _ in (y0f, y1f)]
    z0m, z1m = [_ + off_z - orig_moving[0] + orig_fixed[0] for _ in (z0f, z1f)]
    return np.corrcoef(blk_fixed[z0f:z1f, y0f:y1f, x0f:x1f].flatten(),
                       blk_moving[z0m:z1m, y0m:y1m, x0m:x1m].flatten())[1, 0]


def gradient(blk_fixed:np.ndarray,
             blk_moving:np.ndarray,
             pt: typing.Sequence[int],
             orig_fixed: np.ndarray,
             orig_moving: np.ndarray,
             dx: int, dy: int, dz: int,
             off_x: int = 0, off_y: int = 0, off_z: int = 0)\
        -> np.ndarray:
    """
    Get a 3x3 gradient around a point of interest

    :param blk_fixed: the fixed block of volume pixels
    :param blk_moving: the moving block of volume pixels
    :param pt: the center point of the correlation
    :param orig_fixed: the z, y and x coordinates of the origin of the
    fixed block
    :param orig_moving: the z, y and x coordinates of the origin of the
    moving block
    :param dx: the half-width of the volume to be measured in the x direction
    :param dy: the half-width of the volume to be measured in the y direction
    :param dz: the half-width of the volume to be measured in the z direction
    :param off_x: the x offset of the center for the moving volume
    :param off_y: the y offset of the center for the moving volume
    :param off_z: the z offset of the center for the moving volume
    :return: a 3x3x3 array of correlation coefficients centered at 1, 1, 1
    """
    c = np.zeros((3, 3, 3))
    for ix, inc_x in enumerate((-1, 0, 1)):
        for iy, inc_y in enumerate((-1, 0, 1)):
            for iz, inc_z in enumerate((-1, 0, 1)):
                c[iz, iy, ix] = corr(blk_fixed,
                                     blk_moving,
                                     pt,
                                     orig_fixed,
                                     orig_moving,
                                     dx, dy, dz,
                                     off_x + inc_x, off_y + inc_y,
                                     off_z + inc_z)
    return c


def register(amoving:ArrayReader,
             interpolator:typing.Callable[[np.ndarray], np.ndarray],
             x0:int, x1:int, y0:int, y1:int, z0:int, z1:int,
             pad=2):
    chunks = amoving.scale.chunk_sizes
    #
    # Get several grids.
    #    grid - coordinates in the fixed frame
    #    mgrid - coordinates in the moving frame
    #    tgrid - target coordinates in the target array
    #
    grid = np.mgrid[z0:z1, y0:y1, x0:x1]
    mgrid = interpolator(grid.transpose(1, 2, 3, 0).reshape(-1, 3))
    tgrid = (grid - np.array([z0, y0, x0]).reshape(3, 1, 1, 1))\
        .reshape(3, -1).transpose()
    target = np.zeros(grid.shape[1:], amoving.dtype)
    #
    # Mask out the coordinates that are OOB
    #
    mask = np.all(mgrid >= pad, 1) &\
           np.all(mgrid - pad < np.array([amoving.shape]))
    mmgrid = mgrid[mask]
    mtgrid = tgrid[mask]
    #
    # Retrieve the moving block encompassing the needed points
    #
    m0z, m0y, m0x = mmgrid.min(0).astype(int) - pad
    m1z, m1y, m1x = np.ceil(mmgrid.max(0)).astype(int) + pad
    moving_block = amoving[m0z:m1z, m0y:m1y, m0x:m1x]
    pixels = ndimage.map_coordinates(
        moving_block, 
        (mmgrid - np.array([[m0z, m0y, m0x]])).transpose())
    target[mtgrid[:, 0], mtgrid[:, 1], tgrid[:, 2]] = pixels
    return target


def follow_gradient(afixed:ArrayReader, amoving:ArrayReader,
                    interpolator:typing.Callable[[np.ndarray], np.ndarray],
                    pt:typing.Sequence[int],
                    dx:int,
                    dy:int,
                    dz:int,
                    padding_x:int,
                    padding_y:int,
                    padding_z:int,
                    max_rounds:int,
                    blur:typing.Tuple[float, float, float]) ->\
        typing.Sequence[int]:
    """
    Follow the gradient towards the highest corr coeff.
    Exit when at a local maximum, when we hit an edge, when
    we reach the maximum number of rounds or when we get stuck in a cycle.

    :param afixed: the array reader of the fixed volume
    :param amoving: the array reader of the moving volume
    :param interpolator: function to translate fixed to moving coors
    :param pt: the point to start at
    :param dx: the half-width of the volume on which to calculate the corr.
    :param dy: the half-width of the volume on which to calculate the corr.
    :param dz: the half-width of the volume on which to calculate the corr.
    :param padding_x: pad the moving array by this much to keep from
        endlessly retrieving
    :param padding_y: pad the moving array by this much
    :param padding_z: pad the moving array by this much
    :param max_rounds: exit w/o finding a maximum after this many rounds
    :param blur: blur both images by this sigma - larger blurs give more
    globally relevant maxima, smaller give more accuracy
    :return: local maximum coordinates of corr coeff.
    """
    block_moving = None
    x0m = x1m = y0m = y1m = z0m = z1m = 0
    last = np.array(pt).copy()
    x0f = pt[2] - dx - padding_x
    x1f = pt[2] + dx + padding_x
    y0f = pt[1] - dy - padding_y
    y1f = pt[1] + dy + padding_y
    z0f = pt[0] - dz - padding_z
    z1f = pt[0] + dz + padding_z
    orig_fixed = (z0f, y0f, x0f)
    block_fixed = ndimage.gaussian_filter(
        afixed[z0f:z1f, y0f:y1f, x0f:x1f].astype(np.float32),
        blur)
    def make_id(pt):
        return pt[2] + (pt[1] + pt[0] * afixed.shape[1]) * afixed.shape[2]
    pts_seen = set([make_id(pt)])
    for iround in range(max_rounds):
        x0mt = last[2] - dx
        x1mt = last[2] + dx + 1
        y0mt = last[1] - dy
        y1mt = last[1] + dy + 1
        z0mt = last[0] - dz
        z1mt = last[0] + dz + 1
        if x0mt < 0 or x1mt >= afixed.shape[2] or \
            y0mt < 0 or y1mt >= afixed.shape[1] or \
            z0mt < 0 or z1mt >= afixed.shape[0]:
            # At edge, give up
            return last

        if x0mt < x0m or x1mt >= x1m or \
            y0mt < y0m or y1mt >= y1m or \
            z0mt < z0m or z1mt >= z1m:
            x0m = max(0, x0mt - padding_x)
            x1m = min(x1mt + padding_x, afixed.shape[2])
            y0m = max(0, y0mt - padding_y)
            y1m = min(y1mt + padding_y, afixed.shape[1])
            z0m = max(0, z0mt - padding_z)
            z1m = min(z1mt + padding_z, afixed.shape[0])
            orig_moving = (z0m, y0m, x0m)

            block_moving = register(amoving, interpolator,
                                    x0m, x1m, y0m, y1m, z0m, z1m)
            block_moving = ndimage.gaussian_filter(
                block_moving.astype(np.float32), blur)
        c = gradient(block_fixed, block_moving,
                     pt,
                     orig_fixed, orig_moving,
                     dx, dy, dz,
                     last[2] - pt[2],
                     last[1] - pt[1],
                     last[0] - pt[0])
        cbest = np.max(c)
        off_z, off_y, off_x = [_[0] - 1 for _ in np.where(c == cbest)]
        if off_z == 0 and off_y == 0 and off_x == 0:
            return last, cbest
        last[0] += off_z
        last[1] += off_y
        last[2] += off_x
        last_id = make_id(last)
        if last_id in pts_seen:
            return last, cbest
        pts_seen.add(last_id)
    return last, cbest

def main():
    fixed_url = "file:///mnt/cephfs/users/lee/2020-06-26_mouse-cortex/round-2/alignment/r1c0_precomputed"
    moving_url = "file:///mnt/cephfs/users/lee/2020-06-26_mouse-cortex/round-2/alignment/r2c0_precomputed"
    interpolator_path = "/mnt/cephfs/users/lee/2020-06-26_mouse-cortex/round-2/alignment/fit-nonrigid-transform_round_10.pkl"
    with open(interpolator_path, "rb") as fd:
        interpolator = pickle.load(fd)["interpolator"]
    pt = np.array([273, 812, 83])[::-1]
    afixed = ArrayReader(fixed_url, format="blockfs")
    amoving = ArrayReader(moving_url, format="blockfs")
    moving_pt, corr = follow_gradient(afixed, amoving, interpolator, pt,
                                      20, 20, 10, 20, 20, 10, 100,
                                      [3, 3, 3])
    print(moving_pt[::-1], corr)

if __name__ == "__main__":
    main()
