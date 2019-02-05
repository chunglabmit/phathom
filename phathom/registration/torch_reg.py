import multiprocessing
import numpy as np
import zarr
from zarr import Blosc
import torch
import torch.nn.functional as F
from phathom.registration.registration import chunk_coordinates, interpolate
from phathom import utils
from tqdm import tqdm


class TorchGridSampler:

    def __init__(self, values):
        self.use_gpu = torch.cuda.is_available()
        if self.use_gpu:
            self.values = torch.from_numpy(values).float().cuda()
        else:
            self.values = torch.from_numpy(values).float()

    def __call__(self, grid):
        # if self.use_gpu:
        #     grid = torch.from_numpy(grid).float().cuda()
        # else:
        #     grid = torch.from_numpy(grid).float()

        values = F.grid_sample(self.values, grid)  # A tensor
        return values

        # if self.use_gpu:
        #     return values.cpu().numpy()
        # else:
        #     return values.numpy()


def fit_grid_sampler(values):

    interp_z = TorchGridSampler(values[0])
    interp_y = TorchGridSampler(values[1])
    interp_x = TorchGridSampler(values[2])
    return interp_z, interp_y, interp_x


def sample_grid(grid, interp, moving_shape, padding=2):
    interp_z, interp_y, interp_x = interp
    values_z = interp_z(grid)[0]  # (1, *chunks) tensor
    values_y = interp_y(grid)[0]
    values_x = interp_x(grid)[0]

    z_min = np.floor(values_z.min().cpu().numpy() - padding).astype(np.int)
    z_max = np.ceil(values_z.max().cpu().numpy() + padding).astype(np.int)
    y_min = np.floor(values_y.min().cpu().numpy() - padding).astype(np.int)
    y_max = np.ceil(values_y.max().cpu().numpy() + padding).astype(np.int)
    x_min = np.floor(values_x.min().cpu().numpy() - padding).astype(np.int)
    x_max = np.ceil(values_x.max().cpu().numpy() + padding).astype(np.int)
    transformed_start = np.array([z_min, y_min, x_min])
    transformed_stop = np.array([z_max, y_max, x_max])

    if np.any(transformed_stop < 0) or np.any(np.greater_equal(transformed_start, moving_shape)):
        return None
    else:
        start = np.array([max(0, s) for s in transformed_start])
        stop = np.array([min(e, s) for e, s in zip(moving_shape, transformed_stop)])
        if np.any(np.less_equal(stop - 1, start)):
            return None
        else:
            new_grid_x = (values_x - float(start[2])) / float(stop[2] - start[2] - 1) * 2 - 1
            new_grid_y = (values_y - float(start[1])) / float(stop[1] - start[1] - 1) * 2 - 1
            new_grid_z = (values_z - float(start[0])) / float(stop[0] - start[0] - 1) * 2 - 1
            new_grid = torch.stack([new_grid_x, new_grid_y, new_grid_z], dim=4)

            # new_grid = np.zeros((*values_z.shape, 3))
            # new_grid[..., 0] = (values_x - start[2]) / (stop[2] - start[2] - 1) * 2 - 1
            # new_grid[..., 1] = (values_y - start[1]) / (stop[1] - start[1] - 1) * 2 - 1
            # new_grid[..., 2] = (values_z - start[0]) / (stop[0] - start[0] - 1) * 2 - 1
            return start, stop, new_grid


def sample_grid_coords(grid, interp, padding=2):
    interp_z, interp_y, interp_x = interp
    values_z = interp_z(grid)[0]  # (1, *chunk_shape)
    values_y = interp_y(grid)[0]
    values_x = interp_x(grid)[0]
    coords = np.column_stack([values_z.ravel(), values_y.ravel(), values_x.ravel()])

    # Calculate the moving chunk bounding box
    z_tensor = torch.from_numpy(values_z).float().cuda()
    y_tensor = torch.from_numpy(values_y).float().cuda()
    x_tensor = torch.from_numpy(values_x).float().cuda()
    z_min = np.floor(z_tensor.min().cpu().numpy() - padding).astype(np.int)
    z_max = np.ceil(z_tensor.max().cpu().numpy() + padding).astype(np.int)
    y_min = np.floor(y_tensor.min().cpu().numpy() - padding).astype(np.int)
    y_max = np.ceil(y_tensor.max().cpu().numpy() + padding).astype(np.int)
    x_min = np.floor(x_tensor.min().cpu().numpy() - padding).astype(np.int)
    x_max = np.ceil(x_tensor.max().cpu().numpy() + padding).astype(np.int)

    start = np.array([z_min, y_min, x_min])
    stop = np.array([z_max, y_max, x_max])

    return coords, start, stop


def register_slice(moving_img, zslice, output_shape, values, fixed_shape, padding=2):
    """Apply transformation and interpolate for a single z slice in the output

    Parameters
    ----------
    moving_img : zarr array
        input image to be interpolated
    zslice : int
        index of the z-slice to compute
    output_shape : tuple
        size of the 2D output
    values : ndarray
        grid for the nonlinear registration
    fixed_shape : tuple
        shape of the fixed image in 3D, used for interpolating grid
    padding : int, optional
        amount of padding to use when extracting pixels for interpolation

    Returns
    -------
    registered_img : ndarray
        registered slice from the moving image

    """
    # Get dimensions
    img_shape = np.asarray((1, *output_shape))  # (z, y, x)
    local_indices = np.indices(img_shape)  # (3, z, y, x), first all zeros
    global_indices = local_indices
    global_indices[0] = zslice

    # Make the grid with normalized coordinates [-1, 1]
    # The last axis need to be in x, y, z order... but the others are in z, y, x
    grid = np.zeros((1, 1, *fixed_shape[1:], 3))  # (1, z, y, x, 3)
    grid[..., 0] = 2 * global_indices[2] / (fixed_shape[2] - 1) - 1
    grid[..., 1] = 2 * global_indices[1] / (fixed_shape[1] - 1) - 1
    grid[..., 2] = 2 * global_indices[0] / (fixed_shape[0] - 1) - 1

    # Sample the transformation grid
    interp = fit_grid_sampler(values)
    moving_coords, transformed_start, transformed_stop = sample_grid_coords(grid, interp, padding)

    # Read in the available portion data (not indexing outside the moving image boundary)
    moving_start = np.array([max(0, s) for s in transformed_start])
    moving_stop = np.array([min(e, s) for e, s in zip(moving_img.shape, transformed_stop)])
    moving_data = utils.extract_box(moving_img, moving_start, moving_stop)  # decompresses data from disk

    # interpolate the moving data
    moving_coords_local = moving_coords - np.array(moving_start)
    interp_values = interpolate(moving_data, moving_coords_local, order=1)
    registered_img = np.reshape(interp_values, output_shape)
    return registered_img


local_indices_default = None
moving_img_ram = None


def register_chunk(moving_img, fixed_img, output_img, values, start, chunks, padding=2):
    global local_indices_default
    global moving_img_ram

    # Get dimensions
    chunks = np.asarray(chunks)
    img_shape = np.asarray(output_img.shape)

    # Find the appropriate global stop coordinate and chunk shape accounting for boundary cases
    stop = np.minimum(start + chunks, img_shape)
    chunk_shape = np.array([b - a for a, b in zip(start, stop)])

    # Check the target to see if we need to do anything
    fixed_data = fixed_img[start[0]:stop[0],
                           start[1]:stop[1],
                           start[2]:stop[2]]
    if not np.any(fixed_data):
        output_img[start[0]:stop[0], start[1]:stop[1], start[2]:stop[2]] = np.zeros(chunk_shape, output_img.dtype)
        return

    # Get z, y, x indices for each pixel
    if np.all(chunks == chunk_shape):
        local_indices = local_indices_default  # expected chunk_shape, just reuse the default. (3, z, y, x)
    else:
        local_indices = np.indices(chunk_shape)  # pretty slow so only do this when we have to
    global_indices = np.empty_like(local_indices)  # faster than zeros_like
    for i in range(global_indices.shape[0]):
        global_indices[i] = local_indices[i] + start[i]
    global_indices = torch.from_numpy(global_indices).float().cuda()

    # Make the grid with normalized coordinates [-1, 1]
    grid_x = (global_indices[2] / float(img_shape[2] - 1)) * 2 - 1
    grid_y = (global_indices[1] / float(img_shape[1] - 1)) * 2 - 1
    grid_z = (global_indices[0] / float(img_shape[0] - 1)) * 2 - 1
    grid = torch.stack([grid_x, grid_y, grid_z], dim=3).unsqueeze(0)

    # Sample the transformation grid
    interp = fit_grid_sampler(values)
    result = sample_grid(grid, interp, moving_img.shape, padding)

    if result is not None:
        moving_start, moving_stop, moving_grid = result
        # Get the chunk of moving data
        # TODO: this is the next thing to optimize... could load into memory
        moving_data = utils.extract_box(moving_img, moving_start, moving_stop)  # decompresses data from disk
        # moving_data = utils.extract_box(moving_img_ram, moving_start, moving_stop)
        if not np.any(moving_data):
            interp_chunk = np.zeros(chunk_shape, output_img.dtype)
        else:
            # interpolate the moving data
            moving_data = moving_data.reshape((1, 1, *moving_data.shape)).astype(np.float)
            moving_data_tensor = torch.from_numpy(moving_data).float().cuda()
            interp_chunk = F.grid_sample(moving_data_tensor, moving_grid).cpu().numpy()[0, 0]
    else:
        interp_chunk = np.zeros(chunk_shape, output_img.dtype)

    # write results to disk
    output_img[start[0]:stop[0], start[1]:stop[1], start[2]:stop[2]] = interp_chunk


def _register_chunk(args):
    register_chunk(*args)


def register(moving_img, fixed_img, output_img, values, chunks, nb_workers, padding=2):
    global local_indices_default
    global moving_img_ram

    # Cache local indices
    local_indices_default = np.indices(chunks)

    # Load moving img
    # moving_img_ram = zarr.create(shape=moving_img.shape,
    #                              chunks=moving_img.chunks,
    #                              dtype=moving_img.dtype,
    #                              compressor=Blosc(cname='zstd', clevel=1, shuffle=Blosc.BITSHUFFLE))
    # zarr.copy_store(moving_img.store, moving_img_ram.store)
    # print(moving_img_ram.store)
    # moving_img_ram[:] = moving_img[:]
    # moving_img_ram[:] = moving_img

    print('here')

    start_coords = chunk_coordinates(output_img.shape, chunks)

    args_list = []
    for i, start_coord in tqdm(enumerate(start_coords), total=len(start_coords)):
        start = np.asarray(start_coord)
        args = (moving_img, fixed_img, output_img, values, start, chunks, padding)
        register_chunk(*args)

    # with multiprocessing.Pool(nb_workers) as pool:
        # pool.starmap(register_chunk, args_list)
        # list(tqdm(pool.imap(_register_chunk, args_list), total=len(args_list)))


def main():
    import os
    from phathom import io
    import matplotlib.pyplot as plt

    working_dir = '/media/jswaney/Drive/Justin/marmoset'

    # Open images
    fixed_zarr_path = 'round1/syto16.zarr/1_1_1'
    moving_zarr_path = 'round2/syto16.zarr/1_1_1'

    # Load the grid values and fit the interpolator
    grid_path = 'grid_values.npy'
    values_3d = np.load(os.path.join(working_dir, grid_path))
    values = np.zeros((values_3d.shape[0], 1, 1, *values_3d.shape[1:]))  # (3, 1, 1, gz, gy, gx)
    for i, v in enumerate(values_3d):
        values[i] = np.reshape(v, (1, 1, *v.shape))

    # Open the fixed and moving images
    fixed_img = io.zarr.open(os.path.join(working_dir, fixed_zarr_path), mode='r')
    moving_img = io.zarr.open(os.path.join(working_dir, moving_zarr_path), mode='r')

    # Create a new zarr array for the registered image
    nonrigid_zarr_path = 'round2/registered2/1_1_1'

    nonrigid_img = io.zarr.new_zarr(os.path.join(working_dir,
                                                 nonrigid_zarr_path),
                                    fixed_img.shape,
                                    fixed_img.chunks,
                                    fixed_img.dtype)

    # Register the moving image

    chunks = 3*(128,)
    nb_workers = 1
    padding = 2

    register(moving_img, fixed_img, nonrigid_img, values, chunks, nb_workers, padding)

    # zslice = 1000
    # output_shape = fixed_img.shape[1:]
    #
    # reg_slice = register_slice(moving_img, zslice, output_shape, values, fixed_img.shape, padding)
    #
    # fixed_slice = fixed_img[zslice]
    #
    # plt.figure(figsize=(6, 6))
    # plt.imshow(fixed_slice, cmap='Reds', clim=[0, 3000])
    # plt.imshow(reg_slice, cmap='Greens', clim=[0, 1000], alpha=0.5)
    # plt.show()


if __name__ == '__main__':
    main()