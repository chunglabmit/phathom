import numpy as np
import torch
import torch.nn.functional as F
from phathom.registration.registration import chunk_coordinates, interpolate
from tqdm import tqdm


class TorchGridSampler:

    def __init__(self, values):
        self.use_gpu = torch.cuda.is_available()
        if self.use_gpu:
            self.values = torch.from_numpy(values).float().cuda()
        else:
            self.values = torch.from_numpy(values).float()

    def __call__(self, grid):
        if self.use_gpu:
            grid = torch.from_numpy(grid).float().cuda()
        else:
            grid = torch.from_numpy(grid).float()

        values = F.grid_sample(self.values, grid)

        if self.use_gpu:
            return values.cpu().numpy()
        else:
            return values.numpy()


def fit_grid_sampler(values):
    interp_z = TorchGridSampler(values[0])
    interp_y = TorchGridSampler(values[1])
    interp_x = TorchGridSampler(values[2])
    return interp_z, interp_y, interp_x


def sample_grid(grid, interp):
    interp_z, interp_y, interp_x = interp
    values_z = interp_z(grid)
    values_y = interp_y(grid)
    values_x = interp_x(grid)
    return np.column_stack([values_z.ravel(), values_y.ravel(), values_x.ravel()])


def register_chunk(moving_img, output_img, interp, start, chunks, padding=2):
    # Get dimensions
    chunks = np.asarray(chunks)
    img_shape = np.asarray(output_img.shape)

    # Find the appropriate global stop coordinate and chunk shape accounting for boundary cases
    stop = np.minimum(start + chunks, img_shape)
    chunk_shape = np.array([b - a for a, b in zip(start, stop)])

    # Get z, y, x indices for each pixel
    local_indices = np.indices(chunk_shape)  # (3, z, y, x)
    global_indices = np.zeros_like(local_indices)
    for i in range(global_indices.shape[0]):
        global_indices[i] = local_indices[i] + start[i]

    # print(local_indices[:, 0,0,0])
    # print(start)
    # print(global_indices[:, 0,0,0])

    # Make the grid with normalized coordinates [-1, 1]
    grid = np.zeros((1, *chunk_shape, 3))
    for i in range(grid.shape[-1]):
        grid[..., i] = (global_indices[i] / (img_shape[i] - 1)) * 2 - 1  # Normalize

    # Sample the transformation grid
    moving_coords = sample_grid(grid, interp)

    # Find the padded bounding box of the warped chunk coordinates
    transformed_start = tuple(np.floor(moving_coords.min(axis=0) - padding).astype('int'))
    transformed_stop = tuple(np.ceil(moving_coords.max(axis=0) + padding).astype('int'))

    if np.any(np.asarray(transformed_stop) < 0) or \
            np.any(np.greater(np.asarray(transformed_start), img_shape)):  # Completely outside for some dimension
        interp_chunk = np.zeros(chunk_shape, output_img.dtype)
    else:
        # Read in the available portion data (not indexing outside the moving image boundary)
        moving_start = tuple(max(0, s) for s in transformed_start)
        moving_stop = tuple(min(e, s) for e, s in zip(moving_img.shape, transformed_stop))
        moving_coords_local = moving_coords - np.array(moving_start)
        moving_data = moving_img[moving_start[0]:moving_stop[0],
                                 moving_start[1]:moving_stop[1],
                                 moving_start[2]:moving_stop[2]]
        if not np.any(moving_data):  # No need to interpolate if moving image is just zeros
            interp_chunk = np.zeros(chunk_shape, dtype=output_img.dtype)
        else:
            # interpolate the moving data
            interp_values = interpolate(moving_data, moving_coords_local, order=1)
            interp_chunk = np.reshape(interp_values, chunk_shape)

    # write results to disk
    output_img[start[0]:stop[0], start[1]:stop[1], start[2]:stop[2]] = interp_chunk


def register(moving_img, output_img, interp, padding=2):

    start_coords = chunk_coordinates(output_img.shape, output_img.chunks)
    for i, start_coord in tqdm(enumerate(start_coords)):
        start = np.asarray(start_coord)
        args = (moving_img, output_img, interp, start, output_img.chunks, padding)
        register_chunk(*args)


def main():
    import os
    from phathom import io

    working_dir = '/media/jswaney/Drive/Justin/coregistration/whole_brain_tde'

    # Open images
    fixed_zarr_path = 'fixed/zarr_zstd/1_1_1'
    moving_zarr_path = 'moving/zarr_zstd/1_1_1'

    # Load the grid values
    grid_path = 'grid_values.npy'
    values_3d = np.load(os.path.join(working_dir, grid_path))

    values = np.zeros((values_3d.shape[0], 1, 1, *values_3d.shape[1:]))
    for i, v in enumerate(values_3d):
        values[i] = np.reshape(v, (1, 1, *v.shape))

    interp = fit_grid_sampler(values)

    fixed_img = io.zarr.open(os.path.join(working_dir, fixed_zarr_path), mode='r')
    moving_img = io.zarr.open(os.path.join(working_dir, moving_zarr_path), mode='r')

    # Create a new zarr array for the registered image
    nonrigid_zarr_path = 'moving/registered2/1_1_1'

    nonrigid_img = io.zarr.new_zarr(os.path.join(working_dir,
                                                 nonrigid_zarr_path),
                                    fixed_img.shape,
                                    fixed_img.chunks,
                                    fixed_img.dtype)

    register(moving_img, nonrigid_img, interp, padding=2)


if __name__ == '__main__':
    main()