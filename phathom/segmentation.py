from . import partition
from . import graphcuts
from . import utils
from . import imageio
import numpy as np
import scipy.ndimage as ndi
import skimage
from skimage import segmentation, feature, morphology, filters, exposure, transform, measure
import matplotlib.pyplot as plt
import multiprocessing
import os.path
import shutil


# Set default colormap to grayscale
plt.gray()

# Set consistent numpy random state
np.random.seed(123)

# Initialize the output list
outputs = []


def gradient(data):
    fz, fy, fx = np.gradient(data, edge_order=2)
    gradient = np.zeros((*data.shape, 3))
    gradient[:,:,:,0] = fz
    gradient[:,:,:,1] = fy
    gradient[:,:,:,2] = fx
    return gradient


def hessian_2d(data):
    l1 = np.zeros(data.shape)
    for i in range(data.shape[0]):
        img = data[i,:,:]
        Hyy, Hyx, Hxx = skimage.feature.hessian_matrix(img, sigma=1, mode='reflect')
        l1[i,:,:], l2 = skimage.feature.hessian_matrix_eigvals(Hyy, Hyx, Hxx)


def hessian(data):
    grad = np.gradient(data)

    n_dims = len(grad)
    H = np.zeros((*data.shape, n_dims, n_dims))
    for i, first_deriv in enumerate(grad):
        for j in range(i, n_dims):
            second_deriv = np.gradient(first_deriv, axis=j)
            H[:,:,:,i,j] = second_deriv
            if i != j:
                H[:,:,:,j,i] = second_deriv

    return H


def structure_tensor(grad):
    S = np.zeros((*grad.shape, 3))
    fz = grad[:,:,:,0]
    fy = grad[:,:,:,1]
    fx = grad[:,:,:,2]
    S[:,:,:,0,0] = fz**2
    S[:,:,:,1,1] = fy**2
    S[:,:,:,2,2] = fx**2
    S[:,:,:,0,1] = fz*fy
    S[:,:,:,0,2] = fz*fx
    S[:,:,:,1,2] = fy*fx
    S[:,:,:,1,0] = fz*fy
    S[:,:,:,2,0] = fz*fx
    S[:,:,:,2,1] = fy*fx
    return S


def add_output(name, data):
    outputs.append({'name': name, 'data': data})


def write_output():
    if outputs:
        for d in outputs:
            name = d['name']
            print('saving {}'.format(name))
            imageio.imsave('../data/{}.tif'.format(d['name']), d['data'].astype('float32'))


def weingarten(g):
    grad = gradient(g)
    l = np.sqrt(1+np.linalg.norm(grad, axis=-1))
    H = hessian(g)
    S = structure_tensor(grad)
    L = np.zeros(S.shape)
    for i in range(3):
        for j in range(3):
            L[:,:,:,i,j] = l
    eye = np.zeros(S.shape)
    eye[:,:,:,0,0] = 1
    eye[:,:,:,1,1] = 1
    eye[:,:,:,2,2] = 1
    B = S+eye
    A = H*np.linalg.inv(B)/L
    return A


def eigvalsh(A):
    return np.linalg.eigvalsh(A)


def convex_seeds(eigvals):
    eig_neg = np.clip(eigvals, None, 0)
    det_neg = eig_neg.prod(axis=-1)
    return det_neg


def positive_curvatures(eigvals):
    eig_pos = np.clip(eigvals, 0, None)
    pos_curv = np.linalg.norm(eig_pos, axis=-1)
    return pos_curv


def seed_probability(eigen_seeds):
    return 1-np.exp(-eigen_seeds**2/(2*eigen_seeds.std()**2))


def regionprops(intensity_img, labels_img):
    # Get region statistics
    # print('Getting region statistics')
    verbose = False
    nb_regions = labels_img.max()
    properties = measure.regionprops(labels_img, intensity_image=intensity_img)
    centroids = np.zeros((nb_regions, 3))
    for i, region in enumerate(properties):
        # Shape based
        zc, yc, xc = region.centroid
        nb_vox = region.area
        # volume = nb_vox*voxel_volume
        eq_diam = region.equivalent_diameter
        princple_interia = region.inertia_tensor_eigvals
        inertia_ratio = princple_interia[0]/max(1e-6, princple_interia[1])
        # Intensity-based
        max_intensity = region.max_intensity
        mean_intensity = region.mean_intensity
        min_intensity = region.min_intensity
        # Tabulate data
        centroids[i] = region.centroid
        if verbose:
            print(30*'-')
            print(f'Nucleus {i:d} at ({int(zc):d}, {int(yc):d}, {int(xc):d})')
            print(f'    Shape-based measurements')
            print(f'    Volume [vox]: {nb_vox:d}')
            print(f'    Volume [um3]: {volume:.1f}')
            print(f'    Equivalent Diameter [um]: {eq_diam:.2f}')
            print(f'    Principle moments of inertia: ({princple_interia[0]:.3f}, {princple_interia[1]:.3f}, {princple_interia[2]:.3f})')
            print(f'    Ratio of principle moments of intertia: {inertia_ratio:.3f}')
            print(f'    Intensity-based measurements')
            print(f'    Max intensity: {max_intensity:.3f}')
            print(f'    Mean intensity: {mean_intensity:.3f}')
            print(f'    Min intensity: {min_intensity:.3f}')
    return None


def find_centroids(labels_img):
    nb_regions = labels_img.max()
    properties = measure.regionprops(labels_img)
    centroids = np.zeros((nb_regions, 3))
    for i, region in enumerate(properties):
        centroids[i] = region.centroid
    return centroids


def segment(input_file, output_paths, upsample, sigma, h, T):
    # # Calculate the voxel dimensions after upsampling
    # voxel_shape = np.array([d/s for d, s in zip(orig_vox_dim, upsampling_scale)])
    # print(f'Voxel dimensions [um]: {voxel_shape}')
    # voxel_volume = voxel_shape.prod()
    # print(f'Voxel volume [um3]: {voxel_volume:.3f}')

    # Load the dataset
    # print('Loading the dataset')
    data = imageio.imread(input_file)

    # Contrast stretching
    # print('Normalizing the image')
    # data = exposure.rescale_intensity(data, in_range=(0, 2**12-1))

    # Convert to float32
    # print('Converting to float32')
    data = skimage.img_as_float32(data)

    # Upsampling
    # print('Upsampling the image')
    data = transform.rescale(data, scale=upsample, order=3, multichannel=False, mode='reflect', anti_aliasing=True)
    # add_output('upsampled', data)

    # Extract foreground
    # print('Extracting foreground mask')
    # mask = graphcuts.run_graph_cuts(data, lambdaI=0, lambda_const=2)
    # T = max(minT, filters.threshold_otsu(data))
    mask = data > T

    # Save the foregound image
    # imageio.imsave(file=output_paths['foreground_path'], data=mask.astype('uint8'))

    # Gaussian smoothing
    # print('Smoothing the image')
    g = filters.gaussian(data, sigma=sigma)
    # add_output('g', g)

    # Calculate the Weingarten operator
    # print('Calculating the Weingarten shape operator')
    A = weingarten(g)

    # print('Calculating eigenvalues')
    eigvals = eigvalsh(A)
    eigen_seeds = convex_seeds(eigvals)
    pos_curv = positive_curvatures(eigvals)
    seed_prob = seed_probability(eigen_seeds)*mask # filters out background seeds

    # H-maxima transform
    # print('Performing H-maxima transform')
    hmax = morphology.reconstruction(seed_prob-h, seed_prob, 'dilation')
    extmax = seed_prob - hmax
    seeds = (extmax > 0)*mask # Apply mask here to avoid a trivial background seed

    # Label the detected maxima
    # print('Labeling the detected seeds')
    structure = morphology.cube(width=3) # 28-connected
    markers = ndi.label(seeds, structure=structure)[0]
    # nb_regions = markers.max()
    # print('Number of cells detected: {}'.format(nb_regions))

    # print('Performing seeded watershed segmentation')
    seg = morphology.watershed(pos_curv, markers, mask=mask, watershed_line=True)

    # print('Removing objects on the border')
    # seg = segmentation.clear_border(seg, buffer_size=4)

    # print('Converting to binary mask')
    binary_seg = seg > 0

    # print('Saving image')
    imageio.imsave(path=output_paths['segmentation_path'], data=binary_seg.astype('uint8'))

    # Calculate the distance transform
    dist_map = ndi.morphology.distance_transform_edt(binary_seg)

    # Save the distance transform
    # imageio.imsave(file=output_paths['distance_path'], data=dist_map.astype('float32'))

    # Calculate the normalized gradient of the distance map
    grad = gradient(dist_map)
    norm = np.linalg.norm(grad, axis=-1)
    norm_grad = grad / (1e-6 + np.reshape(norm, (*norm.shape, 1)))

    # Save the direction map
    # imageio.imsave(file=output_paths['direction_path'], data=norm_grad.astype('float32'))


def segment_chunks(working_dir, nb_workers, upsampling, sigma, h, T):
    input_dir = os.path.join(working_dir, 'input/')
    foreground_dir = os.path.join(working_dir, 'foreground/')
    direction_dir = os.path.join(working_dir, 'direction/')
    distance_dir = os.path.join(working_dir, 'distance/')
    segmentation_dir = os.path.join(working_dir, 'segmentation/')

    foregorund_abspath = utils.make_dir(foreground_dir)
    direction_abspath = utils.make_dir(direction_dir)
    distance_abspath = utils.make_dir(distance_dir)
    segmentation_abspath = utils.make_dir(segmentation_dir)

    input_paths, input_files = utils.tifs_in_dir(input_dir)
    args = []
    for i, (input_path, input_file) in enumerate(zip(input_paths, input_files)):
        output_paths = {
            'foreground_path': os.path.join(foregorund_abspath, input_file),
            'direction_path': os.path.join(direction_abspath, input_file),
            'distance_path': os.path.join(distance_abspath, input_file),
            'segmentation_path': os.path.join(segmentation_abspath, input_file),
        }
        args.append((input_path, output_paths, upsampling, sigma, h, T))

    with multiprocessing.Pool(processes=nb_workers) as pool:
        pool.starmap(segment, args)

    # Copy over the metadata
    shutil.copy(os.path.join(input_dir, 'metadata.pkl'), segmentation_abspath)

    print('Done!')


def main():
    # input_path = '../data/spim_crop/input/z00000_y00000_x00000.tif'
    # output_path = '../data/spim_crop/test.tif'
    upsampling = (1, 1, 1)
    sigma = (1.5, 1.5, 1.5)
    h = 0.01
    T = 0.01
    # segment(input_path, output_path, upsampling, sigma, h, minT)

    working_dir = '../data/control/'
    nb_workers = 8
    # segment_chunks(working_dir, nb_workers, upsampling, sigma, h, T)

    centroids_file = '../data/control/centroids.npy'
    binary_seg = partition.combine_chunks(os.path.join(working_dir, 'segmentation/'))
    labeled_seg = ndi.label(binary_seg)[0]
    centroids = find_centroids(labeled_seg)
    np.save(file=centroids_file, arr=centroids)


if __name__ == '__main__':
    main()
