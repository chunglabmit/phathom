from skimage import measure
import tifffile
import os
import numpy as np
from phathom.preprocess.filtering import gaussian_blur
from mayavi import mlab


working_dir = '/media/jswaney/SSD EVO 860/organoid_phenotyping/20181206_eF9_A34_1'
segmentation_path = 'syto16_4x_downsampled_vzseg_8bit.tif'
centers_path = 'centers.npy'
sox2_labels_path = 'sox2_labels.npy'
tbr1_labels_path = 'tbr1_labels.npy'
voxel_size = (2.052, 1.082, 1.082)
spacing = (2.052, 1.082 * 4, 1.082 * 4)
sigma = 2
level = 1  # Isosurface intensity
step_size = 1  # Larger gives coarse mesh
opacity = 0.3
nb_vectors = 5000
nb_nuclei = 100000
scale_factor = 8


vz_seg = tifffile.imread(os.path.join(working_dir, segmentation_path))
vz_binary = (vz_seg > 0)
vz_seg_smooth = 10*gaussian_blur(vz_binary, sigma=sigma)
print(vz_seg_smooth.shape, vz_seg_smooth.dtype, vz_seg_smooth.max())

centers = np.load(os.path.join(working_dir, centers_path))
sox2_labels = np.load(os.path.join(working_dir, sox2_labels_path))
tbr1_labels = np.load(os.path.join(working_dir, tbr1_labels_path))

verts, faces, normals, values = measure.marching_cubes_lewiner(vz_seg_smooth,
                                                               level=level,
                                                               spacing=spacing,
                                                               step_size=step_size,
                                                               allow_degenerate=False)
print(normals.shape)

mlab.triangular_mesh([vert[0] for vert in verts],
                     [vert[1] for vert in verts],
                     [vert[2] for vert in verts],
                     faces,
                     color=(1, 0, 0))

idx = np.arange(verts.shape[0])
np.random.shuffle(idx)
idx = idx[:nb_vectors]

verts_sample = verts[idx]
normals_sample = normals[idx]

mlab.quiver3d(verts_sample[:, 0], verts_sample[:, 1], verts_sample[:, 2],
              normals_sample[:, 0], normals_sample[:, 1], normals_sample[:, 2],
              color=(1, 1, 0), opacity=opacity)


idx = np.arange(centers.shape[0])
np.random.shuffle(idx)
idx = idx[:nb_nuclei]

centers_sample = np.asarray(voxel_size) * centers[idx]
sox2_labels_sample = sox2_labels[idx]
tbr1_labels_sample = tbr1_labels[idx]
negative_idx = np.where(np.logical_and(sox2_labels_sample == 0, tbr1_labels_sample == 0))[0]
sox2_idx = np.where(np.logical_and(sox2_labels_sample > 0, tbr1_labels_sample == 0))[0]
tbr1_idx = np.where(np.logical_and(sox2_labels_sample == 0, tbr1_labels_sample > 0))[0]

negative = centers_sample[negative_idx]
sox2 = centers_sample[sox2_idx]
tbr1 = centers_sample[tbr1_idx]

mlab.points3d(negative[:, 0], negative[:, 1], negative[:, 2], scale_factor=scale_factor, color=(0, 0, 1))
mlab.points3d(sox2[:, 0], sox2[:, 1], sox2[:, 2], scale_factor=scale_factor, color=(1, 0, 0))
mlab.points3d(tbr1[:, 0], tbr1[:, 1], tbr1[:, 2], scale_factor=scale_factor, color=(0, 1, 0))

mlab.show()
