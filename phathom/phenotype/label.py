import os
import shutil
import tifffile
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.sampler import RandomSampler
from torchvision.datasets import DatasetFolder
from phathom.utils import make_dir, tifs_in_dir
import matplotlib.pyplot as plt
from tqdm import tqdm


working_dir = '/home/jswaney/pv_gabi'
npy_path = 'pv_patches.npy'
use_x11 = True


if use_x11:
    os.environ["QT_XKB_CONFIG_ROOT"] = "/usr/share/X11/xkb"  # For remote desktop keyboards


def load_data(path):
    return np.load(path)


class DataLabeler(object):

    def __init__(self, dataset, figsize=(4, 4)):
        self.dataset = dataset

        self.random_iter = iter(RandomSampler(self.dataset))
        self.idx = self.next_idx()

        self.fig = plt.figure(figsize=figsize)
        self.fig.canvas.mpl_connect('key_press_event', self.press)
        self.image = plt.imshow(self.get_image(self.idx))

        self.idx_true = []
        self.idx_false = []

    def next_idx(self):
        return next(self.random_iter)

    @staticmethod
    def show():
        plt.axis('off')
        plt.show()

    def get_image(self, idx):
        img = self.dataset[idx][0]
        return img

    def overwrite_image(self, img):
        self.image.set_data(img)
        self.image.autoscale()
        self.fig.canvas.draw()

    def press(self, event):
        if event.key == 'g':
            print("Index {}: True Positive".format(self.idx))
            self.idx_true.append(self.idx)
        elif event.key == 'b':
            print("Index {}: False Positive".format(self.idx))
            self.idx_false.append(self.idx)
        elif event.key == ' ':
            print("Skip")
        else:
            return
        self.idx = self.next_idx()
        self.overwrite_image(self.get_image(self.idx))


def numpy_arr_to_tiffs(arr, path, basename='', dtype=None):
    if dtype is None:
        dtype = arr.dtype
    abs_path = make_dir(path)
    for i, img in tqdm(enumerate(arr), total=len(arr)):
        filename = '{0}{1:07d}.tif'.format(basename, i)
        img_path = os.path.join(abs_path, filename)
        tifffile.imsave(img_path, img.astype(dtype))


def npy_to_tiffs(npy_path, output_path, basename='', dtype=None):
    data = load_data(npy_path)
    numpy_arr_to_tiffs(data, output_path, basename, dtype)


def tiff_loader(path):
    return tifffile.imread(path).astype(np.float32)


def move_file(src, dst):
    shutil.move(src, dst)


def move_files(files, dst):
    make_dir(dst)
    for src in tqdm(files, total=len(files)):
        move_file(src, dst)


def move_labeled_images(src, dst, true, false):
    paths, filenames = tifs_in_dir(src)
    paths_true = [paths[t] for t in true]
    paths_false = [paths[f] for f in false]
    dst_false = make_dir(os.path.join(dst, 'class_0'))
    dst_true = make_dir(os.path.join(dst, 'class_1'))
    move_files(paths_false, dst_false)
    move_files(paths_true, dst_true)


# TODO: Bring in code for semi-supervised learning with ADGMs
# TODO: Use SubsetRandomSampler to break off some labeled data for validation
# TODO: Try convolutional arcitechtures in the ADGMs


def main():
    # npy_path = os.path.join(working_dir, 'pv_patches.npy')
    # output_path = os.path.join(working_dir, 'unlabeled/class_0')
    # npy_to_tiffs(npy_path, output_path, basename='', dtype=None)

    dataset = DatasetFolder(root=os.path.join(working_dir, 'unlabeled'),
                            loader=tiff_loader,
                            extensions=['.tif'],
                            transform=None)

    labeler = DataLabeler(dataset)
    labeler.show()

    move_labeled_images(src=os.path.join(working_dir, 'unlabeled/class_0'),
                        dst=os.path.join(working_dir, 'labeled'),
                        true=labeler.idx_true,
                        false=labeler.idx_false)




if __name__ == '__main__':
    main()