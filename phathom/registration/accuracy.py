"""
This module provides code to quantify the registration accuracy between rounds.

Matching accuracy = fraction of correct matches in a set
Keypoint residuals = distances to matching nuclei remaining after alignment
Keypoint distances = distances to nearest keypoint in a given point cloud
Deadzones = heatmap of keypoint distances over the image domain

Matching accuracy:
    Need Y/N annotation on matches
    Could use simple matplotlib / widget-based GUI
        Should take two images with two sets of matching coordinates
        Should write to a file as the annotations come in or on keypress

Keypoint residuals:
    Easy using np.llinalg.norm after warping the points

Keypoint distances:
    Need to calculate distances using sklearn neighbors
    Given a set of coordinates and indices of the matched points
        Calculate distance to nearest match

Deadzones:
    Need to turn scattered keypoint distances into images
    Given a set of coordinates, associated keypoint distances, and an output shape
        Use griddata to interpolate the data to an image with output shape
        Use nearest interpolation to be able to extrapolate beyond the convex hull

"""

import os
import numpy as np
from sklearn.neighbors import NearestNeighbors
import tifffile
from phathom.registration import registration as reg
from phathom import io
from phathom.utils import extract_box, pickle_load, pickle_save

import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from tqdm import tqdm

os.environ['QT_XKB_CONFIG_ROOT'] = "/usr/share/X11/xkb"


def nuclei_centered_box(arr, coord, patch_width, patch_height):
    start = coord - np.array([patch_height // 2, patch_width // 2, patch_width // 2])
    stop = start + np.array([patch_height, patch_width, patch_width])
    return extract_box(arr, start, stop)


def box_mip(box, projection, mode='xy'):
    if mode not in ['xy', 'xz']:
        raise ValueError('Only xy and xz projections supported')
    patch_width = box.shape[-1]
    patch_height = box.shape[0]
    if mode == 'xy':
        mip = box[patch_height // 2 - projection // 2:patch_height // 2 + projection // 2].max(axis=0)
    elif mode == 'xz':
        mip = box[:, patch_width // 2 - projection // 2:patch_width // 2 + projection // 2].max(axis=1)
    return mip


def nuclei_centered_mip(arr, coord, patch_width, patch_height, projection, mode='xy'):
    box = nuclei_centered_box(arr, coord, patch_width, patch_height)
    return box_mip(box, projection, mode)


class MatchingAccuracy(object):

    def __init__(self, nb_matches):
        self.idx = 0
        self.order = np.arange(nb_matches)
        np.random.shuffle(self.order)
        self.correct = []
        self.incorrect = []

    @property
    def nb_labeled(self):
        return len(self.correct) + len(self.incorrect)

    @property
    def accuracy(self):
        if self.nb_labeled > 0:
            return len(self.correct) / self.nb_labeled
        else:
            return None

    def add_correct(self, match_idx):
        self.correct.append(match_idx)

    def add_incorrect(self, match_idx):
        self.incorrect.append(match_idx)

    def next_match_idx(self):
        match_idx = self.order[self.idx]
        self.idx += 1
        return match_idx

    def save(self, path):
        pickle_save(path, self)

    def __repr__(self):
        return "Matching accuracy: {0:3.2f} ({1:d} labeled)".format(self.accuracy, self.nb_labeled)


def manual_accuracy(fixed_blobs, moving_blobs, matches_fixed, matches_moving, patch_width, patch_height, projection,
                    arr_fixed, arr_moving, arr_reg=None):

    fig, (ax_fixed, ax_moving, ax_reg) = plt.subplots(ncols=3)
    match_acc = MatchingAccuracy(len(matches_fixed))

    class Callback(object):

        def __init__(self, arr_fixed, arr_moving, arr_reg, fig, match_acc):
            self.idx = match_acc.next_match_idx()
            self.arr_fixed = arr_fixed
            self.arr_moving = arr_moving
            self.arr_reg = arr_reg
            self.fig = fig
            self.match_acc = match_acc

            fixed = nuclei_centered_mip(self.arr_fixed, fixed_blobs[matches_fixed[self.idx]], patch_width, patch_height,
                                           projection)
            if arr_reg is not None:
                reg = nuclei_centered_mip(self.arr_reg, fixed_blobs[matches_fixed[self.idx]], patch_width, patch_height,
                                             projection)

            moving = nuclei_centered_mip(self.arr_moving, moving_blobs[matches_moving[self.idx]], patch_width,
                                            patch_height, projection)

            self.img_fixed = ax_fixed.imshow(fixed)
            self.img_moving = ax_moving.imshow(moving)

            if arr_reg is not None:
                self.img_reg = ax_reg.imshow(reg, cmap='Greens')
                self.img_reg_overlay = ax_reg.imshow(fixed, cmap='Reds', alpha=0.5)

            ax_fixed.scatter(patch_width // 2 - 1, patch_width // 2 - 1, s=500, facecolors='none', edgecolors='b')
            ax_moving.scatter(patch_width // 2 - 1, patch_height // 2 - 1, s=500, facecolors='none', edgecolors='b')
            ax_reg.scatter(patch_width // 2 - 1, patch_height // 2 - 1, s=500, facecolors='none', edgecolors='b')

        def update_images(self, fixed, moving, reg):
            self.img_fixed.set_data(fixed)
            self.img_fixed.autoscale()
            self.img_moving.set_data(moving)
            self.img_moving.autoscale()
            if reg is not None:
                self.img_reg.set_data(reg)
                self.img_reg.autoscale()
                self.img_reg_overlay.set_data(fixed)
                self.img_reg_overlay.autoscale()
            self.fig.canvas.draw()

        def press(self, event):
            if event.key == 'g':
                self.match_acc.add_correct(self.idx)
                print(self.match_acc)
            elif event.key == 'b':
                self.match_acc.add_incorrect(self.idx)
                print(self.match_acc)
            else:
                return
            self.idx = self.match_acc.next_match_idx()
            fixed = nuclei_centered_mip(self.arr_fixed, fixed_blobs[matches_fixed[self.idx]], patch_width, patch_height, projection)
            if self.arr_reg is not None:
                reg = nuclei_centered_mip(self.arr_reg, fixed_blobs[matches_fixed[self.idx]], patch_width, patch_height, projection)
            else:
                reg = None
            moving = nuclei_centered_mip(self.arr_moving, moving_blobs[matches_moving[self.idx]], patch_width, patch_height, projection)
            self.update_images(fixed, moving, reg)

    callback = Callback(arr_fixed, arr_moving, arr_reg, fig, match_acc)
    fig.canvas.mpl_connect('key_press_event', callback.press)
    plt.show()

    return match_acc


def distance_map(fixed_blobs, matches_fixed, voxel_dim, x, y, z, max_dist=1000):
    fixed_keypts = fixed_blobs[matches_fixed]
    fixed_keypts_um = fixed_keypts * np.asarray(voxel_dim)
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree', n_jobs=-1).fit(fixed_keypts_um)

    dist = np.zeros((len(z), len(y), len(x)), np.float32)
    for i, zi in tqdm(enumerate(z), total=len(z)):
        Z, Y, X = np.meshgrid(zi, y, x, indexing='ij')
        grid = np.column_stack([Z.ravel(), Y.ravel(), X.ravel()])
        pts = grid * np.asarray(voxel_dim)
        distances, indices = nbrs.kneighbors(pts)
        distance_img = np.reshape(distances, Z.shape)[0]
        distance_img[np.where(distance_img > max_dist)] = 0
        dist[i] = distance_img
    return dist


def main():
    working_dir = '/home/jswaney/org_registration'
    fixed_dir = 'round1'
    moving_dir = 'round2'

    projection = 8
    patch_height = 128
    patch_width = 128

    fixed_blobs = np.load(os.path.join(working_dir, fixed_dir, 'blobs.npy'))
    moving_blobs = np.load(os.path.join(working_dir, moving_dir, 'blobs.npy'))
    print("Loaded {} fixed points".format(fixed_blobs.shape[0]))
    print("Loaded {} moving points".format(moving_blobs.shape[0]))

    matches_fixed = np.load(os.path.join(working_dir, fixed_dir, 'match_idx.npy'))
    matches_moving = np.load(os.path.join(working_dir, moving_dir, 'match_idx.npy'))
    print("Loaded {} matching indices".format(matches_fixed.size))

    arr_fixed = io.zarr.open(os.path.join(working_dir, fixed_dir, 'syto16.zarr', '1_1_1'))
    arr_moving = io.zarr.open(os.path.join(working_dir, moving_dir, 'syto16.zarr', '1_1_1'))
    # arr_reg = io.zarr.open(os.path.join(working_dir, moving_dir, 'registered', '1_1_1'))
    arr_reg = None
    print("Opened fixed image with shape {}".format(arr_fixed.shape))
    print("Opened moving image with shape {}".format(arr_moving.shape))
    if arr_reg is not None:
        print("Opened registered image with shape {}".format(arr_reg.shape))

    print("Starting GUI for manual assessment of matching accuracy")
    match_acc = manual_accuracy(fixed_blobs, moving_blobs, matches_fixed, matches_moving, patch_width, patch_height,
                                projection, arr_fixed, arr_moving, arr_reg)
    match_acc.save(os.path.join(working_dir, 'match_acc.pkl'))

    #
    # step = 20
    # z = 3000
    # voxel_dim = (2.0, 1.8, 1.8)
    #
    # z = np.arange(0, arr_fixed.shape[0], step)
    # y = np.arange(0, arr_fixed.shape[1], step)
    # x = np.arange(0, arr_fixed.shape[2], step)

    # dist = distance_map(fixed_blobs, matches_fixed, voxel_dim, x, y, z, max_dist=1000)
    # tifffile.imsave(os.path.join(working_dir, 'dist.tif'), dist)

    # match_img = np.zeros((len(z), len(y), len(x)), np.uint8)
    # for pt in fixed_blobs[matches_fixed]:
    #     pt_scaled = np.round(pt / step).astype(np.int)
    #     match_img[pt_scaled[0], pt_scaled[1], pt_scaled[2]] = 255
    #
    # tifffile.imsave(os.path.join(working_dir, 'matches.tif'), match_img)

    # plt.imshow(arr_fixed[z, 0:arr_fixed.shape[1]:step, 0:arr_fixed.shape[2]:step], cmap='gray')
    # plt.imshow(distance_img, alpha=0.5, clim=[0, 1000])
    # plt.show()

    # loc = np.where(np.abs(fixed_keypts[:, 0]-z) < 200)[0]
    # fixed_keypts_near = fixed_keypts[loc]
    # plt.scatter(fixed_keypts_near[:, 2]/step, fixed_keypts_near[:, 1]/step, s=5)
    # plt.imshow(distance_img, clim=[0, 500])
    # plt.show()




if __name__ == '__main__':
    print('Running accuracy.py as main')
    main()
