from phathom.io import conversion as imageio
from phathom import utils
import numpy as np
# from skimage.filters import threshold_otsu
from skimage.util import crop
import skimage
import os.path
from operator import itemgetter


# Set consistent numpy random state
np.random.seed(123)


def random_chunks(img_path, seg_path, output_dir, nb_samples, shape, padding):
    """
    Extract random foreground-containing chunks from a large image.
    The large input image is loaded into memory.
    """
    img_name, _ = os.path.splitext(os.path.basename(img_path))
    utils.make_dir(output_dir)
    patch_dir = os.path.join(output_dir, 'patch')
    seg_dir = os.path.join(output_dir, 'seg')

    utils.make_dir(patch_dir)
    utils.make_dir(seg_dir)

    padded_shape = tuple(int(s+2*p) for s,p in zip(patch_shape, padding))

    img = imageio.imread(img_path)
    seg = imageio.imread(seg_path)

    binary_seg = (seg > 0).astype('float32') # Boolean array

    # Random sample corresponds to top-left of the padded patch
    # Copy=False gives discontiguous view of original data
    crop_width = tuple(zip((0,0,0), (s for s in padded_shape)))
    cropped_img = crop(img, crop_width=crop_width, copy=False)

    # Only sample patches with foreground pixels
    T = threshold_otsu(img)

    fgrd_idx = np.where(cropped_img>=T)
    nfgrd = len(fgrd_idx[0])

    sample_idx = np.random.randint(low=0, high=nfgrd, size=random_samples)
    for i, s in enumerate(sample_idx):
        z1 = fgrd_idx[0][s]
        y1 = fgrd_idx[1][s]
        x1 = fgrd_idx[2][s]
        z2 = z1+padded_shape[0]
        y2 = y1+padded_shape[1]
        x2 = x1+padded_shape[2]

        # Save img chunk
        padded_patch = img[z1:z2, y1:y2, x1:x2]
        patch_filename = '{0}_patch_{1:04}.tif'.format(img_name, i)
        patch_path = os.path.join(patch_dir, patch_filename)
        imageio.imsave(path=patch_path, data=padded_patch)

        padded_seg = binary_seg[z1:z2, y1:y2, x1:x2]
        seg_filename = '{0}_seg_{1:04}.tif'.format(img_name, i)
        seg_path = os.path.join(seg_dir, seg_filename)
        imageio.imsave(path=seg_path, data=padded_seg)


def sample_ground_truth(output_dir, input_dir, grdtruth_dir, nb_train, nb_val):
    """
    Sample pairs of input and outputs from expert system pipeline
    """
    input_paths, input_filenames = utils.tifs_in_dir(input_dir)
    grdtruth_paths, grdtruth_filenames = utils.tifs_in_dir(grdtruth_dir)

    if input_filenames != grdtruth_filenames:
        print('The images in {} and {} do not match'.format(input_dir, grdtruth_dir))
        return

    nb_samples = int(nb_train + nb_val)

    idx = np.array(range(len(input_paths)))
    rand_idx = np.random.choice(idx, size=nb_samples, replace=False)
    train_idx = rand_idx[:nb_train]
    val_idx = rand_idx[nb_train:]

    sorted_train_idx = sorted(train_idx)
    input_train_paths = [input_paths[i] for i in sorted_train_idx]
    input_train_filenames = [input_filenames[i] for i in sorted_train_idx]
    grdtruth_train_paths = [grdtruth_paths[i] for i in sorted_train_idx]

    sorted_val_idx = sorted(val_idx)
    input_val_paths = [input_paths[i] for i in sorted_val_idx]
    input_val_filenames = [input_filenames[i] for i in sorted_val_idx]
    grdtruth_val_paths = [input_paths[i] for i in sorted_val_idx]

    output_absdir = utils.make_dir(output_dir)

    train_absdir = utils.make_dir(os.path.join(output_absdir, 'train'))
    input_train_absdir = utils.make_dir(os.path.join(train_absdir, 'input'))
    grdtruth_train_absdir = utils.make_dir(os.path.join(train_absdir, 'grdtruth'))

    val_absdir = utils.make_dir(os.path.join(output_absdir, 'validation'))
    input_val_absdir = utils.make_dir(os.path.join(val_absdir, 'input'))
    grdtruth_val_absdir = utils.make_dir(os.path.join(val_absdir, 'grdtruth'))


    for input_path, filename, grdtruth_path in zip(input_train_paths, input_train_filenames, grdtruth_train_paths):
        input_img = imageio.imread(input_path)
        input_img = skimage.img_as_float32(input_img)
        grdtruth_img = imageio.imread(grdtruth_path)
        print(os.path.join(input_train_absdir, filename))
        imageio.imsave(path=os.path.join(input_train_absdir, filename), data=input_img)
        imageio.imsave(path=os.path.join(grdtruth_train_absdir, filename), data=grdtruth_img)

    for input_path, filename, grdtruth_path in zip(input_val_paths, input_val_filenames, grdtruth_val_paths):
        input_img = imageio.imread(input_path)
        grdtruth_img = imageio.imread(grdtruth_path)
        imageio.imsave(path=os.path.join(input_val_absdir, filename), data=input_img)
        imageio.imsave(path=os.path.join(grdtruth_val_absdir, filename), data=grdtruth_img)


def chunk_img(img_path, output_dir, chunk_shape, overlap):
    # Setup the output directory with chunking metadata
    output_absdir = utils.make_dir(output_dir)

    # Load the image into memory
    img = imageio.imread(img_path)
    img_shape = img.shape

    # Determine the number of chunks in each dimension
    nb_chunks = tuple(int(np.ceil(i/c)) for i, c in zip(img_shape, chunk_shape))

    # Pad the image while considering overlap
    noremain_shape = tuple(int(n*c) for n, c in zip(nb_chunks, chunk_shape))
    pad_after = tuple(n-i+o for n, i, o in zip(noremain_shape, img_shape, overlap))
    pad_width = tuple((b, a) for b, a in zip(overlap, pad_after))
    img_pad = np.pad(img, pad_width, mode='constant')
    padded_shape = img_pad.shape

    # Saving metadata
    metavars = ('img_shape', 'nb_chunks', 'noremain_shape', 'pad_width', 'padded_shape', 'overlap')
    metadata = dict()
    for i in metavars:
        metadata[i] = locals()[i]
    metadata_pklpath = os.path.join(output_absdir, 'metadata.pkl')
    utils.pickle_save(metadata_pklpath, metadata)

    # Save the chunks to the output directory
    for z in range(nb_chunks[0]):
        zstart = z*chunk_shape[0]
        zstop = zstart + chunk_shape[0] + 2*overlap[0]
        zstr = 'z{0:05d}'.format(zstart)
        for y in range(nb_chunks[1]):
            ystart = y*chunk_shape[1]
            ystop = ystart + chunk_shape[1] + 2*overlap[1]
            ystr = 'y{0:05d}'.format(ystart)
            for x in range(nb_chunks[2]):
                xstart = x*chunk_shape[2]
                xstop = xstart + chunk_shape[2] + 2*overlap[2]
                xstr = 'x{0:05d}'.format(xstart)
                filename = '{}_{}_{}.tif'.format(zstr, ystr, xstr)
                chunk = img_pad[zstart:zstop, ystart:ystop, xstart:xstop]
                path = os.path.join(output_absdir, filename)
                imageio.imsave(path=path, data=chunk)
    print('Finished chunking')


def combine_chunks(chunk_dir, output_path=None):
    # Get the filenames for all the chunks
    chunk_paths, chunk_filenames = utils.tifs_in_dir(chunk_dir)

    # Extract the chunking metadata
    metadata = utils.load_metadata(chunk_dir)
    img_shape, padded_shape, pad_width, overlap = itemgetter('img_shape', 'padded_shape', 'pad_width', 'overlap')(metadata)

    # Initialize the output segmentation array
    output = np.zeros(padded_shape, dtype='float32')

    for i, (chunk_path, file) in enumerate(zip(chunk_paths, chunk_filenames)):
        # Load the chunk into memory
        pad_chunk = imageio.imread(chunk_path)
        chunk = pad_chunk[overlap[0]:-overlap[0], overlap[1]:-overlap[1], overlap[2]:-overlap[2]]

        # Parse the filename into starting indices wrt global coordinates
        zstart = int(file[-23:-18]) + overlap[0]
        ystart = int(file[-16:-11]) + overlap[1]
        xstart = int(file[-9:-4]) + overlap[2]

        # Use the chunk dimensions to get the chunk bounding box
        zstop = zstart + chunk.shape[0]
        ystop = ystart + chunk.shape[1]
        xstop = xstart + chunk.shape[2]

        # Combine the chunk with what is already in the output array
        # temp = output[zstart:zstop, ystart:ystop, xstart:xstop]
        # max_proj = np.logical_or(temp, pad_chunk)
        # output[zstart:zstop, ystart:ystop, xstart:xstop] = np.logical_and(max_proj, pad_chunk)
        output[zstart:zstop, ystart:ystop, xstart:xstop] = chunk

    # Trim the output
    zstart, ystart, xstart = tuple(p[0]-1 for p in pad_width)
    zstop, ystop, xstop = tuple(p[0]-1+d for p,d in zip(pad_width, img_shape))
    output = output[zstart:zstop, ystart:ystop, xstart:xstop]

    if output_path is not None:
        imageio.imsave(path=output_path, data=output)
    else:
        return output


def main():
    img_path = '../data/control.tif'
    chunk_shape = (128, 256, 256)
    overlap = (16, 16, 16)
    chunk_dir = '../data/control/input'

    # chunk_img(img_path, chunk_dir, chunk_shape, overlap)

    chunk_dir = '../data/control/segmentation'
    output_img_path = '../data/control/segmentation.tif'
    combine_chunks(chunk_dir, output_img_path)


if __name__ == '__main__':
    main()
