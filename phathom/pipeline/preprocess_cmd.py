import argparse
try:
    from blockfs.directory import Directory as BlockfsDirectory
    haz_blockfs = True
except ImportError:
    haz_blockfs = False
import multiprocessing
import os
import sys
import tifffile
import tqdm

from phathom.preprocess.filtering import preprocess, clahe_2d
from phathom.utils import tifs_in_dir, make_dir
from phathom.io.conversion import tifs_to_zarr
from phathom.io.zarr import new_zarr
from phathom.utils import SharedMemory, shared_memory_to_zarr, memory_to_blockfs

OF_BLOCKFS = "blockfs"
OF_TIFF = "tiff"
OF_ZARR = "zarr"
if haz_blockfs:
    OF_ALL = (OF_BLOCKFS, OF_TIFF, OF_ZARR)
else:
    OF_ALL = (OF_TIFF, OF_ZARR)


def parse_args(args=sys.argv[1:]):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        help="the path to the input TIFF files",
        required=True)
    parser.add_argument(
        "--output",
        help="the path to the directory for the output volume",
        required=True)
    parser.add_argument(
        "--output-format",
        help="The format for the output. Possible values are \"tiff\" for "
        "a stack of TIFF files, \"blockfs\" for a BLOCKFS volume and \"zarr\" "
        "for a ZARR volume. The default is \"tiff\".",
        default="tiff")
    parser.add_argument(
        "--threshold",
        type=int,
        help="The threshold to use when converting the image to a binary mask. "
        "The default is to store the intensity values instead of thresholding.")
    parser.add_argument(
        "--kernel-size",
        type=int,
        default=127,
        help="The size of the adaptive kernel for CLAHE")
    parser.add_argument(
        "--clip-limit",
        type=float,
        default=.01,
        help="The clip limit for CLAHE")
    parser.add_argument(
        "--n-workers",
        type=int,
        default=1,
        help="The number of worker processes to use for multiprocessing")
    return parser.parse_args(args)


def do_shmem_clahe(tiff_path, shared_memory, idx, threshold, kernel_size,
                   clip_limit):
    img = tifffile.imread(tiff_path)
    with shared_memory.txn() as m:
        img_max = img.max()
        img_min = img.min()
        enhanced_normalized = clahe_2d(img, kernel_size, clip_limit)
        enhanced = enhanced_normalized * (img_max - img_min) + img_min
        if threshold is not None:
            m[idx] = (enhanced * (img >= threshold)).astype(shared_memory.dtype)
        else:
            m[idx] = enhanced.astype(shared_memory.dtype)


def main(args=sys.argv[1:]):
    args = parse_args(args)
    if args.output_format not in OF_ALL:
        raise ValueError("--output-format must be one of %s" % str(OF_ALL))
    paths, filenames = tifs_in_dir(args.input)
    img0 = tifffile.imread(paths[0])
    shared_memory = SharedMemory(
        (min(len(paths), 64), img0.shape[0], img0.shape[1]),
        img0.dtype)
    shape = (len(paths), img0.shape[0], img0.shape[1])
    if args.output_format == OF_TIFF:
        if not os.path.exists(args.output):
            os.mkdir(args.output)
    elif args.output_format == OF_BLOCKFS:
        if not os.path.exists(os.path.dirname(args.output)):
            os.mkdir(os.path.dirname(args.output))
    if args.output_format == OF_ZARR:
        dest = new_zarr(args.output, shape, (64, 64, 64), img0.dtype)
    elif args.output_format == OF_BLOCKFS:
        dest = BlockfsDirectory(
            shape[2], shape[1], shape[0], img0.dtype, args.output,
            n_filenames = args.n_workers)
        dest.create()
        dest.start_writer_processes()
    try:
        with multiprocessing.Pool(args.n_workers) as pool:
            for idx64 in tqdm.tqdm(range(0, len(paths), 64)):
                idx64_end = min(len(paths), idx64+64)
                cmd_args = []

                for idx, (path, filename) in enumerate(
                        zip(paths[idx64:idx64_end],
                            filenames[idx64:idx64_end])):
                    cmd_args.append((path, shared_memory, idx,
                                     args.threshold, args.kernel_size,
                                     args.clip_limit))
                pool.starmap(do_shmem_clahe, cmd_args)
                with shared_memory.txn() as m:
                    if args.output_format == OF_TIFF:
                        for idx, (path, filename) in enumerate(
                            zip(paths[idx64:idx64_end],
                                filenames[idx64:idx64_end])):
                            out_path = os.path.join(args.output, filename)
                            tifffile.imsave(out_path, m[idx])
                    elif args.output_format == OF_ZARR:
                        shared_memory_to_zarr(
                            shared_memory, dest, pool, (idx64, 0, 0))
                    else:
                        memory_to_blockfs(
                            m[:idx64_end-idx64], dest, (idx64, 0, 0))
    finally:
        if args.output_format == OF_BLOCKFS:
            dest.close()


if __name__=="__main__":
    main()
