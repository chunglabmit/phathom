import os
import sys

# TODO: call imagej subprocesses with preprocessing script


def setup_folders(input_dir):
    input_dir = str(input_dir)
    parent_dir = os.path.dirname(input_dir)
    dirname = str(os.path.relpath(input_dir, parent_dir))
    output_dir = os.path.join(parent_dir, dirname + '_subtracted')
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    return input_dir, output_dir


def build_paths(input_dir, output_dir):
    input_paths = []
    output_paths = []
    for directory, dirnames, filenames in os.walk(input_dir):
        for filename in filenames:
            if filename.endswith('.tiff') or filename.endswith('.tif'):
                input_path = os.path.join(directory, filename)
                relpath = os.path.relpath(input_path, input_dir)
                output_directory = os.path.join(output_dir, relpath)
                output_path = os.path.join(output_dir, filename)
                input_paths.append(input_path)
                output_paths.append(output_path)
    return input_paths, output_paths


def batch_process(input_paths, output_paths):
    for i, (input_path, output_path) in enumerate(zip(input_paths, output_paths)):
        print("Processing image %d of %d" % (i, len(input_paths)))
        process_image(input_path, output_path)


def main(input_dir):
    input_dir, output_dir = setup_folders(input_dir)
    input_paths, output_paths = build_paths(input_dir, output_dir)
    results = batch_process(input_paths, output_paths)


if __name__ == '__main__':
    main(input_dir)
