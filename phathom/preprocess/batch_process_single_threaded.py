#@ File(label="Select input folder", style='directory') input_dir
#@ File(label="Select output folder", style='directory') output_dir
#@ Integer(label="Starting index",value=0) start
#@ Integer(label="Stopping index",value=-1) stop

import os
from ij import IJ
from ij.io import FileSaver


def setup_folders(input_dir, output_dir):
	input_dir = str(input_dir)
	output_dir = str(output_dir)
	if not os.path.exists(input_dir):
	 	raise ValueError('Inpuyt folder does not exist')
	if not os.path.exists(output_dir):
		raise ValueError('Output folder does not exist')
	return input_dir, output_dir


def build_paths(input_dir, output_dir, start, stop):
	files = sorted(os.listdir(input_dir))
	if stop < 0:
		stop = len(files)
	input_paths = []
	output_paths = []
	for filename in files[start:stop]:
		input_paths.append(os.path.join(input_dir, filename))
		output_paths.append(os.path.join(output_dir, filename))
	return input_paths, output_paths


def process_image(input_path, output_path):
	imp = IJ.openImage(input_path)
	IJ.run(imp, 
		   "Enhance Local Contrast (CLAHE)", 
		   "blocksize=127 histogram=256 maximum=3 mask=*None* fast_(less_accurate)")
	IJ.run(imp, "Subtract Background...", "rolling=50")	
	fs = FileSaver(imp)
	fs.saveAsTiff(output_path)
     

def batch_process(input_paths, output_paths):
	for i, (input_path, output_path) in enumerate(zip(input_paths, output_paths)):
		process_image(input_path, output_path)


if __name__ in ['__main__', '__builtin__']:
	input_dir, output_dir = setup_folders(input_dir, output_dir)
	input_paths, output_paths = build_paths(input_dir, output_dir, start, stop)
	batch_process(input_paths, output_paths)
