#@ File(label="Select an input folder", style='directory') input_dir
#@ Integer(label="Number of threads", value=1) nb_threads

import os
import sys
import threading
from java.util.concurrent import Executors, TimeUnit, Callable
from java.lang import InterruptedException
from ij import IJ
from ij.io import FileSaver


def setup_folders(input_dir):
	input_dir = str(input_dir)
	parent_dir = os.path.dirname(input_dir)
	dirname = str(os.path.relpath(input_dir, parent_dir))
	output_dir = os.path.join(parent_dir, dirname+'_subtracted')
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


def process_image(input_path, output_path):
	imp = IJ.openImage(input_path)
	# IJ.run(imp, "Enhance Local Contrast (CLAHE)", "blocksize=127 histogram=256 maximum=3 mask=*None* fast_(less_accurate)")
	IJ.run(imp, "Subtract Background...", "rolling=50")	
	fs = FileSaver(imp)
	fs.saveAsTiff(output_path)


class Preprocessor(Callable):
    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path
        self.result = None
        self.thread_used = None
        self.exception = None

    def call(self):
        self.thread_used = threading.currentThread().getName()
        try:
            self.result = process_image(self.input_path, self.output_path)
        except Exception, ex:
            self.exception = ex
        return self


def shutdown_and_await_termination(pool, timeout):
	pool.shutdown()
	try:
		if not pool.awaitTermination(timeout, TimeUnit.SECONDS):
			pool.shutdownNow()
			if (not pool.awaitTermination(timeout, TimeUnit.SECONDS)):
				print >> sys.stderr, "Pool did not terminate"
	except InterruptedException, ex:
		pool.shutdownNow()
		Thread.currentThread().interrupt()
        

def batch_process(input_paths, output_paths):
	for i, (input_path, output_path) in enumerate(zip(input_paths, output_paths)):
		print("Processing image %d of %d" % (i, len(input_paths)))
		process_image(input_path, output_path)


if __name__ in ['__main__', '__builtin__']:
	input_dir, output_dir = setup_folders(input_dir)
	input_paths, output_paths = build_paths(input_dir, output_dir)
	pool = Executors.newFixedThreadPool(nb_threads)
	preprocessors = [Preprocessor(i, o) for i, o in zip(input_paths, output_paths)]
	futures = pool.invokeAll(preprocessors)
	shutdown_and_await_termination(pool, 1e9)
