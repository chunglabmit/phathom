#@ File(label="Select an input folder", style='file') input_path
#@ File(label="Select an input folder", style='file') output_path

imp = IJ.openImage(input_path)
IJ.run(imp, 
	   "Enhance Local Contrast (CLAHE)", 
	   "blocksize=127 histogram=256 maximum=3 mask=*None* fast_(less_accurate)")
IJ.run(imp, "Subtract Background...", "rolling=50")	
fs = FileSaver(imp)
fs.saveAsTiff(output_path)