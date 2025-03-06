import context

from multicelldetection.segmentation.SSC_segmentation import get_segmentation_mask_SSC

import numpy as np
import os
from skimage import io, filters, img_as_float, color, util, morphology
import matplotlib.pyplot as plt


if __name__ == "__main__":
	# #################### apply for 20x-UC-1-r.tif
	# step 1 - read RGB TIFF images 
	file_folder = 'inputs//test_1_20x'

	# image_rgb_1_path = os.path.join(file_folder, '20x-UC-1-rgb.tif') # main image
	image_rgb_2_path = os.path.join(file_folder, '20x-UC-1-r.tif')

	# Load the TIFF images
	# image_rgb_1 = io.imread(image_rgb_1_path)
	image_rgb_2 = io.imread(image_rgb_2_path)

	# step 2 - do segmentation of SSC cells
	image_ssc = get_segmentation_mask_SSC()


	# step 3 - Plotting
	fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 3.5))
	ax[0].imshow(image_rgb_2, cmap='gray')
	ax[0].set_title('Original')
	ax[0].axis('off')

	# Plotting the segmentation results
	ax[2].imshow(image_ssc, cmap='gray')
	ax[2].set_title('SSC masks')
	ax[2].axis('off')

	plt.subplots_adjust()

	plt.show()

	# step 4 - save it to outputs folder    
	output_folder = os.path.join('outputs', os.path.basename(file_folder)) # output folder
	os.makedirs(output_folder, exist_ok=True) # Create the directory if it doesn't exist
	
	image_ssc_filename = os.path.splitext(os.path.basename(image_rgb_2_path))[0] + '-segmentation-SSC.tif'
	image_ssc_path = os.path.join(output_folder, image_ssc_filename) 
	io.imsave(image_ssc_path, image_ssc)

	#################### Do the same for another image :  20x-UC-4-r,tif

	

	################### Think about process all images in a folder