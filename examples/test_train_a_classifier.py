import context

from multicelldetection.classification.classification import train
from multicelldetection.imagedata.cellcountermakerfile import read_Cell_Counter_Maker_XML_file

import os
from skimage import io

if __name__ == "__main__":
	# #################### use 20x-UC-1-r.tif as training data
	# step 1 - read RGB TIFF images 
	image_folder = 'inputs//test_1_20x'

	train_image_rgb_1_path = os.path.join(image_folder, '20x-UC-1-rgb.tif') # main image
	train_image_rgb_2_path = os.path.join(image_folder, '20x-UC-1-r.tif')

	train_image_rgb_1 = io.imread(train_image_rgb_1_path)
	train_image_rgb_2 = io.imread(train_image_rgb_2_path)
	
	# Load segmented mask obtained from cellpose
	mask_folder = 'outputs//test_1_20x'
	train_image_rgb_1_mask_filename = os.path.join(mask_folder, '20x-UC-1-rgb-enhanced_cp_masks.png')
	train_image_rgb_1_mask = io.imread(train_image_rgb_1_mask_filename)

	# Load cells' positions by manual annotation
	xml_file_path = os.path.join(image_folder, 'CellCounter_20x-UC-1-rgb.xml')

	train_positions, _ = read_Cell_Counter_Maker_XML_file(xml_file_path)

	# step 2 - train a classifier
	output_folder = os.path.join('outputs', os.path.basename(image_folder)) # output folder
	os.makedirs(output_folder, exist_ok=True) # Create the directory if it doesn't exist
	
	classifier_filepath = os.path.join(output_folder, 'RF_classifier_test_1.pkl') 

	_, _= train(train_image_rgb_1, train_positions, train_image_rgb_1_mask, train_image_rgb_2, classifier_filepath)





	
