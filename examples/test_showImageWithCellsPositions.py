import context

from multicelldetection.imagedata.cellcountermakerfile import read_Cell_Counter_Maker_XML_file, visualize_cells, save_cell_patches

import os
from datetime import datetime
import numpy as np 
import matplotlib.pyplot as plt
from skimage import io



if __name__ == "__main__":

    """
    Main function to process and visualize cell data.
    
    Args:
        tif_path (str): Path to the TIF image
        xml_path (str): Path to the XML file
        output_path (str, optional): Path to save the visualization
    """
    ####################
    # step 1 - Parse cell positions and calibration from XML
    file_folder = 'inputs//test_1_20x'
    xml_file_path = os.path.join(file_folder, 'CellCounter_20x-UC-1-rgb.xml')

    cell_positions, calibration = read_Cell_Counter_Maker_XML_file(xml_file_path)

    # Visualize cells
    
    image_file_path = os.path.join(file_folder, '20x-UC-1-rgb.tif')

    output_folder = 'outputs//test_1'
    output_file_path = os.path.join(output_folder, f'20x-UC-1-rgb-annotation_window48.jpg')

    visualize_cells(image_file_path, cell_positions, calibration)

    # visualize_cells(image_file_path, cell_positions, calibration, window_size=48, output_path=output_file_path)

    # visualize_cells(image_file_path, cell_positions, calibration, window_size=None, output_path=output_file_path)

    train_image = io.imread(image_file_path)


    save_cell_patches(train_image, cell_positions, patch_size=48, base_dir=output_folder)





    ################################################
    # # step 1 - Parse cell positions and calibration from XML
    # file_folder = 'inputs//test_1_20x'
    # xml_file_path = os.path.join(file_folder, 'CellCounter_20x-UC-4-rgb.xml')

    # cell_positions, calibration = read_Cell_Counter_Maker_XML_file(xml_file_path)

    # # Visualize cells
    # output_folder = 'outputs//test_1'
    
    # image_file_path = os.path.join(file_folder, '20x-UC-4-rgb.tif')

    # # output_file_path = os.path.join(output_folder, f'20x-UC-4-rgb.jpg')
    # # visualize_cells(image_file_path, cell_positions, calibration, output_path=output_file_path)

    # visualize_cells(image_file_path, cell_positions, calibration, window_size=48)