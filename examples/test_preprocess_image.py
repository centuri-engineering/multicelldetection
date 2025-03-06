import context

from multicelldetection.preprocessing.enhancement import convert_RGB_to_HSV, merge_channels, convert_RGB_to_grayscale

import os
from skimage import io


if __name__ == "__main__":
    # #################### apply for 20x-UC-1-rgb.tif and 20x-UC-1-r.tif
    # step 1 - read RGB TIFF images 
    file_folder = 'inputs//test_1_20x'

    image_rgb_1_path = os.path.join(file_folder, '20x-UC-1-rgb.tif') # main image
    image_rgb_2_path = os.path.join(file_folder, '20x-UC-1-r.tif')

    # Load the TIFF images
    image_rgb_1 = io.imread(image_rgb_1_path)
    image_rgb_2 = io.imread(image_rgb_2_path)

    # step 2 - do preprocessing 
    image_rgb_1_hsv = convert_RGB_to_HSV(image_rgb_1)
    image_rgb_21_hsv_enhanced = merge_channels(image_rgb_2, image_rgb_1_hsv)
    image_rgb_21_hsv_enhanced_grayscale = convert_RGB_to_grayscale(image_rgb_21_hsv_enhanced)


    # step 3 - save it to outputs folder    
    output_folder = os.path.join('outputs', os.path.basename(file_folder)) # output folder
    os.makedirs(output_folder, exist_ok=True) # Create the directory if it doesn't exist
    
    image_rgb_21_hsv_enhanced_grayscale_filename = os.path.splitext(os.path.basename(image_rgb_1_path))[0] + '-enhanced.tif'
    image_rgb_21_hsv_enhanced_grayscale_path = os.path.join(output_folder, image_rgb_21_hsv_enhanced_grayscale_filename) 
    io.imsave(image_rgb_21_hsv_enhanced_grayscale_path, image_rgb_21_hsv_enhanced_grayscale)

    #################### Do the same for another image :  20x-UC-4-rgb.tif and 20x-UC-4-r,tif

    # # step 1 - read RGB TIFF images
    # file_folder = 'inputs//test_1_20x'

    # image_rgb_1_path = os.path.join(file_folder, '20x-UC-4-rgb.tif') # main image
    # image_rgb_2_path = os.path.join(file_folder, '20x-UC-4-r.tif')

    # # Load the TIFF images
    # image_rgb_1 = io.imread(image_rgb_1_path)
    # image_rgb_2 = io.imread(image_rgb_2_path)

    # # step 2 - do preprocessing 
    # image_rgb_1_hsv = convert_RGB_to_HSV(image_rgb_1)
    # image_rgb_21_hsv_enhanced = merge_channels(image_rgb_2, image_rgb_1_hsv)
    # image_rgb_21_hsv_enhanced_grayscale = convert_RGB_to_grayscale(image_rgb_21_hsv_enhanced)


    # # step 3 - save it to outputs folder    
    # output_folder = os.path.join('outputs', os.path.basename(file_folder)) # output folder
    # os.makedirs(output_folder, exist_ok=True) # Create the directory if it doesn't exist
    
    # image_rgb_21_hsv_enhanced_grayscale_filename = os.path.splitext(os.path.basename(image_rgb_1_path))[0] + '-enhanced.tif'
    # image_rgb_21_hsv_enhanced_grayscale_path = os.path.join(output_folder, image_rgb_21_hsv_enhanced_grayscale_filename) 
    # io.imsave(image_rgb_21_hsv_enhanced_grayscale_path, image_rgb_21_hsv_enhanced_grayscale)


    ################### Think about process all images in a folder