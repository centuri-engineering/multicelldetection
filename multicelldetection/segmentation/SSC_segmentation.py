import numpy as np
import os
from skimage import io, filters, img_as_float, color, util, morphology


def get_segmentation_mask_SSC(image_rgb_2, min_obj_size=None):
	# step 1 - pre-processing 
	image_rgb_2_float = img_as_float(image_rgb_2)
	image_rgb_2_float_rgb = color.rgb2gray(image_rgb_2_float)

	# step 2 - -multi-otsu thresholding 
	thresholds = filters.threshold_multiotsu(image_rgb_2_float_rgb, 5)
	regions = np.digitize(image_rgb_2_float_rgb, bins=thresholds)
	image_rgb_2_float_rgb_thresh = (regions == 4)

	# step 3 - post-processing i.e., remove small objects
	if min_obj_size is not None:
		image_ssc = util.img_as_ubyte(morphology.remove_small_objects(image_rgb_2_float_rgb_thresh, min_obj_size))
	else:
		image_ssc = util.img_as_ubyte(image_rgb_2_float_rgb_thresh)
								
	return image_ssc