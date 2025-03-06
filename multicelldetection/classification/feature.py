import numpy as np
from skimage import color, feature, filters, measure


def get_features(color_patch):
	# 1. Color Features
	color_features = []
	for channel in range(3):  # RGB channels
		channel_data = color_patch[:,:,channel]
		color_features.extend([
                                np.mean(channel_data),
                                np.std(channel_data),
                                np.min(channel_data),
                                np.max(channel_data),
                                np.median(channel_data)
                            ])
                        
	# 2. Texture Features
	gray_patch = color.rgb2gray(color_patch)
	
	# GLCM features
	glcm = feature.graycomatrix(
                                    (gray_patch * 255).astype(np.uint8),
                                    distances=[1],
                                    angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                                    levels=256,
                                    symmetric=True,
                                    normed=True
                                )
                                
	texture_features = [
                            feature.graycoprops(glcm, 'contrast')[0, 0],
                            feature.graycoprops(glcm, 'dissimilarity')[0, 0],
                            feature.graycoprops(glcm, 'homogeneity')[0, 0],
                            feature.graycoprops(glcm, 'energy')[0, 0],
                            feature.graycoprops(glcm, 'correlation')[0, 0],
                            feature.graycoprops(glcm, 'ASM')[0, 0]
                        ]
                        
	# 3. Edge Features
	edges_sobel = filters.sobel(gray_patch)
	edges_scharr = filters.scharr(gray_patch)
	
	edge_features = [
                        np.mean(edges_sobel),
                        np.std(edges_sobel),
                        np.mean(edges_scharr),
                        np.std(edges_scharr)
                    ]

	binary = (gray_patch > 0).astype(np.uint8)
	
    # 4. Shape Features
	# Label the binary image
	labeled_image = measure.label(binary)
	# # Get the label of the center pixel
	# center_label = labeled_image[int(labeled_image.shape[0]*0.75), int(labeled_image.shape[1]*0.75)]
	# # Create a binary image of just the center region
	# center_region = (labeled_image == center_label).astype(np.uint8)
	# Calculate region properties
	try:
		region_props = measure.regionprops(labeled_image)[0]
		shape_features = [
                            region_props.area,
                            region_props.perimeter,
                            region_props.eccentricity,
                            region_props.solidity,
                            region_props.extent
                        ]
	except:
		# If region properties calculation fails, use default values
		shape_features = [0, 0, 0, 0, 0]
	
	# 5. Intensity features in local neighborhood
	intensity_features = [
		np.mean(gray_patch),
		np.std(gray_patch),
		np.percentile(gray_patch, 25),
		np.percentile(gray_patch, 75),
		np.max(gray_patch) - np.min(gray_patch)  # dynamic range
	]
	
	# Combine all features
	all_features = np.concatenate([
                                color_features,
                                texture_features,
                                edge_features,
                                shape_features,
                                intensity_features
                            ])

	return all_features


def extract_cell_features(label_image, list_regionprops, positions):
	"""
	Extract features for each cell patch.
	Args:
		image_rgb: labeled image
		positions: Dictionary of cell positions by type
	"""
	features_list = []
	labels = []
	
	for cell_type, coords in positions.items():
		for x, y in coords:

			x, y = int(x), int(y)

			for i in range(len(list_regionprops)):

				min_row, min_col, max_row, max_col = list_regionprops[i].bbox
				
				if (min_row <= y < max_row) and (min_col <= x < max_col):

					if label_image[y,x] !=0:

						patch = list_regionprops[i].image_intensity

						all_features = get_features(patch)

						features_list.append(all_features)

						labels.append(cell_type)

						break
	
	return np.array(features_list), labels    