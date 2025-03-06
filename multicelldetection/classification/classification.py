from sklearn import preprocessing, ensemble
from skimage import morphology, measure
import pickle 

import matplotlib.pyplot as plt
import numpy as np

from ..segmentation.SSC_segmentation import get_segmentation_mask_SSC
from .label import merge_labels, get_labels_from_mask
from .feature import extract_cell_features, get_features

def train_a_classifier(features, labels, save_path=None):
	"""Train a Random Forest classifier on the extracted features"""
	# Scale features
	scaler = preprocessing.StandardScaler()
	features_scaled = scaler.fit_transform(features)
	
	# Train classifier
	clf = ensemble.RandomForestClassifier(n_estimators=50, n_jobs=-1, random_state=42, max_depth=10, max_samples=0.05)
	clf.fit(features_scaled, labels)
	
	if save_path:
		with open(save_path, 'wb') as f:
			pickle.dump((clf, scaler), f)

	return clf, scaler


def load_a_classifier(load_path):
    with open(load_path, 'rb') as f:
        clf, scaler = pickle.load(f)
    return clf, scaler



def train(train_image_rgb, train_positions, train_image_cellpose_mask, train_image_r, save_path=None):
	# Get segmentation mask of SSC
	mask_ssc = get_segmentation_mask_SSC(train_image_r)

	# Get segmentation mask from cellpose
	mask_cellpose = train_image_cellpose_mask.copy()

	mask_cellpose[train_image_cellpose_mask > 0] = 255

	# Remove SSC mask
	mask_cellpose_remove_scc = 255 - mask_cellpose.copy()
	mask_cellpose_remove_scc[mask_ssc == 255] = 0

	# Labelize
	image_rgb_remove_ssc_label = get_labels_from_mask(mask_cellpose_remove_scc, do_processing=True, filter_size=4, min_obj_size=80)
	image_rgb_label = get_labels_from_mask(train_image_cellpose_mask)

	total_label = merge_labels(image_rgb_label, image_rgb_remove_ssc_label)
	props = measure.regionprops(total_label, train_image_rgb)

	# Extract image features
	train_feature, train_label = extract_cell_features(total_label, props, train_positions)

	# Train a classifier 
	clf, scaler = train_a_classifier(train_feature, train_label, save_path)

	return clf, scaler


def predict(classifier, scaler, test_image_rgb, test_image_cellpose_mask, test_image_r):
	# Get segmentation mask of SSC
	mask_ssc = get_segmentation_mask_SSC(test_image_r)

	# Get segmentation mask from cellpose
	mask_cellpose = test_image_cellpose_mask.copy()

	mask_cellpose[test_image_cellpose_mask > 0] = 255

	# Remove SSC mask
	mask_cellpose_remove_scc = 255 - mask_cellpose.copy()
	mask_cellpose_remove_scc[mask_ssc == 255] = 0

	# Labelize
	image_rgb_remove_ssc_label = get_labels_from_mask(mask_cellpose_remove_scc, do_processing=True, filter_size=5, min_obj_size=100)
	image_rgb_label = get_labels_from_mask(test_image_cellpose_mask)

	total_label = merge_labels(image_rgb_label, image_rgb_remove_ssc_label)
	props = measure.regionprops(total_label, test_image_rgb)

	# SSC cells from image_r
	SSC_detection = detect_SSC(test_image_r, do_processing=True, filter_size=4, min_obj_size=70)

	# cells from classifier
	cell_detection = get_cell_detection_from_a_classifier(props, classifier, scaler)

	# merge all detection
	all_cell_detection = cell_detection + SSC_detection

	return all_cell_detection


def get_SSC_detection(SSC_label):
	# ## Remember
	# marker_to_type = {
	# 		'1': 'GC_withPNA',
	# 		'2': 'GC_noPNA',
	# 		'3': 'ISC_targeted',
	# 		'4': 'ISC_untargeted',
	# 		'5': 'ISC_unclear',
	# 		'6': 'MCC_targeted',
	# 		'7': 'MCC_untargeted',
	# 		'8': 'MCC_unclear',
	# 		'9': 'SSC_targeted',
	# 		'10': 'SSC_untargeted',
	# 		'11': 'SSC_unclear'
	# 	}
	cell_type = 'SSC_untargeted'
	
	SSC_detections = []
	list_regionprops = measure.regionprops(SSC_label)
	
	for i in range(len(list_regionprops)):
		SSC_detections.append({
			'position': (list_regionprops[i].centroid[1],list_regionprops[i].centroid[0]),
			'type': cell_type,
			'confidence': 1.0
		})

	return SSC_detections



def get_cell_detection_from_a_classifier(list_regionprops, classifier, scaler):
	"""
	Detect cells in a new image using trained classifier.
	"""

	# props1 = measure.regionprops(image_label, image_grayscale2)
	detections = []
	
	for i in range(len(list_regionprops)):
		
		patch = list_regionprops[i].image_intensity
		
		features = get_features(patch)
		
		if len(features) > 0:
				# plt.imshow(patch)
				# Scale features
				features_list = np.array([features])
				features_scaled = scaler.transform(features_list)
				
				# Predict
				prob = classifier.predict_proba(features_scaled)
				max_prob = np.max(prob)
				
				if max_prob > 0.5:  # Confidence threshold
					cell_type = classifier.classes_[np.argmax(prob)]
					detections.append({
						'position': (list_regionprops[i].centroid[1],list_regionprops[i].centroid[0]),
						'type': cell_type,
						'confidence': max_prob
					})
				else:
					cell_type = classifier.classes_[np.argmax(prob)]
					detections.append({
						'position': (list_regionprops[i].centroid[1],list_regionprops[i].centroid[0]),
						'type': cell_type,
						'confidence': max_prob
					})
					
	return detections


def visualize_detection_results(test_image, detections):
	"""
	Visualize detected cells on the test image with different markers/colors for each cell type.
	
	Args:
		test_image: Original RGB image array
		detections: List of dictionaries containing detection results
				   Each dict has 'position', 'type', and 'confidence'
	"""
	fig = plt.figure(figsize=(15, 10))
	
	# Display the original image
	plt.imshow(test_image, cmap="gray")
	
	# Define different markers and colors for each cell type exactly as in XML
	cell_type_styles = {
		'GC_withPNA': {'marker': 'o', 'color': 'red', 'label': 'GC with PNA'},
		'GC_noPNA': {'marker': 'o', 'color': 'lightcoral', 'label': 'GC no PNA'},
		'ISC_targeted': {'marker': 's', 'color': 'green', 'label': 'ISC targeted'},
		'ISC_untargeted': {'marker': 's', 'color': 'lightgreen', 'label': 'ISC untargeted'},
		'ISC_unclear': {'marker': 's', 'color': 'palegreen', 'label': 'ISC unclear'},
		'MCC_targeted': {'marker': '^', 'color': 'blue', 'label': 'MCC targeted'},
		'MCC_untargeted': {'marker': '^', 'color': 'lightblue', 'label': 'MCC untargeted'},
		'MCC_unclear': {'marker': '^', 'color': 'powderblue', 'label': 'MCC unclear'},
		'SSC_targeted': {'marker': 'D', 'color': 'magenta', 'label': 'SSC targeted'},
		'SSC_untargeted': {'marker': 'D', 'color': 'plum', 'label': 'SSC untargeted'},
		'SSC_unclear': {'marker': 'D', 'color': 'thistle', 'label': 'SSC unclear'}
	}
	
	# Group detections by cell type
	cell_type_positions = {}
	for det in detections:
		cell_type = det['type']
		if cell_type not in cell_type_positions:
			cell_type_positions[cell_type] = []
		cell_type_positions[cell_type].append(det['position'])
	
	# Plot each cell type with different marker/color
	legend_elements = []  # Keep track of legend entries
	for cell_type, positions in cell_type_positions.items():
		if positions and cell_type in cell_type_styles:
			x_coords, y_coords = zip(*positions)
			style = cell_type_styles[cell_type]
			scatter = plt.scatter(x_coords, y_coords, 
								marker=style['marker'],
								c=style['color'],
								label=f"{style['label']} (n={len(positions)})",
								s=100,
								alpha=0.9)
			legend_elements.append(scatter)
	
	# Add legend if there are any plotted points
	if legend_elements:
		plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
	
	# Add title with total counts
	total_cells = sum(len(pos) for pos in cell_type_positions.values())
	plt.title(f'Cell Detection Results\nTotal Cells: {total_cells}')
	
	# Remove axes
	plt.axis('off')
	
	# Adjust layout to prevent legend cutoff
	plt.tight_layout()
	
	# Show plot
	plt.show()
	
	# Print statistical summary
	print("\nDetection Summary:")
	print("-----------------")
	for cell_type, style in cell_type_styles.items():
		count = len(cell_type_positions.get(cell_type, []))
		percentage = (count / total_cells * 100) if total_cells > 0 else 0
		print(f"{style['label']}: {count} ({percentage:.1f}%)")
	
	return fig



def detect_SSC(image_r, do_processing=False, filter_size=None, min_obj_size=None):
	
	if do_processing:
		morp_size = filter_size if filter_size is not None else 5
		obj_size = min_obj_size if min_obj_size is not None else 80
	
		# step 1 - do segmentation of SSC cells 
		mask = get_segmentation_mask_SSC(image_r, obj_size)

		# step 2 - labelize
		image_scc_label = get_labels_from_mask(mask, do_processing=True, filter_size=morp_size, min_obj_size=obj_size)

	else:
		mask = get_segmentation_mask_SSC(image_r)
		image_scc_label = get_labels_from_mask(mask)

	# step 3 - get positions
	SSC_detection = get_SSC_detection(image_scc_label)
	
	return SSC_detection