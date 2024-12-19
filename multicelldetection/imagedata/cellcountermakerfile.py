import numpy as np
import xml.etree.ElementTree as ET
import tifffile
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import pandas as pd
import seaborn as sns
from matplotlib.patches import Rectangle

def read_Cell_Counter_Maker_XML_file(xml_file_path):
	"""
	Parse the CellCounter XML file to extract cell positions and types.
	
	Args:
		xml_path (str): Path to the XML file
		
	Returns:
		dict: Dictionary with cell types as keys and list of (x,y) coordinates as values
	"""
	if not xml_file_path.lower().endswith('.xml'):
		raise ValueError("Input must be a XML file with .xml extension")
	
	try:
		tree = ET.parse(xml_file_path)
		root = tree.getroot()
		# Get calibration information
		calibration = {
			'x': float(root.find('.//X_Calibration').text),
			'y': float(root.find('.//Y_Calibration').text),
			'unit': root.find('.//Calibration_Unit').text
		}

		# Initialize dictionary for cell positions
		cell_positions = {
			'GC_withPNA': [],
			'GC_noPNA': [],
			'ISC_targeted': [],
			'ISC_untargeted': [],
			'ISC_unclear': [],
			'MCC_targeted': [],
			'MCC_untargeted': [],
			'MCC_unclear': [],
			'SSC_targeted': [],
			'SSC_untargeted': [],
			'SSC_unclear': []
		}

		# Map marker types to cell categories
		marker_to_type = {
			'1': 'GC_withPNA',
			'2': 'GC_noPNA',
			'3': 'ISC_targeted',
			'4': 'ISC_untargeted',
			'5': 'ISC_unclear',
			'6': 'MCC_targeted',
			'7': 'MCC_untargeted',
			'8': 'MCC_unclear',
			'9': 'SSC_targeted',
			'10': 'SSC_untargeted',
			'11': 'SSC_unclear'
		}

		# Parse each marker type
		for marker_type in root.findall('.//Marker_Type'):
			type_num = marker_type.find('Type').text
			cell_type = marker_to_type.get(type_num)

			if cell_type:
			# Get all markers for this type
				for marker in marker_type.findall('Marker'):
					x = float(marker.find('MarkerX').text) #* calibration['x']
					y = float(marker.find('MarkerY').text) #* calibration['y']

					cell_positions[cell_type].append((x, y))

		return cell_positions, calibration
	except IOError:
		raise IOError(f"Unable to read the file: {xml_file_path}")
	except UnicodeDecodeError:
		raise ValueError("The file is not a valid XML file or contains non-text content")
	

def visualize_cells(image_path, cell_positions, calibration, window_size=None, output_path=None):
	# """
	# Create visualization of the fluorescence image with cell type annotations.
	
	# Args:
	# 	image_path (str): Path to the TIF image
	# 	cell_positions (dict): Dictionary of cell positions by type
	# 	calibration (dict): Calibration information from XML
	# 	output_path (str, optional): Path to save the visualization
	# """
	# # Read the TIF image
	# img = tifffile.imread(image_path)
	
	# # If image has multiple channels, create a composite RGB image
	# if len(img.shape) == 3:
	# 	# Normalize each channel
	# 	img_normalized = np.zeros_like(img, dtype=float)
	# 	for i in range(img.shape[0]):
	# 		channel = img[i]
	# 		img_normalized[i] = (channel - channel.min()) / (channel.max() - channel.min())
		
	# 	# Create RGB composite
	# 	rgb_img = np.stack([
	# 		img_normalized[0],  # R channel
	# 		img_normalized[1],  # G channel
	# 		img_normalized[2] if img.shape[0] > 2 else np.zeros_like(img_normalized[0])  # B channel
	# 	], axis=-1)
	# else:
	# 	# If single channel, create grayscale image
	# 	rgb_img = (img - img.min()) / (img.max() - img.min())
	# 	rgb_img = np.stack([rgb_img, rgb_img, rgb_img], axis=-1)
	
	# # Create visualization
	# plt.figure(figsize=(15, 10))
	# plt.imshow(rgb_img)
	
	# # Define colors for each cell type
	# colors = {
	# 	'GC_withPNA': '#FF0000',      # Red
	# 	'GC_noPNA': '#FF7F7F',        # Light red
	# 	'ISC_targeted': '#00FF00',     # Green
	# 	'ISC_untargeted': '#7FFF7F',   # Light green
	# 	'ISC_unclear': '#CCFFCC',      # Very light green
	# 	'MCC_targeted': '#0000FF',     # Blue
	# 	'MCC_untargeted': '#7F7FFF',   # Light blue
	# 	'MCC_unclear': '#CCCCFF',      # Very light blue
	# 	'SSC_targeted': '#FF00FF',     # Magenta
	# 	'SSC_untargeted': '#FF7FFF',   # Light magenta
	# 	'SSC_unclear': '#FFCCFF'       # Very light magenta
	# }
	
	# # Plot cell positions
	# for cell_type, positions in cell_positions.items():
	# 	if positions:  # Only plot if there are cells of this type
	# 		x_coords, y_coords = zip(*positions)
	# 		plt.scatter(x_coords, y_coords, c=[colors[cell_type]], 
	# 				   label=f'{cell_type} (n={len(positions)})', 
	# 				   alpha=0.7, s=50)
	
	# plt.title('Cell Type Distribution')
	# plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
	# plt.axis('off')
	
	# if output_path:
	# 	plt.savefig(output_path, dpi=300, bbox_inches='tight')
	# plt.show()
	"""
	Create visualization of the fluorescence image with cell type annotations using Seaborn.
	
	Args:
		image_path (str): Path to the TIF image
		cell_positions (dict): Dictionary of cell positions by type
		calibration (dict): Calibration information from XML
		output_path (str, optional): Path to save the visualization
	"""
	# Set Seaborn style
	sns.set_style("white")
	sns.set_context("talk")
	
	# Create figure with multiple subplots
	fig, ax1 = plt.subplots(figsize=(20, 15))
	
	# # Main plot for cell positions
	# ax1 = plt.subplot2grid((1, 1), (0, 0), colspan=1)
	
	# Read and display the RGB TIF image directly
	img = tifffile.imread(image_path)
	ax1.imshow(img)
	
	# Define color palette for cell types
	# colors = {
	# 	'GC_withPNA': sns.color_palette("husl", 11)[0],
	# 	'GC_noPNA': sns.color_palette("husl", 11)[1],
	# 	'ISC_targeted': sns.color_palette("husl", 11)[2],
	# 	'ISC_untargeted': sns.color_palette("husl", 11)[3],
	# 	'ISC_unclear': sns.color_palette("husl", 11)[4],
	# 	'MCC_targeted': sns.color_palette("husl", 11)[5],
	# 	'MCC_untargeted': sns.color_palette("husl", 11)[6],
	# 	'MCC_unclear': sns.color_palette("husl", 11)[7],
	# 	'SSC_targeted': sns.color_palette("husl", 11)[8],
	# 	'SSC_untargeted': sns.color_palette("husl", 11)[9],
	# 	'SSC_unclear': sns.color_palette("husl", 11)[10]
	# }

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
	
	# Plot cell positions
	if window_size:	
		half_window = window_size//2
	else:
		half_window = None

	for cell_type, positions in cell_positions.items():
		if positions:
			x_coords, y_coords = zip(*positions)
			# ax1.scatter(x_coords, y_coords, c=[colors[cell_type]], 
			# 		   label=f'{cell_type} (n={len(positions)})', 
			# 		   alpha=0.7, s=50)
			
			style = cell_type_styles[cell_type]
			ax1.scatter(x_coords, y_coords, 
								marker=style['marker'],
								c=style['color'],
								label=f"{style['label']} (n={len(positions)})",
								s=100,
								alpha=0.6)
	
			if half_window:
				# Add windows around each position
				for x, y in positions:
					# Create a rectangle patch
					rect = Rectangle(
						(x - half_window, y - half_window),  # (x,y) of bottom left corner
						window_size,  # width
						window_size,  # height
						linewidth=1,
						edgecolor=style['color'],
						facecolor='none',
						alpha=0.5,
						linestyle='--'
					)
					# Add the rectangle to the plot
					ax1.add_patch(rect)

	ax1.set_title('Cell Type Distribution', pad=20)
	ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
	ax1.axis('off')
	
	# # Create DataFrame for additional plots
	# plot_data = []
	# for cell_type, positions in cell_positions.items():
	# 	if positions:
	# 		for x, y in positions:
	# 			plot_data.append({
	# 				'Cell Type': cell_type,
	# 				f'X Position ({calibration["unit"]})': x,
	# 				f'Y Position ({calibration["unit"]})': y,
	# 				'Category': cell_type.split('_')[0],
	# 				'Status': '_'.join(cell_type.split('_')[1:])
	# 			})
	
	# df = pd.DataFrame(plot_data)
	
	# # Distribution plots
	# ax2 = plt.subplot2grid((2, 2), (1, 0))
	# ax3 = plt.subplot2grid((2, 2), (1, 1))
	
	# # X-position distribution by cell type
	# sns.violinplot(data=df, x='Category', y=f'X Position ({calibration["unit"]})',
	# 			  hue='Status', ax=ax2, palette='husl')
	# ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)
	# ax2.set_title('X Position Distribution by Cell Type')
	
	# # Y-position distribution by cell type
	# sns.violinplot(data=df, x='Category', y=f'Y Position ({calibration["unit"]})',
	# 			  hue='Status', ax=ax3, palette='husl')
	# ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45)
	# ax3.set_title('Y Position Distribution by Cell Type')
	
	plt.tight_layout()
	
	if output_path:
		plt.savefig(output_path, dpi=300, bbox_inches='tight')
	plt.show()


def analyze_cell_distribution(cell_positions, calibration):
	"""
	Analyze the spatial distribution of different cell types.
	
	Args:
		cell_positions (dict): Dictionary of cell positions by type
		calibration (dict): Calibration information from XML
		
	Returns:
		pd.DataFrame: Statistics about cell distribution
	"""
	stats = []
	
	for cell_type, positions in cell_positions.items():
		if positions:
			positions = np.array(positions)
			x_coords = positions[:, 0]
			y_coords = positions[:, 1]
			
			stats.append({
				'Cell Type': cell_type,
				'Count': len(positions),
				f'Mean X ({calibration["unit"]})': np.mean(x_coords),
				f'Mean Y ({calibration["unit"]})': np.mean(y_coords),
				f'Std X ({calibration["unit"]})': np.std(x_coords),
				f'Std Y ({calibration["unit"]})': np.std(y_coords)
			})
	
	return pd.DataFrame(stats)