import numpy as np
from skimage import feature, filters, measure
from skimage.color import rgb2gray
from skimage.feature import blob_dog, blob_log, blob_doh
from skimage.segmentation import felzenszwalb, slic
from scipy import ndimage as ndi
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN
from scipy.ndimage import gaussian_filter
from skimage.feature import peak_local_max
import numpy as np
from scipy.spatial import distance_matrix

import os
from skimage.io import imsave
import numpy as np
from datetime import datetime


def extract_cell_features(image, positions, patch_size=32):
    """
    Extract features for each cell patch using classical computer vision techniques.
    
    Args:
        image: RGB image array
        positions: Dictionary of cell positions by type
        patch_size: Size of patches to extract around each position
    """
    features_list = []
    labels = []
    half_size = patch_size // 2
    
    for cell_type, coords in positions.items():
        for x, y in coords:
            x, y = int(x), int(y)
            if (x >= half_size and x < image.shape[1] - half_size and 
                y >= half_size and y < image.shape[0] - half_size):
                
                # Extract patch
                patch = image[y-half_size:y+half_size, x-half_size:x+half_size]
                
                # 1. Color Features
                color_features = []
                for channel in range(3):  # RGB channels
                    channel_data = patch[:,:,channel]
                    color_features.extend([
                        np.mean(channel_data),
                        np.std(channel_data),
                        np.min(channel_data),
                        np.max(channel_data),
                        np.median(channel_data)
                    ])
                
                # 2. Texture Features
                gray_patch = rgb2gray(patch)
                
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
                
                # 4. Blob Features
                blobs_log = blob_log(gray_patch, max_sigma=30, num_sigma=10, threshold=.1)
                
                blob_features = [
                    len(blobs_log),  # number of detected blobs
                    np.mean(blobs_log[:, 2]) if len(blobs_log) > 0 else 0  # mean blob size
                ]
                
                # 5. Local Region Properties
                # Convert the patch to binary using Otsu's method
                from skimage.filters import threshold_otsu
                try:
                    thresh = threshold_otsu(gray_patch)
                    binary = (gray_patch > thresh).astype(np.uint8)
                except:
                    # If Otsu's method fails, use a simple threshold
                    binary = (gray_patch > gray_patch.mean()).astype(np.uint8)
                
                # Label the binary image
                labeled_image = measure.label(binary)
                
                # Get the label of the center pixel
                center_label = labeled_image[half_size, half_size]
                
                # Create a binary image of just the center region
                center_region = (labeled_image == center_label).astype(np.uint8)
                
                # Calculate region properties
                try:
                    region_props = measure.regionprops(center_region)[0]
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
                
                # 6. Intensity features in local neighborhood
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
                    blob_features,
                    shape_features,
                    intensity_features
                ])
                
                features_list.append(all_features)
                labels.append(cell_type)
    
    return np.array(features_list), labels

def train_cell_classifier(features, labels):
    """Train a Random Forest classifier on the extracted features"""
    # Scale features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Train classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(features_scaled, labels)
    
    return clf, scaler

def detect_cells(image, clf, scaler, patch_size=32, stride=16):
    """
    Detect cells in a new image using sliding window and trained classifier.
    """
    height, width = image.shape[:2]
    detections = []
    half_size = patch_size // 2
    
    for y in range(half_size, height - half_size, stride):
        for x in range(half_size, width - half_size, stride):
            patch = image[y-half_size:y+half_size, x-half_size:x+half_size]
            
            # Extract features for this patch
            features, _ = extract_cell_features(
                image[y-half_size:y+half_size+1, x-half_size:x+half_size+1],
                {'temp': [(half_size, half_size)]}
            )
            
            if len(features) > 0:
                # Scale features
                features_scaled = scaler.transform(features)
                
                # Predict
                prob = clf.predict_proba(features_scaled)
                max_prob = np.max(prob)
                
                if max_prob > 0.7:  # Confidence threshold
                    cell_type = clf.classes_[np.argmax(prob)]
                    detections.append({
                        'position': (x, y),
                        'type': cell_type,
                        'confidence': max_prob
                    })
    
    return detections

def analyze_results(detections, image_shape):
    """Analyze detection results"""
    # Convert detections to DataFrame
    df = pd.DataFrame(detections)
    
    # Analyze distribution of cell types
    type_counts = df['type'].value_counts()
    
    # Calculate spatial distribution
    positions = np.array([d['position'] for d in detections])
    
    results = {
        'cell_type_distribution': type_counts,
        'total_cells': len(detections),
        'density': len(detections) / (image_shape[0] * image_shape[1]),
        'mean_confidence': df['confidence'].mean()
    }
    
    return results


def visualize_results(test_image, detections, results, output_path=None):
    """
    Visualize cell detection results and statistics on the test image.
    
    Args:
        test_image: Input RGB image
        detections: List of dictionaries containing detection results
        results: Dictionary containing analysis results
        output_path: Optional path to save the visualization
    """
    # Set up the figure with subplots
    fig = plt.figure(figsize=(20, 10))
    
    # 1. Main image with detections
    ax1 = plt.subplot2grid((2, 3), (0, 0), colspan=2, rowspan=2)
    ax1.imshow(test_image)
    
    # Define colors for each cell type
    colors = {
        'GC': '#FF0000',    # Red
        'LSC': '#00FF00',   # Green
        'MCC': '#0000FF',   # Blue
        'SSC': '#FF00FF'    # Magenta
    }
    
    # Plot detections
    for det in detections:
        x, y = det['position']
        cell_type = det['type']
        conf = det['confidence']
        
        # Draw circle for each detection
        circle = plt.Circle((x, y), radius=10, color=colors[cell_type], 
                          fill=False, linewidth=2, alpha=conf)
        ax1.add_patch(circle)
        
        # Add small confidence indicator
        ax1.text(x+12, y+12, f'{conf:.2f}', color=colors[cell_type], 
                fontsize=8, ha='left', va='bottom')
    
    ax1.set_title('Cell Detections')
    ax1.axis('off')
    
    # 2. Cell type distribution (Bar plot)
    ax2 = plt.subplot2grid((2, 3), (0, 2))
    dist = results['cell_type_distribution']
    bars = ax2.bar(dist.index, dist.values, color=[colors[t] for t in dist.index])
    ax2.set_title('Cell Type Distribution')
    ax2.set_xlabel('Cell Type')
    ax2.set_ylabel('Count')
    ax2.tick_params(axis='x', rotation=45)
    
    # Add count labels on bars
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom')
    
    # 3. Statistics summary box
    ax3 = plt.subplot2grid((2, 3), (1, 2))
    ax3.axis('off')
    
    # Create statistics text
    stats_text = [
        f"Total Cells: {results['total_cells']}",
        f"Cell Density: {results['density']:.2e} cells/pixel²",
        f"Mean Confidence: {results['mean_confidence']:.2f}",
        "\nDistribution:",
    ]
    
    for cell_type, count in results['cell_type_distribution'].items():
        percentage = (count / results['total_cells']) * 100
        stats_text.append(f"{cell_type}: {count} ({percentage:.1f}%)")
    
    ax3.text(0.05, 0.95, '\n'.join(stats_text), 
             transform=ax3.transAxes, 
             verticalalignment='top',
             fontfamily='monospace')
    
    # Add legend for all cell types
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                 markerfacecolor=color, label=cell_type,
                                 markersize=10)
                      for cell_type, color in colors.items()]
    ax1.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if output path is provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    plt.show()



def separate_close_cells(detections, image_shape, min_distance=20, gaussian_sigma=5):
    """
    Separate closely positioned detections of the same cell type using density estimation
    and local maxima detection.
    
    Args:
        detections: List of dictionaries containing detection results
        image_shape: Tuple of (height, width) of the original image
        min_distance: Minimum distance between detected cell centers
        gaussian_sigma: Sigma for Gaussian smoothing of density map
    
    Returns:
        refined_detections: List of refined detection dictionaries
    """
    # Group detections by cell type
    detections_by_type = {}
    for det in detections:
        cell_type = det['type']
        if cell_type not in detections_by_type:
            detections_by_type[cell_type] = []
        detections_by_type[cell_type].append(det)
    
    refined_detections = []
    
    # Process each cell type separately
    for cell_type, type_detections in detections_by_type.items():
        if not type_detections:
            continue
        
        # Create density map for this cell type
        density_map = np.zeros(image_shape[:2])
        positions = np.array([d['position'] for d in type_detections])
        confidences = np.array([d['confidence'] for d in type_detections])
        
        # Add weighted contributions to density map
        for pos, conf in zip(positions, confidences):
            x, y = int(pos[0]), int(pos[1])
            if 0 <= x < image_shape[1] and 0 <= y < image_shape[0]:
                density_map[y, x] += conf
        
        # Smooth density map
        smoothed_density = gaussian_filter(density_map, sigma=gaussian_sigma)
        
        # Find local maxima in the smoothed density map
        coordinates = peak_local_max(smoothed_density,
                                   min_distance=min_distance,
                                   threshold_rel=0.2)  # Relative threshold
        
        # For each local maximum, find the closest original detections
        for y, x in coordinates:
            # Find closest original detections
            distances = distance_matrix([(x, y)], positions)[0]
            nearby_indices = np.where(distances < min_distance)[0]
            
            if len(nearby_indices) > 0:
                # Take the detection with highest confidence among nearby ones
                best_idx = nearby_indices[np.argmax(confidences[nearby_indices])]
                best_detection = type_detections[best_idx]
                
                refined_detections.append({
                    'position': (x, y),
                    'type': cell_type,
                    'confidence': best_detection['confidence'],
                    'nearby_detections': len(nearby_indices)
                })
    
    return refined_detections

def visualize_cell_separation(test_image, original_detections, refined_detections):
    """
    Visualize original and refined detections with density maps.
    
    Args:
        test_image: Original RGB image array
        original_detections: List of original detection dictionaries
        refined_detections: List of refined detection dictionaries
    """
    fig = plt.figure(figsize=(20, 15))
    
    # Define cell type styles
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
    
    # 1. Original detections
    ax1 = plt.subplot(221)
    ax1.imshow(test_image)
    ax1.set_title('Original Detections')
    
    # Plot original detections
    for det in original_detections:
        if det['type'] in cell_type_styles:
            style = cell_type_styles[det['type']]
            ax1.scatter(det['position'][0], det['position'][1],
                       marker=style['marker'], c=style['color'],
                       s=100, alpha=0.6)
    
    # 2. Density maps for each cell type
    ax2 = plt.subplot(222)
    density_map = np.zeros(test_image.shape[:2])
    
    for det in original_detections:
        x, y = int(det['position'][0]), int(det['position'][1])
        if 0 <= x < density_map.shape[1] and 0 <= y < density_map.shape[0]:
            density_map[y, x] += det['confidence']
    
    smoothed_density = gaussian_filter(density_map, sigma=5)
    ax2.imshow(smoothed_density, cmap='hot')
    ax2.set_title('Detection Density Map')
    
    # 3. Refined detections
    ax3 = plt.subplot(223)
    ax3.imshow(test_image)
    ax3.set_title('Refined Detections')
    
    # Plot refined detections
    for det in refined_detections:
        if det['type'] in cell_type_styles:
            style = cell_type_styles[det['type']]
            ax3.scatter(det['position'][0], det['position'][1],
                       marker=style['marker'], c=style['color'],
                       s=100 * np.sqrt(det['nearby_detections']),
                       alpha=0.6)
            
            # Add count of merged detections
            if det['nearby_detections'] > 1:
                ax3.annotate(str(det['nearby_detections']),
                           det['position'],
                           xytext=(5, 5),
                           textcoords='offset points',
                           fontsize=8,
                           color='white',
                           bbox=dict(facecolor='black', alpha=0.5))
    
    # 4. Comparison zoomed region (optional)
    ax4 = plt.subplot(224)
    # Find region with highest density of detections
    kernel_size = 50
    counts = np.zeros(test_image.shape[:2])
    for det in original_detections:
        x, y = int(det['position'][0]), int(det['position'][1])
        if 0 <= x < counts.shape[1] and 0 <= y < counts.shape[0]:
            counts[y, x] += 1
    
    smoothed_counts = gaussian_filter(counts, sigma=kernel_size/4)
    y, x = np.unravel_index(np.argmax(smoothed_counts), smoothed_counts.shape)
    
    # Show zoomed region
    zoom_size = 100
    y_min = max(0, y - zoom_size//2)
    y_max = min(test_image.shape[0], y + zoom_size//2)
    x_min = max(0, x - zoom_size//2)
    x_max = min(test_image.shape[1], x + zoom_size//2)
    
    ax4.imshow(test_image[y_min:y_max, x_min:x_max])
    ax4.set_title('Zoomed Region Comparison')
    
    # Plot original detections in zoomed region
    for det in original_detections:
        x, y = det['position']
        if x_min <= x < x_max and y_min <= y < y_max:
            style = cell_type_styles[det['type']]
            ax4.scatter(x - x_min, y - y_min,
                       marker=style['marker'], c=style['color'],
                       s=50, alpha=0.3)
    
    # Plot refined detections in zoomed region
    for det in refined_detections:
        x, y = det['position']
        if x_min <= x < x_max and y_min <= y < y_max:
            style = cell_type_styles[det['type']]
            ax4.scatter(x - x_min, y - y_min,
                       marker=style['marker'], c=style['color'],
                       s=100, alpha=0.6)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print("\nRefinement Summary:")
    print("-----------------")
    print(f"Original detections: {len(original_detections)}")
    print(f"Refined detections: {len(refined_detections)}")
    
    # Count by type
    orig_counts = {}
    refined_counts = {}
    for det in original_detections:
        orig_counts[det['type']] = orig_counts.get(det['type'], 0) + 1
    for det in refined_detections:
        refined_counts[det['type']] = refined_counts.get(det['type'], 0) + 1
    
    print("\nDetections by cell type:")
    for cell_type in cell_type_styles:
        if cell_type in orig_counts or cell_type in refined_counts:
            orig = orig_counts.get(cell_type, 0)
            refined = refined_counts.get(cell_type, 0)
            print(f"\n{cell_type_styles[cell_type]['label']}:")
            print(f"  Original: {orig}")
            print(f"  Refined: {refined}")
            print(f"  Change: {refined - orig}")



def save_cell_patches(image, positions, patch_size=32, base_dir=None):
    """
    Extract and save cell patches as individual images, organized by cell type.
    
    Args:
        image: Original RGB image array
        positions: Dictionary of cell positions by type
        patch_size: Size of patches to extract (default: 32)
        base_dir: Base directory to save patches (default: creates timestamp-based directory)
    """
    # Create base directory if not specified
    if base_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_dir = f"cell_patches_{timestamp}"
    
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
        
    # Save original image parameters
    params = {
        'image_size': image.shape,
        'patch_size': patch_size,
        'num_patches': 0
    }
    
    half_size = patch_size // 2
    patches_info = []  # Store information about saved patches
    
    # Process each cell type
    for cell_type, coords in positions.items():
        # Create directory for this cell type
        cell_type_dir = os.path.join(base_dir, cell_type)
        if not os.path.exists(cell_type_dir):
            os.makedirs(cell_type_dir)
        
        # Process each position for this cell type
        for idx, (x, y) in enumerate(coords):
            x, y = int(x), int(y)
            
            # Check if patch is within image bounds
            if (x >= half_size and x < image.shape[1] - half_size and 
                y >= half_size and y < image.shape[0] - half_size):
                
                # Extract patch
                patch = image[y-half_size:y+half_size, x-half_size:x+half_size]
                
                # Generate filename
                filename = f"{cell_type}_x{x}_y{y}_patch{idx+1}.png"
                filepath = os.path.join(cell_type_dir, filename)
                
                # Save patch
                imsave(filepath, patch)
                
                # Store patch information
                patches_info.append({
                    'cell_type': cell_type,
                    'filename': filename,
                    'position': (x, y),
                    'patch_size': patch_size,
                    'filepath': filepath
                })
                
                params['num_patches'] += 1

    
    return base_dir, patches_info