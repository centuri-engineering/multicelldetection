import context
from celldetection.imagedata.cellcountermakerfile import read_Cell_Counter_Maker_XML_file, visualize_cells, analyze_cell_distribution
from celldetection.imagedata.classicdetection import extract_cell_features, train_cell_classifier, detect_cells, analyze_results,separate_close_cells, visualize_cell_separation
import os
from datetime import datetime
import numpy as np 
import matplotlib.pyplot as plt
from skimage import io
import tifffile

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
    plt.imshow(test_image)
    
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
                                alpha=0.6)
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



def visualize_detection_details(test_image, detections, patch_size=32):
    """
    Create a detailed visualization showing high-confidence detections
    and their local patches.
    """
    # Sort detections by confidence
    high_conf_detections = sorted(detections, 
                                key=lambda x: x['confidence'], 
                                reverse=True)[:10]  # Show top 10
    
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 5))
    
    # Plot original image with all detections
    ax1 = plt.subplot(121)
    ax1.imshow(test_image)
    
    # Define colors for each cell type
    colors = {
        'GC_withPNA': 'red',
        'GC_noPNA': 'lightcoral',
        'ISC_targeted': 'green',
        'ISC_untargeted': 'lightgreen',
        'ISC_unclear': 'palegreen',
        'MCC_targeted': 'blue',
        'MCC_untargeted': 'lightblue',
        'MCC_unclear': 'powderblue',
        'SSC_targeted': 'magenta',
        'SSC_untargeted': 'plum',
        'SSC_unclear': 'thistle'
    }
    
    # Plot all detections
    for det in detections:
        if det['type'] in colors:
            x, y = det['position']
            ax1.scatter(x, y, c=colors[det['type']], alpha=0.6, s=50)
    
    ax1.set_title('All Detections')
    ax1.axis('off')
    
    # Create grid of high-confidence patches
    ax2 = plt.subplot(122)
    ax2.axis('off')
    
    half_size = patch_size // 2
    for i, det in enumerate(high_conf_detections):
        x, y = det['position']
        
        # Extract patch
        if (x >= half_size and x < test_image.shape[1] - half_size and 
            y >= half_size and y < test_image.shape[0] - half_size):
            patch = test_image[y-half_size:y+half_size, 
                             x-half_size:x+half_size]
            
            # Add small subplot for this patch
            ax_patch = fig.add_subplot(5, 2, i+1)
            ax_patch.imshow(patch)
            ax_patch.axis('off')
            ax_patch.set_title(f"{det['type']}\nconf: {det['confidence']:.2f}")
    
    plt.tight_layout()
    plt.show()



import json
import csv
import pandas as pd
import pickle
from datetime import datetime

def save_detection_results(detections, base_filename=None):
    """
    Save detection results in multiple formats (CSV, JSON, Excel, and Pickle).
    
    Args:
        detections: List of dictionaries containing detection results
        base_filename: Base name for the output files (if None, uses timestamp)
    """
    if base_filename is None:
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"cell_detection_{timestamp}"
    
    # Convert detections to DataFrame for easier handling
    detection_data = []
    for det in detections:
        x, y = det['position']
        detection_data.append({
            'cell_type': det['type'],
            'x_position': x,
            'y_position': y,
            'confidence': det['confidence']
        })
    
    df = pd.DataFrame(detection_data)
    
    # 1. Save as CSV
    csv_filename = f"{base_filename}.csv"
    df.to_csv(csv_filename, index=False)
    print(f"Saved CSV file: {csv_filename}")
    
    # 3. Save as JSON
    json_filename = f"{base_filename}.json"
    with open(json_filename, 'w') as f:
        json.dump(detection_data, f, indent=2)
    print(f"Saved JSON file: {json_filename}")
    
    # 4. Save as Pickle (preserves all original data structures)
    pickle_filename = f"{base_filename}.pkl"
    with open(pickle_filename, 'wb') as f:
        pickle.dump(detections, f)
    print(f"Saved Pickle file: {pickle_filename}")
    
    # 5. Save summary statistics
    summary_filename = f"{base_filename}_summary.txt"
    with open(summary_filename, 'w') as f:
        # Count by cell type
        cell_type_counts = df['cell_type'].value_counts()
        
        f.write("Cell Detection Summary\n")
        f.write("====================\n\n")
        f.write(f"Total detections: {len(detections)}\n\n")
        
        f.write("Counts by cell type:\n")
        for cell_type, count in cell_type_counts.items():
            f.write(f"{cell_type}: {count}\n")
        
        f.write(f"\nMean confidence: {df['confidence'].mean():.3f}\n")
        f.write(f"Min confidence: {df['confidence'].min():.3f}\n")
        f.write(f"Max confidence: {df['confidence'].max():.3f}\n")
    
    print(f"Saved summary file: {summary_filename}")
    
    return {
        'csv': csv_filename,
        'json': json_filename,
        'pickle': pickle_filename,
        'summary': summary_filename
    }

def load_detection_results(filename):
    """
    Load detection results from saved files.
    
    Args:
        filename: Path to the saved file
        
    Returns:
        detections: List of detection dictionaries
    """
    extension = filename.split('.')[-1].lower()
    
    if extension == 'csv':
        df = pd.read_csv(filename)
        detections = []
        for _, row in df.iterrows():
            detections.append({
                'type': row['cell_type'],
                'position': (row['x_position'], row['y_position']),
                'confidence': row['confidence']
            })
        return detections
        
    elif extension == 'json':
        with open(filename, 'r') as f:
            data = json.load(f)
        detections = []
        for det in data:
            detections.append({
                'type': det['cell_type'],
                'position': (det['x_position'], det['y_position']),
                'confidence': det['confidence']
            })
        return detections
        
    elif extension == 'pkl':
        with open(filename, 'rb') as f:
            return pickle.load(f)
            
    else:
        raise ValueError(f"Unsupported file format: {extension}")

# Example usage
def save_and_load_example():
    """
    Example of how to save and load detection results
    """
    # After running detection
    # detections = detect_cells(test_image, clf, scaler)
    
    # Save results
    filenames = save_detection_results(detections, "my_detection_results")
    
    # Later, load results
    # From CSV
    detections_csv = load_detection_results(filenames['csv'])
    
    # From Excel
    detections_excel = load_detection_results(filenames['excel'])
    
    # From JSON
    detections_json = load_detection_results(filenames['json'])
    
    # From Pickle (preserves original format exactly)
    detections_pickle = load_detection_results(filenames['pickle'])
    
    return detections_pickle  # Original format



if __name__ == "__main__":

    """
    Main function to process and visualize cell data.
    
    Args:
        tif_path (str): Path to the TIF image
        xml_path (str): Path to the XML file
        output_path (str, optional): Path to save the visualization
    """

    #########################################
    ### Step 1 - Load image and annotations for training
    file_folder = 'inputs//test_1_20x'
    xml_file_path = os.path.join(file_folder, 'CellCounter_20x-UC-1-rgb.xml')

    train_positions, calibration = read_Cell_Counter_Maker_XML_file(xml_file_path)

    image_file_path = os.path.join(file_folder, '20x-UC-1-rgb.tif')
    train_image = tifffile.imread(image_file_path)

    ##########################################
    ### Step 2 - Extract features from training data
    features, labels = extract_cell_features(train_image, train_positions, patch_size=64)
    
    ##########################################
    ### Step 3 - Train classifier 
    clf, scaler = train_cell_classifier(features, labels)
    
    ##########################################
    ### Step 4 - Load test image
    image_file_path = os.path.join(file_folder, '20x-UC-4-rgb.tif')
    test_image = tifffile.imread(image_file_path)

    ##########################################
    ### Step 4.1 - Get detection result from trained classifier
    detections = detect_cells(test_image, clf, scaler, patch_size=64)
    
    # filenames1 = save_detection_results(
    #     detections,
    #     base_filename="experiment_1-raw_detection-results"
    # )


    ##########################################
    ### Step 4.2 - Refine the detection results
    # refined_detections = separate_close_cells(detections, 
    #                                         test_image.shape,
    #                                         min_distance=30)
    
    # # Save results
    # filenames2 = save_detection_results(
    #     detections,
    #     base_filename="experiment_1-refined_detection-30-results"
    # )

    # Analyze results
    # results = analyze_results(detections, test_image.shape)
    # filenames = 'experiment_1-refined_detection-30-results.pkl'
    # refined_detections = load_detection_results(filenames)

    ##########################################
    ## Step 5 - Visualize the results
    fig = visualize_detection_results(test_image, detections)


    # Save statistical analysis of height distribution
    output_folder = 'outputs//test_1'
    output_file_path = os.path.join(output_folder, f'20x-UC-4-rgb-before_refined-detection-64-raw.jpg')
    fig.savefig(output_file_path,
            dpi=300,
            bbox_inches='tight',     # removes extra white space
            pad_inches=0.1,          # adds small padding
            format='jpg',
            transparent=True)        # transparent background