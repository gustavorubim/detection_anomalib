from pathlib import Path
import torch
from anomalib.data import Folder
from anomalib.models import (
    Dfm, EfficientAd, Padim, Patchcore,
    ReverseDistillation, Stfpm, Supersimplenet, Uflow,
    Cflow, Csflow, Draem, Fastflow, Ganomaly, 
    Dfkde, WinClip, Cfa, Dsr
)
from anomalib.engine import Engine
from anomalib.utils.post_processing import superimpose_anomaly_map
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import gc  # Garbage collection for memory management
import matplotlib

# Increase font sizes for better readability
matplotlib.rcParams.update({'font.size': 14})

# Step 1: Define the path to your dataset
dataset_root = "./datasets"

# Step 2: Configure the datamodule using the Folder class
datamodule = Folder(
    name="custom_dataset",
    root=dataset_root,
    normal_dir="train/good",
    abnormal_dir="test"
)

# Step 3: Define the mapping of algorithm names to their respective model classes and checkpoint paths
model_info = {
    # "CFA": {
    #     "class": Cfa,
    #     "checkpoint": Path("results/CFA/v1/Cfa/custom_dataset/latest/weights/lightning/model.ckpt"),
    # },
    "CFlow": {
        "class": Cflow,
        "checkpoint": Path("results/CFlow/v1/Cflow/custom_dataset/latest/weights/lightning/model.ckpt"),
    },
    "CS-Flow": {
        "class": Csflow,
        "checkpoint": Path("results/CS-Flow/v1/Csflow/custom_dataset/latest/weights/lightning/model.ckpt"),
    },
    "DFKDE": {
        "class": Dfkde,
        "checkpoint": Path("results/DFKDE/v1/Dfkde/custom_dataset/latest/weights/lightning/model.ckpt"),
    },
    "DFM": {
        "class": Dfm,
        "checkpoint": Path("results/DFM/v1/Dfm/custom_dataset/latest/weights/lightning/model.ckpt"),
    },
    "DRAEM": {
        "class": Draem,
        "checkpoint": Path("results/DRAEM/v1/Draem/custom_dataset/latest/weights/lightning/model.ckpt"),
    },
    "DSR": {
        "class": Dsr,
        "checkpoint": Path("results/DSR/v1/Dsr/custom_dataset/latest/weights/lightning/model.ckpt"),
    },
    "EfficientAD": {
        "class": EfficientAd,
        "checkpoint": Path("results/EfficientAD/v1/EfficientAd/custom_dataset/latest/weights/lightning/model.ckpt"),
    },
    "GANomaly": {
        "class": Ganomaly,
        "checkpoint": Path("results/GANomaly/v1/Ganomaly/custom_dataset/latest/weights/lightning/model.ckpt"),
    },
    "PaDiM": {
        "class": Padim,
        "checkpoint": Path("results/PaDiM/v1/Padim/custom_dataset/latest/weights/lightning/model.ckpt"),
    },
    "PatchCore": {
        "class": Patchcore,
        "checkpoint": Path("results/PatchCore/v1/Patchcore/custom_dataset/latest/weights/lightning/model.ckpt"),
    },
    "ReverseDistillation": {
        "class": ReverseDistillation,
        "checkpoint": Path("results/ReverseDistillation/v1/ReverseDistillation/custom_dataset/latest/weights/lightning/model.ckpt"),
    },
    "STFPM": {
        "class": Stfpm,
        "checkpoint": Path("results/STFPM/v1/Stfpm/custom_dataset/latest/weights/lightning/model.ckpt"),
    },
    "U-Flow": {
        "class": Uflow,
        "checkpoint": Path("results/U-Flow/v1/Uflow/custom_dataset/latest/weights/lightning/model.ckpt"),
    },
}

# Step 4: Set up the engine
engine = Engine(accelerator="auto", devices=1, logger=False)

# Step 5: Define the image path for prediction
dataset_root = Path("./datasets")
data_path = dataset_root / "test" / "000_fold.png"

# Step 6: Create a figure with a grid of subplots - one row per model, 4 columns per row
num_models = len(model_info)

# Much larger figure size for better resolution
# Increase height per model and overall width
fig, axes = plt.subplots(num_models, 4, figsize=(24, 5 * num_models))

# Give more space at the top for the title
fig.subplots_adjust(top=0.9)  # This leaves 10% of the figure height for the title area

# Set the title for the entire figure - larger font and positioned higher
fig.suptitle("Anomaly Detection Results Comparison", fontsize=28, y=0.95)

# Set column titles (create a separate row for them)
col_titles = ["Original Image", "Anomaly Map", "Heat Map", "Predicted Mask"]
for col, title in enumerate(col_titles):
    fig.text(0.125 + col * 0.25, 0.91, title, ha='center', va='center', fontsize=20, fontweight='bold')

# Create a list to track which models were processed successfully
successful_models = []

# Step 7: Iterate through each model and perform prediction
for i, (model_name, info) in enumerate(model_info.items()):
    print(f"Processing model {i+1}/{num_models}: {model_name}")
    
    # Store prediction information
    pred_info = {"name": model_name, "success": False, "score": None, "label": None}
    
    try:
        # Load the model if checkpoint exists
        checkpoint_path = info["checkpoint"]
        if checkpoint_path.exists():
            model = info["class"].load_from_checkpoint(checkpoint_path)
            
            # Perform prediction
            predictions = engine.predict(model=model, data_path=data_path)
            prediction = predictions[0]  # Get the first prediction
            
            # Get the original image
            image_path = prediction.image_path[0]
            image_size = prediction.image.shape[-2:]
            original_image = np.array(Image.open(image_path).resize(image_size))
            
            # Get prediction score and label if available
            if hasattr(prediction, 'pred_score') and prediction.pred_score is not None:
                pred_score = prediction.pred_score[0].item()
                pred_label = "Anomalous" if prediction.pred_label[0].item() else "Normal" if hasattr(prediction, 'pred_label') else "N/A"
                pred_info["score"] = pred_score
                pred_info["label"] = pred_label
            
            # Plot original image (first column)
            axes[i, 0].imshow(original_image)
            axes[i, 0].axis('off')
            
            # Check if the model has an anomaly map and plot it (second column)
            has_anomaly_map = hasattr(prediction, 'anomaly_map') and prediction.anomaly_map is not None
            
            if has_anomaly_map:
                try:
                    anomaly_map = prediction.anomaly_map[0].cpu().numpy().squeeze()
                    axes[i, 1].imshow(anomaly_map, cmap='viridis')
                    
                    # Plot heat map (third column)
                    heat_map = superimpose_anomaly_map(anomaly_map=anomaly_map, image=original_image, normalize=True)
                    axes[i, 2].imshow(heat_map)
                except Exception as e:
                    print(f"Error plotting anomaly map for {model_name}: {e}")
                    axes[i, 1].text(0.5, 0.5, 'No Anomaly Map', 
                                    ha='center', va='center', fontsize=14)
                    axes[i, 2].text(0.5, 0.5, 'No Heat Map', 
                                    ha='center', va='center', fontsize=14)
            else:
                # If no anomaly map, show empty box with text
                axes[i, 1].text(0.5, 0.5, 'No Anomaly Map', 
                                ha='center', va='center', fontsize=14)
                axes[i, 2].text(0.5, 0.5, 'No Heat Map', 
                                ha='center', va='center', fontsize=14)
            
            axes[i, 1].axis('off')
            axes[i, 2].axis('off')
            
            # Plot predicted mask (fourth column)
            if hasattr(prediction, 'pred_mask') and prediction.pred_mask is not None:
                try:
                    pred_mask = prediction.pred_mask[0].squeeze().cpu().numpy()
                    axes[i, 3].imshow(pred_mask, cmap='viridis')
                except Exception as e:
                    print(f"Error plotting pred mask for {model_name}: {e}")
                    axes[i, 3].text(0.5, 0.5, 'No Predicted Mask', 
                                    ha='center', va='center', fontsize=14)
            else:
                # If no pred mask, show empty box with text
                axes[i, 3].text(0.5, 0.5, 'No Predicted Mask', 
                                ha='center', va='center', fontsize=14)
            
            axes[i, 3].axis('off')
            
            # Mark as successful
            pred_info["success"] = True
            successful_models.append(pred_info)
            
            # Free memory
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            gc.collect()
            
        else:
            print(f"Checkpoint not found for {model_name}: {checkpoint_path}")
            
            # Mark all plots as 'Checkpoint not found'
            for j in range(4):
                axes[i, j].text(0.5, 0.5, 'Checkpoint not found', 
                                ha='center', va='center', fontsize=14)
                axes[i, j].axis('off')
                
    except Exception as e:
        print(f"Error processing model {model_name}: {e}")
        
        # Mark all plots as 'Error'
        for j in range(4):
            axes[i, j].text(0.5, 0.5, f'Error: {str(e)[:20]}...', 
                            ha='center', va='center', fontsize=14)
            axes[i, j].axis('off')
    
    # Add model name and score to the left of the row
    # Use a text box with bold font and background for better visibility
    if pred_info["score"] is not None:
        label_text = f"{model_name}\nScore: {pred_info['score']:.4f}, Label: {pred_info['label']}"
    else:
        label_text = f"{model_name}"
    
    # Add a clear text box for model information
    axes[i, 0].text(-0.2, 0.5, label_text, 
                    transform=axes[i, 0].transAxes,
                    fontsize=14, fontweight='bold', 
                    ha='right', va='center',
                    bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))

# Adjust subplot spacing
plt.tight_layout()
plt.subplots_adjust(top=0.88, left=0.15, wspace=0.05, hspace=0.1)

# Save a high-resolution image
plt.savefig('anomaly_detection_comparison.png', dpi=300, bbox_inches='tight')

# Show the plot
plt.show()

# Print a summary of model performance
print("\nModel Performance Summary:")
for model in successful_models:
    print(f"{model['name']}: Score = {model['score']:.4f}, Label = {model['label']}")

####

# Additional code to process all images in specified directories and create a summary
# More efficient approach to process all images across models

import os
import pandas as pd
from tqdm import tqdm
import time

# Define the directories to process
base_dirs = ["test", "train/good", "good_for_test"]

# Create output directory if it doesn't exist
output_base_dir = "inference_analysis"
os.makedirs(output_base_dir, exist_ok=True)

# Create a dictionary to store all results
# Structure: {image_path: {model_name: {'score': score, 'label': label, 'predictions': prediction_object}}}
all_results = {}

# First gather all image paths
all_images = []
for base_dir in base_dirs:
    input_dir = os.path.join(dataset_root, base_dir)
    output_dir = os.path.join(output_base_dir, base_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all image files in the directory
    img_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']
    for img_file in os.listdir(input_dir):
        if os.path.splitext(img_file)[1].lower() in img_extensions:
            all_images.append({
                'path': os.path.join(input_dir, img_file),
                'base_dir': base_dir,
                'file_name': img_file
            })

print(f"Found {len(all_images)} images to process across all directories")

# Process each model
for model_name, info in model_info.items():
    print(f"\nProcessing model: {model_name}")
    start_time = time.time()
    
    try:
        # Load the model if checkpoint exists
        checkpoint_path = info["checkpoint"]
        if checkpoint_path.exists():
            model = info["class"].load_from_checkpoint(checkpoint_path)
            
            # Process all images with this model
            for img_info in tqdm(all_images, desc=f"Processing {model_name}"):
                img_path = img_info['path']
                img_key = f"{img_info['base_dir']}/{img_info['file_name']}"
                
                # Initialize entry in results dict if not exists
                if img_key not in all_results:
                    all_results[img_key] = {}
                
                try:
                    # Perform prediction
                    predictions = engine.predict(model=model, data_path=img_path)
                    prediction = predictions[0]  # Get the first prediction
                    
                    # Get prediction score and label if available
                    results = {'prediction': prediction}
                    if hasattr(prediction, 'pred_score') and prediction.pred_score is not None:
                        results['score'] = prediction.pred_score[0].item()
                        if hasattr(prediction, 'pred_label') and prediction.pred_label is not None:
                            results['label'] = "Anomalous" if prediction.pred_label[0].item() else "Normal"
                        else:
                            results['label'] = "N/A"
                    
                    # Store the results
                    all_results[img_key][model_name] = results
                    
                except Exception as e:
                    print(f"  Error processing image {img_key} with model {model_name}: {e}")
                    all_results[img_key][model_name] = {'error': str(e)}
            
            # Free memory
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            gc.collect()
            
        else:
            print(f"Checkpoint not found for {model_name}: {checkpoint_path}")
            
    except Exception as e:
        print(f"Error loading model {model_name}: {e}")
    
    elapsed_time = time.time() - start_time
    print(f"Completed model {model_name} in {elapsed_time:.2f} seconds")

# Create a DataFrame for the CSV output
results_df = pd.DataFrame(columns=['image_file'])

# Generate plots for each image and populate the DataFrame
print("\nGenerating visualizations and CSV...")
for img_key, model_results in tqdm(all_results.items()):
    # Parse the image key
    base_dir, img_file = img_key.split('/', 1)
    img_path = os.path.join(dataset_root, base_dir, img_file)
    output_dir = os.path.join(output_base_dir, base_dir)
    output_file = os.path.splitext(img_file)[0] + "_inference.png"
    output_path = os.path.join(output_dir, output_file)
    
    # Create a row for the results DataFrame
    result_row = {'image_file': img_key}
    
    # Create a figure with a grid of subplots
    fig, axes = plt.subplots(num_models, 4, figsize=(24, 5 * num_models))
    
    # Set the title with image name
    fig.suptitle(f"Anomaly Detection Results Comparison - {img_file}", fontsize=28, y=0.95)
    
    # Set column titles
    col_titles = ["Original Image", "Anomaly Map", "Heat Map", "Predicted Mask"]
    for col, title in enumerate(col_titles):
        fig.text(0.125 + col * 0.25, 0.91, title, ha='center', va='center', fontsize=20, fontweight='bold')
    
    # Cache original image - we only need to load it once per image
    original_image = None
    
    # Iterate through each model in the same order as model_info
    for i, (model_name, _) in enumerate(model_info.items()):
        if model_name in model_results:
            results = model_results[model_name]
            
            # Add to DataFrame
            if 'score' in results:
                result_row[f"{model_name}_score"] = results['score']
                result_row[f"{model_name}_label"] = results['label']
            
            # Check if we have prediction data
            if 'prediction' in results:
                prediction = results['prediction']
                
                # Get the original image if not cached yet
                if original_image is None and hasattr(prediction, 'image_path'):
                    image_path = prediction.image_path[0]
                    image_size = prediction.image.shape[-2:]
                    original_image = np.array(Image.open(image_path).resize(image_size))
                
                # Plot original image (first column)
                if original_image is not None:
                    axes[i, 0].imshow(original_image)
                    axes[i, 0].axis('off')
                else:
                    axes[i, 0].text(0.5, 0.5, 'Image Error', ha='center', va='center', fontsize=14)
                    axes[i, 0].axis('off')
                
                # Check if the model has an anomaly map and plot it (second column)
                has_anomaly_map = hasattr(prediction, 'anomaly_map') and prediction.anomaly_map is not None
                
                if has_anomaly_map:
                    try:
                        anomaly_map = prediction.anomaly_map[0].cpu().numpy().squeeze()
                        axes[i, 1].imshow(anomaly_map, cmap='viridis')
                        
                        # Plot heat map (third column) if we have the original image
                        if original_image is not None:
                            heat_map = superimpose_anomaly_map(anomaly_map=anomaly_map, image=original_image, normalize=True)
                            axes[i, 2].imshow(heat_map)
                        else:
                            axes[i, 2].text(0.5, 0.5, 'No Heat Map (Image Error)', 
                                         ha='center', va='center', fontsize=14)
                    except Exception as e:
                        axes[i, 1].text(0.5, 0.5, 'No Anomaly Map', 
                                     ha='center', va='center', fontsize=14)
                        axes[i, 2].text(0.5, 0.5, 'No Heat Map', 
                                     ha='center', va='center', fontsize=14)
                else:
                    # If no anomaly map, show empty box with text
                    axes[i, 1].text(0.5, 0.5, 'No Anomaly Map', 
                                 ha='center', va='center', fontsize=14)
                    axes[i, 2].text(0.5, 0.5, 'No Heat Map', 
                                 ha='center', va='center', fontsize=14)
                
                axes[i, 1].axis('off')
                axes[i, 2].axis('off')
                
                # Plot predicted mask (fourth column)
                if hasattr(prediction, 'pred_mask') and prediction.pred_mask is not None:
                    try:
                        pred_mask = prediction.pred_mask[0].squeeze().cpu().numpy()
                        axes[i, 3].imshow(pred_mask, cmap='viridis')
                    except Exception as e:
                        axes[i, 3].text(0.5, 0.5, 'No Predicted Mask', 
                                     ha='center', va='center', fontsize=14)
                else:
                    # If no pred mask, show empty box with text
                    axes[i, 3].text(0.5, 0.5, 'No Predicted Mask', 
                                 ha='center', va='center', fontsize=14)
                
                axes[i, 3].axis('off')
            
            else:
                # If we have an error for this model
                error_msg = results.get('error', 'Unknown error')
                for j in range(4):
                    axes[i, j].text(0.5, 0.5, f'Error: {error_msg[:20]}...', 
                                 ha='center', va='center', fontsize=14)
                    axes[i, j].axis('off')
        
        else:
            # If we don't have results for this model
            for j in range(4):
                axes[i, j].text(0.5, 0.5, 'No results', 
                             ha='center', va='center', fontsize=14)
                axes[i, j].axis('off')
        
        # Add model name and score to the left of the row
        score = results.get('score') if model_name in model_results and 'score' in model_results[model_name] else None
        label = results.get('label') if model_name in model_results and 'label' in model_results[model_name] else None
        
        label_text = f"{model_name}"
        if score is not None:
            label_text += f"\nScore: {score:.4f}, Label: {label}"
        
        # Add a clear text box for model information
        axes[i, 0].text(-0.2, 0.5, label_text, 
                        transform=axes[i, 0].transAxes,
                        fontsize=14, fontweight='bold', 
                        ha='right', va='center',
                        bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    # Adjust subplot spacing
    plt.tight_layout()
    plt.subplots_adjust(top=0.88, left=0.15, wspace=0.05, hspace=0.1)
    
    # Save the figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)  # Close the figure to free memory
    
    # Add the results row to the DataFrame
    results_df = pd.concat([results_df, pd.DataFrame([result_row])], ignore_index=True)

# Save the results DataFrame to a CSV file
csv_path = os.path.join(output_base_dir, "anomaly_detection_results.csv")
results_df.to_csv(csv_path, index=False)
print(f"\nSummary saved to {csv_path}")

# Print some statistics
print("\nSummary Statistics:")
for model_name in model_info.keys():
    if f"{model_name}_score" in results_df.columns:
        avg_score = results_df[f"{model_name}_score"].mean()
        anomalous_count = results_df[results_df[f"{model_name}_label"] == "Anomalous"].shape[0]
        normal_count = results_df[results_df[f"{model_name}_label"] == "Normal"].shape[0]
        total_count = anomalous_count + normal_count
        
        print(f"{model_name}:")
        print(f"  Average Score: {avg_score:.4f}")
        print(f"  Classified as Anomalous: {anomalous_count}/{total_count} ({anomalous_count/total_count*100:.1f}%)")
        print(f"  Classified as Normal: {normal_count}/{total_count} ({normal_count/total_count*100:.1f}%)")