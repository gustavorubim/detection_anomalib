def main():
    # 1. Import required modules
    import torch
    from pathlib import Path
    import logging
    import cv2
    import numpy as np
    import os
    import re
    import csv

    from anomalib.data import PredictDataset
    from anomalib.engine import Engine
    from anomalib.models import (
        Cfa, Cflow, Csflow, Dfkde, Dfm, Draem,
        EfficientAd, Fastflow, Ganomaly, Padim, Patchcore,
        ReverseDistillation, Stfpm, Supersimplenet, Uflow, WinClip, Dsr
    )

    # Set matrix multiplication precision for CUDA devices with Tensor Cores
    torch.set_float32_matmul_precision('high')

    # Create a dictionary to store anomaly detection results for the CSV
    # Dictionary mapping model names to their respective classes and directory names
    MODELS = {
        "CFA": {"class": Cfa, "dir_names": ["Cfa", "CFA"]},
        "CFlow": {"class": Cflow, "dir_names": ["C-Flow", "CFlow", "Cflow"]},
        "CS-Flow": {"class": Csflow, "dir_names": ["CS-Flow", "CSFlow", "Csflow"]},
        "DFKDE": {"class": Dfkde, "dir_names": ["DFKDE", "Dfkde"]},
        "DFM": {"class": Dfm, "dir_names": ["DFM", "Dfm"]},
        "DRAEM": {"class": Draem, "dir_names": ["DRAEM", "Draem"]},
        "EfficientAD": {"class": EfficientAd, "dir_names": ["EfficientAD", "EfficientAd"]},
        "GANomaly": {"class": Ganomaly, "dir_names": ["GANomaly", "Ganomaly"]},
        "PaDiM": {"class": Padim, "dir_names": ["PaDiM", "Padim"]},
        "PatchCore": {"class": Patchcore, "dir_names": ["PatchCore", "Patchcore"]},
        "ReverseDistillation": {"class": ReverseDistillation, "dir_names": ["ReverseDistillation"]},
        "STFPM": {"class": Stfpm, "dir_names": ["STFPM", "Stfpm"]},
        "U-Flow": {"class": Uflow, "dir_names": ["U-Flow", "UFlow", "Uflow"]},
        "DSR": {"class": Dsr, "dir_names": ["DSR", "Dsr"]}
    }

    # Create a dictionary to store anomaly detection results for the CSV
    anomaly_results = {}
    
    # List to store all model names in order they were processed
    model_names_list = []
    
    # Function to track and record anomaly detection results for the CSV
    def record_anomaly_result(model_name, image_name, is_anomaly, score=None):
        # Initialize entry for this image if it doesn't exist
        if image_name not in anomaly_results:
            anomaly_results[image_name] = {}
        
        # Record the detection result
        anomaly_results[image_name][model_name] = {
            "is_anomaly": is_anomaly,
            "score": score if score is not None else "N/A"
        }

    # Configure logging to reduce warning noise
    logging.getLogger("anomalib.visualization.image.item_visualizer").setLevel(logging.ERROR)
    
    # Create engine for predictions
    engine = Engine()

    # Test folder path - use your custom dataset structure
    dataset_path = "./datasets"
    test_folder = Path(dataset_path) / "test"
    
    # Create output directory
    output_base_dir = Path("./output_visualizations")
    output_base_dir.mkdir(exist_ok=True)
    
    # Function to find the latest checkpoint for a model
    def find_latest_checkpoint(model_name, dir_names):
        # Base results directory
        results_dir = Path("./results")
        
        print(f"  Looking for checkpoints for model: {model_name}")
        print(f"  Possible directory names: {dir_names}")
        
        # Find all possible checkpoint paths for this model
        checkpoint_paths = []
        
        # Check for version directories within model directories
        for dir_name in dir_names:
            model_dir = results_dir / dir_name / "v1"  # Using v1 as per your training script
            if model_dir.exists():
                print(f"  Checking directory: {model_dir}")
                
                # Find model.ckpt files in this directory and subdirectories
                for ckpt_path in model_dir.glob("**/model.ckpt"):
                    checkpoint_paths.append((ckpt_path, 1))  # Version 1
                    print(f"  Found checkpoint: {ckpt_path}")
        
        # If no direct matches in v1, try v2, v3, etc.
        if not checkpoint_paths:
            for version in range(2, 11):  # Try up to v10
                for dir_name in dir_names:
                    model_dir = results_dir / dir_name / f"v{version}"
                    if model_dir.exists():
                        print(f"  Checking directory: {model_dir}")
                        
                        for ckpt_path in model_dir.glob("**/model.ckpt"):
                            checkpoint_paths.append((ckpt_path, version))
                            print(f"  Found checkpoint: {ckpt_path}")
        
        # If still no matches, try a more flexible approach without version folders
        if not checkpoint_paths:
            print(f"  No version-specific matches found, trying flexible matching")
            for dir_name in dir_names:
                model_dir = results_dir / dir_name
                if model_dir.exists():
                    print(f"  Checking directory: {model_dir}")
                    
                    for ckpt_path in model_dir.glob("**/model.ckpt"):
                        checkpoint_paths.append((ckpt_path, 0))  # Version 0 for non-versioned
                        print(f"  Found checkpoint: {ckpt_path}")
        
        # If still no matches, look for directories containing the model name
        if not checkpoint_paths:
            print(f"  No direct matches found, trying partial name matching")
            for ckpt_path in results_dir.glob("**/model.ckpt"):
                path_str = str(ckpt_path)
                
                # Check if any of the model directory names appear in the path
                for dir_name in dir_names:
                    if dir_name.lower() in path_str.lower():
                        checkpoint_paths.append((ckpt_path, 0))
                        print(f"  Found matching checkpoint via flexible search: {ckpt_path}")
                        break
        
        if not checkpoint_paths:
            print(f"  No checkpoint found for {model_name}")
            return None
        
        # Sort by version number and get the latest
        latest_ckpt = sorted(checkpoint_paths, key=lambda x: x[1], reverse=True)[0][0]
        print(f"  Using checkpoint for {model_name}: {latest_ckpt}")
        return latest_ckpt
    
    # Create fallback visualization when anomaly map is missing
    def create_fallback_visualization(model_name, dataset, output_dir, test_folder_path):
        print(f"  Creating fallback visualizations for {model_name}")
        
        # Try to get image paths from the dataset or directory
        try:
            # First attempt to get the image paths from the dataset
            image_paths = []
            for i, item in enumerate(dataset):
                # Try different ways to access the image path based on the item type
                try:
                    if hasattr(item, 'image_path'):
                        # If it's an object with image_path attribute
                        image_path = item.image_path
                        if isinstance(image_path, list):
                            image_path = image_path[0]
                        image_paths.append(image_path)
                    elif isinstance(item, tuple) or isinstance(item, list):
                        # If it's a tuple or list
                        image_path = item[0]
                        image_paths.append(image_path)
                    else:
                        # Try string representation
                        image_path = str(item)
                        if os.path.exists(image_path):
                            image_paths.append(image_path)
                except Exception as e:
                    print(f"  Error accessing item {i}: {str(e)}")
            
            # If we couldn't get any paths, fall back to listing files in the directory
            if not image_paths:
                print(f"  Could not get image paths from dataset, scanning directory")
                image_paths = [str(f) for f in test_folder_path.glob("**/*.png")]
                if not image_paths:
                    image_paths = [str(f) for f in test_folder_path.glob("**/*.jpg")]
        except Exception as e:
            print(f"  Error accessing dataset: {str(e)}, scanning directory instead")
            image_paths = [str(f) for f in test_folder_path.glob("**/*.png")]
            if not image_paths:
                image_paths = [str(f) for f in test_folder_path.glob("**/*.jpg")]
        
        for i, image_path in enumerate(image_paths):
            # Get original image
            original_image = cv2.imread(image_path)
            if original_image is None:
                print(f"  Error: Could not read image from {image_path}")
                continue
                
            # Get base filename without extension
            base_filename = os.path.basename(image_path)
            filename_no_ext = os.path.splitext(base_filename)[0]
            
            # For models without anomaly maps, create a visual gradient based on image number 
            # (since we don't have scores, this is just for visualization)
            h, w = original_image.shape[:2]
            
            # Randomly generate a score for visualization purposes (0.3-0.7 range)
            np.random.seed(i)  # For reproducibility
            random_score = 0.3 + (0.4 * np.random.random())  # 0.3-0.7 range
            
            # Create a simple gradient image
            gradient_img = np.ones((h, w), dtype=np.uint8) * int(random_score * 255)
            
            # Apply colormap to create gradient
            anomaly_map_colored = cv2.applyColorMap(gradient_img, cv2.COLORMAP_JET)
            
            # Add text explaining this is a fallback visualization
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            text1 = "Fallback Visualization"
            text2 = f"for {model_name}"
            
            # Add darkened rectangle for text visibility
            overlay = anomaly_map_colored.copy()
            cv2.rectangle(overlay, (0, h//2-30), (w, h//2+30), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, anomaly_map_colored, 0.4, 0, anomaly_map_colored)
            
            # Calculate positions (centered)
            text_size1 = cv2.getTextSize(text1, font, font_scale, thickness)[0]
            text_size2 = cv2.getTextSize(text2, font, font_scale, thickness)[0]
            x1 = (w - text_size1[0]) // 2
            x2 = (w - text_size2[0]) // 2
            y1 = h // 2 - 10
            y2 = h // 2 + 20
            
            # Add text with white color for visibility
            cv2.putText(anomaly_map_colored, text1, (x1, y1), font, font_scale, (255, 255, 255), thickness)
            cv2.putText(anomaly_map_colored, text2, (x2, y2), font, font_scale, (255, 255, 255), thickness)
            
            # Add text labels to each image
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            color = (255, 255, 255)
            thickness = 1
            
            # Create copies to avoid modifying originals
            original_with_text = original_image.copy()
            anomaly_map_with_text = anomaly_map_colored.copy()
            
            # Add text at the top of each image
            cv2.putText(original_with_text, "Original", (10, 20), font, font_scale, color, thickness)
            cv2.putText(anomaly_map_with_text, "Anomaly Map (Simulated)", (10, 20), font, font_scale, color, thickness)
            
            # Add model name and prediction status at the top
            prediction_text = "Prediction: Unable to determine"
            cv2.putText(original_with_text, f"Model: {model_name}", (10, 40), font, font_scale, color, thickness)
            cv2.putText(original_with_text, prediction_text, (10, 60), font, font_scale, color, thickness)
            
            # Combine images horizontally - only original and anomaly map
            combined_image = np.hstack((original_with_text, anomaly_map_with_text))
            
            # Save the combined image
            cv2.imwrite(str(output_dir / f"{filename_no_ext}_combined.png"), combined_image)
            
            # Record fallback result for CSV (mark as unable to determine)
            record_anomaly_result(model_name, filename_no_ext, None, "Unable to determine")
            
            if i == 0 or (i+1) % 10 == 0 or i == len(image_paths)-1:  # Print first, every 10th, and last
                print(f"  Image {i+1}/{len(image_paths)}: {filename_no_ext}")
                print(f"    Prediction: Unable to determine (fallback visualization)")
    
    # Function to process predictions and create visualizations
    def process_predictions(model_name, predictions, output_dir, dataset, test_folder_path):
        if predictions is None or len(predictions) == 0:
            print(f"  No predictions returned for {model_name}, creating fallback visualizations")
            create_fallback_visualization(model_name, dataset, output_dir, test_folder_path)
            return
        
        print(f"  Processing {len(predictions)} images")
        for i, prediction in enumerate(predictions):
            try:
                # Get image path - handle both string and list formats
                if isinstance(prediction.image_path, list):
                    image_path = prediction.image_path[0]  # image_path is a list with one element
                else:
                    image_path = prediction.image_path
                
                # Handle pred_label (could be tensor or scalar)
                try:
                    pred_label = prediction.pred_label
                    if hasattr(pred_label, 'item'):
                        pred_label = pred_label.item()
                except (AttributeError, TypeError):
                    pred_label = 0  # Default to normal if we can't get the label
                
                # Handle pred_score (could be tensor or scalar)
                try:
                    if hasattr(prediction.pred_score, 'item'):
                        pred_score = prediction.pred_score.item()  # Image-level anomaly score
                    else:
                        pred_score = prediction.pred_score
                except (AttributeError, TypeError):
                    pred_score = 0.0  # Default if we can't get the score
                
                # Get original image
                original_image = cv2.imread(image_path)
                if original_image is None:
                    print(f"  Error: Could not read image from {image_path}")
                    continue
                    
                # Get base filename without extension
                base_filename = os.path.basename(image_path)
                filename_no_ext = os.path.splitext(base_filename)[0]
                
                # Check if anomaly_map exists and is not None
                if hasattr(prediction, 'anomaly_map') and prediction.anomaly_map is not None:
                    # Get anomaly map/mask and ensure proper format
                    anomaly_map = prediction.anomaly_map
                    
                    # Convert tensor to numpy and ensure proper shape
                    if hasattr(anomaly_map, 'cpu'):
                        anomaly_map_np = anomaly_map.cpu().numpy()
                    else:
                        # Handle the case where anomaly_map is already a numpy array
                        anomaly_map_np = np.array(anomaly_map)
                    
                    # If it's not 2D, reshape or take the first channel
                    if len(anomaly_map_np.shape) > 2:
                        anomaly_map_np = anomaly_map_np[0] if anomaly_map_np.shape[0] > 1 else anomaly_map_np.squeeze()
                    
                    # Ensure it's 2D
                    anomaly_map_np = anomaly_map_np.squeeze()
                    
                    # Normalize to 0-1 range
                    min_val = anomaly_map_np.min()
                    max_val = anomaly_map_np.max()
                    if max_val > min_val:
                        anomaly_map_np = (anomaly_map_np - min_val) / (max_val - min_val)
                    else:
                        anomaly_map_np = np.zeros_like(anomaly_map_np)
                    
                    # Convert to uint8 (0-255 range) for colormap
                    anomaly_map_uint8 = (anomaly_map_np * 255).astype(np.uint8)
                    
                    # Apply colormap
                    anomaly_map_colored = cv2.applyColorMap(anomaly_map_uint8, cv2.COLORMAP_JET)
                    
                    # Resize anomaly map to match original image size if needed
                    if anomaly_map_colored.shape[:2] != original_image.shape[:2]:
                        anomaly_map_colored = cv2.resize(
                            anomaly_map_colored, 
                            (original_image.shape[1], original_image.shape[0]), 
                            interpolation=cv2.INTER_LINEAR
                        )
                    
                else:
                    # Create informative placeholder for models that don't provide anomaly maps
                    print(f"  Warning: No anomaly map available for {filename_no_ext} with model {model_name}")
                    
                    # Create a gray background with text
                    anomaly_map_colored = np.ones_like(original_image) * 200  # Light gray background
                    
                    # Add text explaining the missing anomaly map
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.7
                    thickness = 2
                    text1 = "Anomaly Map"
                    text2 = "Not Available"
                    text3 = f"for {model_name}"
                    
                    # Get text sizes
                    text_size1 = cv2.getTextSize(text1, font, font_scale, thickness)[0]
                    text_size2 = cv2.getTextSize(text2, font, font_scale, thickness)[0]
                    text_size3 = cv2.getTextSize(text3, font, font_scale, thickness)[0]
                    
                    # Calculate positions (centered)
                    h, w = anomaly_map_colored.shape[:2]
                    x1 = (w - text_size1[0]) // 2
                    x2 = (w - text_size2[0]) // 2
                    x3 = (w - text_size3[0]) // 2
                    y1 = h // 2 - 20
                    y2 = h // 2 + 10
                    y3 = h // 2 + 40
                    
                    # Add text to the image
                    cv2.putText(anomaly_map_colored, text1, (x1, y1), font, font_scale, (0, 0, 0), thickness)
                    cv2.putText(anomaly_map_colored, text2, (x2, y2), font, font_scale, (0, 0, 0), thickness)
                    cv2.putText(anomaly_map_colored, text3, (x3, y3), font, font_scale, (0, 0, 0), thickness)
                
                # Add text labels to each image
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                color = (255, 255, 255)
                thickness = 1
                
                # Create copies to avoid modifying originals
                original_with_text = original_image.copy()
                anomaly_map_with_text = anomaly_map_colored.copy()
                
                # Add text at the top of each image
                cv2.putText(original_with_text, "Original", (10, 20), font, font_scale, color, thickness)
                
                # For anomaly map, check if it's a placeholder
                if np.all(anomaly_map_colored > 190) and np.all(anomaly_map_colored < 210):  # Check if it's our light gray background
                    cv2.putText(anomaly_map_with_text, "Anomaly Map (N/A)", (10, 20), font, font_scale, color, thickness)
                else:
                    cv2.putText(anomaly_map_with_text, "Anomaly Map", (10, 20), font, font_scale, color, thickness)
                
                # Create a title that shows both model name and prediction
                title_text = f"Model: {model_name} - {'ANOMALOUS' if pred_label == 1 else 'NORMAL'}"
                score_text = f"Score: {pred_score:.4f}"
                
                # Add a darkened rectangle at the top for better visibility of the title
                cv2.rectangle(original_with_text, (0, 0), (original_with_text.shape[1], 80), (0, 0, 0), -1)
                
                # Add model name and prediction with larger font for emphasis
                title_font_scale = 0.8
                cv2.putText(original_with_text, title_text, (10, 30), font, title_font_scale, (255, 255, 255), 2)
                cv2.putText(original_with_text, score_text, (10, 60), font, font_scale, (255, 255, 255), thickness)
                
                # Combine only original and anomaly map horizontally
                combined_image = np.hstack((original_with_text, anomaly_map_with_text))
                
                # Save the combined image
                cv2.imwrite(str(output_dir / f"{filename_no_ext}_combined.png"), combined_image)
                
                # Record the result for the CSV file
                is_anomaly = pred_label == 1
                record_anomaly_result(model_name, filename_no_ext, is_anomaly, pred_score)
                
                if i == 0 or (i+1) % 10 == 0 or i == len(predictions)-1:  # Print first, every 10th, and last
                    print(f"  Image {i+1}/{len(predictions)}: {filename_no_ext}")
                    print(f"    Prediction: {'Anomalous' if pred_label == 1 else 'Normal'}")
                    print(f"    Anomaly Score: {pred_score:.4f}")
            
            except Exception as e:
                print(f"  Error processing image {i}: {str(e)}")
                continue
    
    # Process each model
    for model_name, model_info in MODELS.items():
        try:
            print(f"\n{'-'*50}")
            print(f"Processing model: {model_name}")
            
            # Add model to the list
            model_names_list.append(model_name)
            
            # Find the latest checkpoint for this model
            checkpoint_path = find_latest_checkpoint(model_name, model_info["dir_names"])
            
            if checkpoint_path is None:
                print(f"  No checkpoint found for {model_name}, skipping...")
                continue
            
            # Create model-specific output directory
            model_output_dir = output_base_dir / model_name
            model_output_dir.mkdir(exist_ok=True)
            print(f"  Saving results to: {model_output_dir}")
            
            # Initialize the model
            model = model_info["class"]()
            
            # Find all test folders containing any images
            test_subfolders = [d for d in test_folder.glob("*") if d.is_dir() and 
                              (list(d.glob("*.png")) or list(d.glob("*.jpg")))]
            
            # If no subfolders with images, use the test folder directly
            if not test_subfolders:
                test_subfolders = [test_folder] if (list(test_folder.glob("*.png")) or 
                                                    list(test_folder.glob("*.jpg"))) else []
            
            # Process each test subfolder
            for test_subfolder in test_subfolders:
                subfolder_name = test_subfolder.name
                print(f"  Processing test subfolder: {subfolder_name}")
                
                # Create subfolder-specific output directory
                subfolder_output_dir = model_output_dir / subfolder_name
                subfolder_output_dir.mkdir(exist_ok=True)
                
                # Prepare test data
                dataset = PredictDataset(
                    path=test_subfolder,
                    image_size=(256, 256),
                )
                
                # Get predictions
                try:
                    predictions = engine.predict(
                        model=model,
                        dataset=dataset,
                        ckpt_path=str(checkpoint_path),
                    )
                    
                    # Process the predictions
                    process_predictions(model_name, predictions, subfolder_output_dir, dataset, test_subfolder)
                    
                except Exception as e:
                    print(f"  Error processing {model_name} for subfolder {subfolder_name}: {str(e)}")
                    # Create fallback visualizations when prediction fails
                    create_fallback_visualization(model_name, dataset, subfolder_output_dir, test_subfolder)
            
            if not test_subfolders:
                print(f"  No test subfolders or images found in {test_folder}")
                
        except Exception as e:
            print(f"  Error with model {model_name}: {str(e)}")
            continue
    
    # Generate the CSV file with anomaly detection results
    csv_output_path = output_base_dir / "anomaly_detection_results.csv"
    print(f"\n{'-'*50}")
    print(f"Generating CSV summary at: {csv_output_path}")
    
    # Get all image names
    all_image_names = sorted(anomaly_results.keys())
    
    # Write CSV file
    with open(csv_output_path, 'w', newline='') as csvfile:
        # Create CSV writer
        writer = csv.writer(csvfile)
        
        # Write header row
        header = ["Image Name"]
        for model_name in model_names_list:
            header.extend([f"{model_name} (Anomaly)", f"{model_name} (Score)"])
        writer.writerow(header)
        
        # Write data rows
        for image_name in all_image_names:
            row = [image_name]
            for model_name in model_names_list:
                # Get anomaly detection result for this model and image
                if model_name in anomaly_results.get(image_name, {}):
                    result = anomaly_results[image_name][model_name]
                    if result["is_anomaly"] is None:
                        row.append("Unknown")
                    else:
                        row.append("Yes" if result["is_anomaly"] else "No")
                    row.append(str(result["score"]))
                else:
                    # If no result for this model, add placeholder values
                    row.extend(["N/A", "N/A"])
            writer.writerow(row)
    
    print(f"CSV generation complete. File saved to: {csv_output_path}")
    print(f"\nAll anomaly detection models processed successfully!")

if __name__ == "__main__":
    main()