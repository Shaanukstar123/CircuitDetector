import cv2
import numpy as np
import torch
from ultralytics import YOLO
from torchvision.ops import nms
import pandas as pd
import json
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from pathlib import Path

def get_global_predictions(image_input, model, patch_size=2048, overlap=0.2, conf_thresh=0.5, iou_thresh=0.4):
    """
    image_input: Can be a file path (str) OR a numpy image array (np.ndarray).
    """
    # Check if the input is a path or an already loaded image array
    if isinstance(image_input, (str, Path)):
        img = cv2.imread(str(image_input))
        if img is None:
            print(f"Error: Could not read image at {image_input}")
            return []
    else:
        # It's already an image array from cv2.imread or a crop
        img = image_input

    h_img, w_img = img.shape[:2]
    
    model = YOLO(model)
    
    stride = int(patch_size * (1 - overlap))
    global_boxes = []
    global_scores = []
    
    # 1. Sliding Window Inference
    print("Running sliding window inference...")
    for y_start in range(0, h_img, stride):
        for x_start in range(0, w_img, stride):
            x_end = x_start + patch_size
            y_end = y_start + patch_size
            
            # Snap-to-edge logic
            if x_end > w_img:
                x_end = w_img
                x_start = max(0, w_img - patch_size)
            if y_end > h_img:
                y_end = h_img
                y_start = max(0, h_img - patch_size)
                
            patch = img[y_start:y_end, x_start:x_end]
            
            # Run YOLO on the patch (imgsz=800 matches training)
            results = model.predict(patch, imgsz=800, conf=conf_thresh, verbose=False)
            
            # 2. Coordinate Shifting
            boxes = results[0].boxes
            for box in boxes:
                # Get xyxy coordinates relative to the PATCH
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                score = box.conf[0].cpu().item()
                
                # Shift coordinates to the GLOBAL 14k image space
                global_x1 = x1 + x_start
                global_y1 = y1 + y_start
                global_x2 = x2 + x_start
                global_y2 = y2 + y_start
                
                global_boxes.append([global_x1, global_y1, global_x2, global_y2])
                global_scores.append(score)
                
            if x_end == w_img: break
        if y_end == h_img: break

    if not global_boxes:
        return []

    # 3. Global Non-Maximum Suppression (Removing Duplicates from Overlaps)
    print(f"Total raw predictions: {len(global_boxes)}. Running NMS...")
    boxes_tensor = torch.tensor(global_boxes, dtype=torch.float32)
    scores_tensor = torch.tensor(global_scores, dtype=torch.float32)
    
    # Keep boxes that don't heavily overlap
    keep_indices = nms(boxes_tensor, scores_tensor, iou_thresh)
    final_boxes = boxes_tensor[keep_indices].numpy()
    
    # 4. Extract Center Points
    centers = []
    for box in final_boxes:
        x1, y1, x2, y2 = box
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        centers.append({'x': cx, 'y': cy})
        
    print(f"Final unique pockets found: {len(centers)}")
    return centers

def evaluate_spatial_predictions(yolo_centers, csv_path, metadata_path, crop_bbox, tolerance=0.0085):
    """
    yolo_centers: List of dicts [{'x': 1500, 'y': 2500}, ...] from the cropped image space.
    tolerance: Max distance in canonical space (0.0 to 1.0) to count as a match. Smallest_pallet_width * 0.5 = 0.0085
    """
    if not yolo_centers:
        print("No predictions to evaluate.")
        return
        
    x_min, y_min, _, _ = crop_bbox
    
    # Load Ground Truth
    df = pd.read_csv(csv_path, header=None, names=['x_c', 'y_c', 'stock_index'])
    gt_points = df[['x_c', 'y_c']].values
    
    # Load and Invert the Matrix
    with open(metadata_path, 'r') as f:
        M_flat = json.load(f)
    M = np.array(M_flat).reshape(3, 3)
    if abs(M[0, 0]) < 1000:
        M = np.linalg.inv(M)
    M_inv = np.linalg.inv(M)
    
    # Convert YOLO predictions back to Canonical [0, 1] Space
    pred_points = []
    for center in yolo_centers:
        # Add crop offset to get back to original high-res pixel space
        orig_x = center['x'] + x_min
        orig_y = center['y'] + y_min
        
        # Homogeneous coordinate multiplication
        pixel_point = np.array([orig_x, orig_y, 1.0])
        canonical_point = M_inv @ pixel_point
        
        # Normalise
        c_x = canonical_point[0] / canonical_point[2]
        c_y = canonical_point[1] / canonical_point[2]
        pred_points.append([c_x, c_y])
        
    pred_points = np.array(pred_points)
    
    # Using The Hungarian Algorithm to match Predictions to Ground Truth
    # Calculate Euclidean distance between all pairs
    distance_matrix = cdist(pred_points, gt_points)
    
    # Find the optimal 1-to-1 matching
    pred_indices, gt_indices = linear_sum_assignment(distance_matrix)
    
    # 5. Calculate Metrics 
    true_positives = 0
    for p_idx, g_idx in zip(pred_indices, gt_indices):
        if distance_matrix[p_idx, g_idx] <= tolerance:
            true_positives += 1
            
    false_positives = len(pred_points) - true_positives
    false_negatives = len(gt_points) - true_positives
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"--- Spatial Evaluation Results ---")
    print(f"Ground Truth Pockets: {len(gt_points)}")
    print(f"Predicted Pockets:    {len(pred_points)}")
    print(f"True Positives (Matched): {true_positives}")
    print(f"False Positives (Hallucinations): {false_positives}")
    print(f"False Negatives (Missed): {false_negatives}")
    print(f"----------------------------------")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1_score:.4f}")
    
    return pred_points
