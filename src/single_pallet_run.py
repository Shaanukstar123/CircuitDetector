import sys

import cv2
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

from generate_data import get_exact_pallet_roi
from detector import get_global_predictions
from stock_index import assign_stock_indices

def evaluate_strict_predictions(final_predictions, csv_path, tolerance=0.0085):
    """
    Evaluates both spatial accuracy AND temporal (stock_index) accuracy.
    final_predictions: List of dicts [{'x_c': 0.5, 'y_c': 0.5, 'stock_index': 1}, ...]
    """
    if not final_predictions:
        print("No predictions to evaluate.")
        return

    # 1. Load Ground Truth
    df = pd.read_csv(csv_path, header=None, names=['x_c', 'y_c', 'stock_index'])
    gt_points = df[['x_c', 'y_c']].values
    gt_stocks = df['stock_index'].values

    pred_points = np.array([[p['x_c'], p['y_c']] for p in final_predictions])
    pred_stocks = np.array([p['stock_index'] for p in final_predictions])

    # 2. Hungarian Matching (Spatial)
    distance_matrix = cdist(pred_points, gt_points)
    pred_indices, gt_indices = linear_sum_assignment(distance_matrix)

    # 3. Calculate Metrics
    spatial_matches = 0
    strict_matches = 0

    for p_idx, g_idx in zip(pred_indices, gt_indices):
        if distance_matrix[p_idx, g_idx] <= tolerance:
            spatial_matches += 1
            # Check if it also got the timing right!
            if pred_stocks[p_idx] == gt_stocks[g_idx]:
                strict_matches += 1

    precision = strict_matches / len(pred_points) if len(pred_points) > 0 else 0
    recall = strict_matches / len(gt_points) if len(gt_points) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\n--- FINAL EVALUATION ---")
    print(f"Ground Truth Pockets: {len(gt_points)}")
    print(f"Predicted Pockets:    {len(pred_points)}")
    print(f"----------------------------------")
    print(f"Spatial Matches (Location only): {spatial_matches}")
    print(f"Strict Matches (Location + Time): {strict_matches}")
    print(f"False Positives (Hallucinations/Wrong Time): {len(pred_points) - strict_matches}")
    print(f"False Negatives (Missed): {len(gt_points) - strict_matches}")
    print(f"----------------------------------")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1_score:.4f}")

def run_full_pipeline(pallet_dir, model_path, results_csv_dir="../results/results_csv"):
    pallet_path = Path(pallet_dir)
    results_dir_path = Path(results_csv_dir)
    results_dir_path.mkdir(parents=True, exist_ok=True)
    
    test_img = pallet_path / "pallet_capture.jpg"
    metadata_path = pallet_path / "pallet_metadata.json"
    gt_csv_path = pallet_path / "locations.csv"

    # Cropping and Matrix Extraction
    print("\n--- PHASE 1: ROI Extraction ---")
    cropped_img, crop_bbox, M = get_exact_pallet_roi(str(test_img), str(metadata_path))
    x_min, y_min, _, _ = crop_bbox

    temp_crop_path = "temp_crop.jpg"
    cv2.imwrite(temp_crop_path, cropped_img)

    # YOLO Spatial Inference (Returns coordinates relative to cropped image)
    print("--- PHASE 2: Spatial Detection ---")
    yolo_centers = get_global_predictions(temp_crop_path, model_path, conf_thresh=0.75)
    
    # Assign time indices using the stock images
    print("--- PHASE 3: Temporal Association ---")
    temporal_results = assign_stock_indices(yolo_centers, pallet_dir=pallet_path)

    # Canonical Transformation
    print("--- PHASE 4: Canonical Mapping ---")
    with open(metadata_path, 'r') as f:
        M_flat = json.load(f)
    M_orig = np.array(M_flat).reshape(3, 3)
    if abs(M_orig[0, 0]) < 1000:
        M_orig = np.linalg.inv(M_orig)
    M_inv = np.linalg.inv(M_orig)

    final_predictions = []
    for res in temporal_results:
        # Shift back to the global canonical space
        global_x = res['x'] + x_min
        global_y = res['y'] + y_min
        
        pixel_point = np.array([global_x, global_y, 1.0])
        canonical_point = M_inv @ pixel_point
        
        # Clip to ensure bounds are strictly between [0.0, 1.0]
        c_x = np.clip(canonical_point[0] / canonical_point[2], 0.0, 1.0)
        c_y = np.clip(canonical_point[1] / canonical_point[2], 0.0, 1.0)
        
        final_predictions.append({
            'x_c': c_x,
            'y_c': c_y,
            'stock_index': res['stock_index']
        })

    # Save output CSV to the results folder
    df_out = pd.DataFrame(final_predictions)
    if not df_out.empty:
        df_out = df_out[['x_c', 'y_c', 'stock_index']] 
        
        # Sorting output (Row-by-Row, Left-to-Right)

        # Temporary rounded Y column to group items into horizontal rows
        df_out['y_round'] = df_out['y_c'].round(2)
        # Sort by the rounded Y (top to bottom), then by X (left to right)
        df_out = df_out.sort_values(by=['y_round', 'x_c'])
        # Drop temporary column before saving
        df_out = df_out.drop(columns=['y_round'])
        
        output_csv = results_dir_path / f"{pallet_path.name}.csv"
        df_out.to_csv(output_csv, header=False, index=False)
        print(f"Saved final predictions to {output_csv}")
    else:
        print("No predictions generated to save.")
    
    evaluate_strict_predictions(final_predictions, gt_csv_path, tolerance=0.0085)

def main():
    if len(sys.argv) < 2:
        print("Usage: python single_eval.py <pallet_number>")
        print("Example: python single_eval.py 24764")
        sys.exit(1)

    raw_input = sys.argv[1]
    pallet_number = raw_input.replace("pallet_", "")
    base_dir = Path("../data/all_data") 
    pallet_dir = base_dir / f"pallet_{pallet_number}"
    model_path = "../runs/detect/pallet_model_v1/weights/best.pt"

    if not pallet_dir.is_dir():
        print(f"Error: Directory {pallet_dir} does not exist.")
        sys.exit(1)
        
    if not Path(model_path).exists():
        print(f"Error: Model not found at {model_path}")
        sys.exit(1)

    print(f"Starting pipeline for Pallet: {pallet_number}")
    run_full_pipeline(pallet_dir, model_path)

if __name__ == "__main__":
    main()