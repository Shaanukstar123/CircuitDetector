import cv2
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

from generate_data import get_exact_pallet_roi
from detector import get_global_predictions
from stock_index import assign_stock_indices

def evaluate_final_predictions(final_predictions, csv_path, tolerance=0.0085):
    """
    Evaluates predictions and returns raw metric counts for aggregation.
    """
    if not final_predictions:
        return {'pred': 0, 'gt': 0, 'spatial': 0, 'strict': 0}

    df = pd.read_csv(csv_path, header=None, names=['x_c', 'y_c', 'stock_index'])
    gt_points = df[['x_c', 'y_c']].values
    gt_stocks = df['stock_index'].values

    pred_points = np.array([[p['x_c'], p['y_c']] for p in final_predictions])
    pred_stocks = np.array([p['stock_index'] for p in final_predictions])

    distance_matrix = cdist(pred_points, gt_points)
    pred_indices, gt_indices = linear_sum_assignment(distance_matrix)

    spatial_matches = 0
    strict_matches = 0

    for p_idx, g_idx in zip(pred_indices, gt_indices):
        if distance_matrix[p_idx, g_idx] <= tolerance:
            spatial_matches += 1
            if pred_stocks[p_idx] == gt_stocks[g_idx]:
                strict_matches += 1

    return {
        'pred': len(pred_points),
        'gt': len(gt_points),
        'spatial': spatial_matches,
        'strict': strict_matches
    }

def process_single_pallet(pallet_dir, model_path, results_csv_dir):
    """
    Runs the pipeline silently, saves CSV to results, and returns metrics AND raw predictions.
    """
    pallet_path = Path(pallet_dir)
    test_img = pallet_path / "pallet_capture.jpg"
    metadata_path = pallet_path / "pallet_metadata.json"
    gt_csv_path = pallet_path / "locations.csv"

    if not test_img.exists() or not gt_csv_path.exists():
        return None

    # Pallet image crop
    cropped_img, crop_bbox, M = get_exact_pallet_roi(str(test_img), str(metadata_path))
    # h, w = cropped_img.shape[:2]
    # print(f"Smallest component width in pixels: {0.017 * w}")
    x_min, y_min, _, _ = crop_bbox

    temp_crop_path = "temp_crop_eval.jpg"
    cv2.imwrite(temp_crop_path, cropped_img)

    # Component detection using YOLO
    yolo_centers = get_global_predictions(temp_crop_path, model_path, conf_thresh=0.75)
    
    # Time index assignment using stock images
    temporal_results = assign_stock_indices(yolo_centers, pallet_dir=pallet_path)

    # Canonical Transformation
    with open(metadata_path, 'r') as f:
        M_orig = np.array(json.load(f)).reshape(3, 3)
    if abs(M_orig[0, 0]) < 1000:
        M_orig = np.linalg.inv(M_orig)
    M_inv = np.linalg.inv(M_orig)

    final_predictions = []
    for res in temporal_results:
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

    df_out = pd.DataFrame(final_predictions)
    if not df_out.empty:
        df_out = df_out[['x_c', 'y_c', 'stock_index']] 
        
        # Sorting (Row-by-Row, Left-to-Right) ---
        # 1. Create a temporary rounded Y column to group items into horizontal rows
        df_out['y_round'] = df_out['y_c'].round(2)
        # 2. Sort by the rounded Y (top to bottom), then by X (left to right)
        df_out = df_out.sort_values(by=['y_round', 'x_c'])
        # 3. Drop the temporary column before saving
        df_out = df_out.drop(columns=['y_round'])
        
        output_csv = results_csv_dir / f"{pallet_path.name}.csv"
        df_out.to_csv(output_csv, header=False, index=False)

    metrics = evaluate_final_predictions(final_predictions, gt_csv_path)
    return metrics, final_predictions

def plot_tolerance_sweep(all_predictions_dict, all_gt_paths, output_dir):
    """
    Sweeps through spatial tolerances and plots F1-Score degradation.
    """
    tolerances = np.linspace(0.001, 0.01, 100)
    f1_scores = []

    print("\nRunning Tolerance Sweep...")
    for tol in tolerances:
        tot_pred, tot_gt, tot_strict = 0, 0, 0
        
        for pallet_name, preds in all_predictions_dict.items():
            metrics = evaluate_final_predictions(preds, all_gt_paths[pallet_name], tolerance=tol)
            tot_pred += metrics['pred']
            tot_gt += metrics['gt']
            tot_strict += metrics['strict']
            
        p = tot_strict / tot_pred if tot_pred > 0 else 0
        r = tot_strict / tot_gt if tot_gt > 0 else 0
        f1 = 2 * (p * r) / (p + r) if (p + r) > 0 else 0
        f1_scores.append(f1)

    # Generate Graph
    plt.figure(figsize=(10, 6))
    plt.plot(tolerances, f1_scores, marker='o', linestyle='-', color='#4C72B0', linewidth=2)
    
    # Highlight the baseline 0.0085 tolerance
    baseline_tol = 0.0085
    idx_baseline = np.abs(tolerances - baseline_tol).argmin()
    plt.axvline(x=baseline_tol, color='#C44E52', linestyle='--', label=f'Selected Tolerance ({baseline_tol})')
    plt.scatter(baseline_tol, f1_scores[idx_baseline], color='#C44E52', s=100, zorder=5)
    
    plt.title("Pipeline Robustness: F1-Score vs. Spatial Tolerance")
    plt.xlabel("Spatial Tolerance (Canonical Coordinate Distance)")
    plt.ylabel("Dataset F1-Score")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    plt.tight_layout()
    plot_path = output_dir / "tolerance_sweep.png"
    plt.savefig(plot_path, dpi=300)
    print(f"Saved tolerance sweep graph to '{plot_path}'")
    plt.close()

def evaluate_dataset(dataset_dir, model_path, results_dir="../results"):
    """
    Iterates through all pallets, aggregates results, and plots graphs.
    """
    base_path = Path(dataset_dir)
    results_path = Path(results_dir)
    
    # Create the specific CSV output directory
    results_csv_dir = results_path / "results_csv"
    results_csv_dir.mkdir(parents=True, exist_ok=True)
    
    pallets = sorted([d for d in base_path.iterdir() if d.is_dir()])
    
    print(f"Starting evaluation on {len(pallets)} pallets in {dataset_dir}...\n")
    print(f"Saving predicted CSVs to {results_csv_dir}...\n")
    print(f"{'Pallet Name':<20} | {'GT':<5} | {'Pred':<5} | {'Spatial':<7} | {'Strict':<6} | {'F1':<6}")
    print("-" * 65)

    results_per_pallet = {}
    totals = {'pred': 0, 'gt': 0, 'spatial': 0, 'strict': 0}
    
    # Dictionaries to store raw predictions and GT paths for the tolerance sweep
    all_predictions_dict = {}
    all_gt_paths = {}

    for pallet in pallets:
        res = process_single_pallet(pallet, model_path, results_csv_dir)
        if res is None:
            continue
            
        metrics, final_preds = res
        
        # Save for later sweep
        all_predictions_dict[pallet.name] = final_preds
        all_gt_paths[pallet.name] = pallet / "locations.csv"

        totals['pred'] += metrics['pred']
        totals['gt'] += metrics['gt']
        totals['spatial'] += metrics['spatial']
        totals['strict'] += metrics['strict']
        
        # Calculate local F1 for the printout
        p = metrics['strict'] / metrics['pred'] if metrics['pred'] > 0 else 0
        r = metrics['strict'] / metrics['gt'] if metrics['gt'] > 0 else 0
        f1 = 2 * (p * r) / (p + r) if (p + r) > 0 else 0
        
        results_per_pallet[pallet.name] = f1
        print(f"{pallet.name:<20} | {metrics['gt']:<5} | {metrics['pred']:<5} | {metrics['spatial']:<7} | {metrics['strict']:<6} | {f1:.4f}")

    # Calculate Global Metrics
    precision = totals['strict'] / totals['pred'] if totals['pred'] > 0 else 0
    recall = totals['strict'] / totals['gt'] if totals['gt'] > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print("\n" + "="*40)
    print(" 🏆 AGGREGATED DATASET METRICS 🏆")
    print("="*40)
    print(f"Total Ground Truth: {totals['gt']}")
    print(f"Total Predicted:    {totals['pred']}")
    print(f"Total Spatial:   {totals['spatial']}")
    print(f"Total Strict (Location + Time):    {totals['strict']}")
    print("-" * 40)
    print(f"Global Precision:   {precision:.4f}")
    print(f"Global Recall:      {recall:.4f}")
    print(f"Global F1-Score:    {f1_score:.4f}")
    print("="*40)

    # --- Plotting Baseline Graphs ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot 1: Overall Metrics
    metrics_names = ['Precision', 'Recall', 'F1-Score']
    metrics_vals = [precision, recall, f1_score]
    bars = ax1.bar(metrics_names, metrics_vals, color=['#4C72B0', '#55A868', '#C44E52'])
    ax1.set_ylim(0, 1.1)
    ax1.set_title("Overall Pipeline Performance")
    ax1.set_ylabel("Score")
    for bar in bars:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f"{yval:.3f}", ha='center', va='bottom', fontweight='bold')

    # Plot 2: F1 Score per Pallet
    names = list(results_per_pallet.keys())
    scores = list(results_per_pallet.values())
    ax2.bar(names, scores, color='gray')
    ax2.set_ylim(0, 1.1)
    ax2.set_title("F1-Score per Pallet")
    ax2.set_ylabel("F1-Score")
    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels(names, rotation=45, ha='right', rotation_mode='anchor')
    
    plt.tight_layout()
    eval_plot_path = results_path / "evaluation_results.png"
    plt.savefig(eval_plot_path, dpi=300)
    print(f"\nSaved performance graphs to '{eval_plot_path}'")
    plt.close() # Close plot to prevent overlap
    
    # --- Run the Tolerance Sweep ---
    plot_tolerance_sweep(all_predictions_dict, all_gt_paths, results_path)

if __name__ == "__main__":
    TEST_DIR = "../data/all_data"
    MODEL = "../runs/detect/pallet_model_v1/weights/best.pt"
    RESULTS_DIR = "../results"
    
    evaluate_dataset(TEST_DIR, MODEL, RESULTS_DIR)