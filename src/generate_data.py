import os
import cv2
import json
import numpy as np
import pandas as pd
import random
import shutil
from pathlib import Path
from scipy.spatial.distance import pdist, squareform

def get_exact_pallet_roi(image_path, metadata_path):
    img = cv2.imread(str(image_path))
    with open(metadata_path, 'r') as f:
        M_flat = json.load(f)
    M = np.array(M_flat).reshape(3, 3)
    if abs(M[0, 0]) < 1000:
        M = np.linalg.inv(M)
        
    canonical_corners = np.array([[0,0,1], [1,0,1], [1,1,1], [0,1,1]]).T
    pixel_corners = M @ canonical_corners
    x_coords = pixel_corners[0, :] / pixel_corners[2, :]
    y_coords = pixel_corners[1, :] / pixel_corners[2, :]
    
    pad = 50
    h, w = img.shape[:2]
    x_min, x_max = max(0, int(np.min(x_coords)) - pad), min(w, int(np.max(x_coords)) + pad)
    y_min, y_max = max(0, int(np.min(y_coords)) - pad), min(h, int(np.max(y_coords)) + pad)
    
    cropped_img = img[y_min:y_max, x_min:x_max]
    return cropped_img, (x_min, y_min, x_max, y_max), M

# YOLO Patch Generator
def process_pallet_for_yolo(pallet_dir, split_name, output_dir, patch_size=2048, overlap=0.2):
    img_path = pallet_dir / "pallet_capture.jpg"
    meta_path = pallet_dir / "pallet_metadata.json"
    csv_path = pallet_dir / "locations.csv"
    
    if not img_path.exists(): return
    
    # Get the cropped image and the projection matrix
    cropped_img, (crop_x_min, crop_y_min, _, _), M = get_exact_pallet_roi(img_path, meta_path)
    
    # Ground Truth
    df = pd.read_csv(csv_path, header=None, names=['x_c', 'y_c', 'stock_index'])
    ones = np.ones(len(df))
    canonical_coords = np.column_stack((df['x_c'], df['y_c'], ones))
    
    # Project to Pixel Space
    pixel_coords = (M @ canonical_coords.T).T
    df['x_pix'] = (pixel_coords[:, 0] / pixel_coords[:, 2]) - crop_x_min
    df['y_pix'] = (pixel_coords[:, 1] / pixel_coords[:, 2]) - crop_y_min
    
    # Calculate Dynamic BBox Size (Pitch Heuristic)
    df['bbox_size'] = 40.0 # Default
    for stock_idx, group in df.groupby('stock_index'):
        if len(group) > 1:
            pts = group[['x_pix', 'y_pix']].values
            dist_matrix = squareform(pdist(pts))
            np.fill_diagonal(dist_matrix, np.inf)
            median_pitch = np.median(dist_matrix.min(axis=1))
            df.loc[group.index, 'bbox_size'] = median_pitch * 1.2 # 20% margin
            
    # Sliding Window Extraction
    h_img, w_img = cropped_img.shape[:2]
    stride = int(patch_size * (1 - overlap))
    patch_id = 0
    
    for y_start in range(0, h_img, stride):
        for x_start in range(0, w_img, stride):
            x_end = min(x_start + patch_size, w_img)
            y_end = min(y_start + patch_size, h_img)
            
            # Find pockets inside this patch (with safe boundary checks)
            mask = (df['x_pix'] >= x_start) & (df['x_pix'] < x_end) & \
                   (df['y_pix'] >= y_start) & (df['y_pix'] < y_end)
            patch_labels = df[mask].copy()
            
            if len(patch_labels) == 0: continue # Skip empty background patches
            
            patch = cropped_img[y_start:y_end, x_start:x_end]
            
            # Convert to YOLO Normalized Coordinates (0.0 to 1.0)
            actual_patch_w = x_end - x_start
            actual_patch_h = y_end - y_start
            
            # Discard useless microscopic edge slivers ---
            # If the remaining patch is smaller than a typical component, skip.
            if actual_patch_w < 100 or actual_patch_h < 100:
                continue 
            
            patch_labels['yolo_x'] = (patch_labels['x_pix'] - x_start) / actual_patch_w
            patch_labels['yolo_y'] = (patch_labels['y_pix'] - y_start) / actual_patch_h
            
            # Clip the width and height ---
            # Even if a box tries to overhang, .clip() forces it to stop perfectly at the image edge (1.0).
            patch_labels['yolo_w'] = (patch_labels['bbox_size'] / actual_patch_w).clip(upper=1.0)
            patch_labels['yolo_h'] = (patch_labels['bbox_size'] / actual_patch_h).clip(upper=1.0)
            # Save files
            base_name = f"{pallet_dir.name}_patch_{patch_id}"
            cv2.imwrite(str(output_dir / "images" / split_name / f"{base_name}.jpg"), patch)
            
            with open(output_dir / "labels" / split_name / f"{base_name}.txt", 'w') as f:
                for _, row in patch_labels.iterrows():
                    # Class 0: 'pocket'
                    f.write(f"0 {row['yolo_x']:.6f} {row['yolo_y']:.6f} {row['yolo_w']:.6f} {row['yolo_h']:.6f}\n")
            
            patch_id += 1
            
    print(f"[{split_name.upper()}] Processed {pallet_dir.name}: {patch_id} patches generated.")

def create_dataset_explicit(train_dir="../data/training", 
                            val_dir="../data/validation", 
                            test_dir="../data/test", 
                            output_dir="../data/yolo_dataset"):
    
    out_path = Path(output_dir)
    
    for split in ['train', 'val', 'test']:
        (out_path / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_path / "labels" / split).mkdir(parents=True, exist_ok=True)
        
    def process_directory(source_dir, split_name):
        src_path = Path(source_dir)
        if not src_path.exists():
            print(f"Warning: Directory {source_dir} not found. Skipping {split_name} split.")
            return
            
        # Grab all pallet folders inside directory
        pallets = sorted([d for d in src_path.iterdir() if d.is_dir()])
        print(f"Found {len(pallets)} pallets in {source_dir} for {split_name.upper()}")
        
        for p in pallets:
            process_pallet_for_yolo(p, split_name, out_path)

    print("--- Starting Training Data Generation ---")
    process_directory(train_dir, 'train')
    
    print("\n--- Starting Validation Data Generation ---")
    process_directory(val_dir, 'val')
    
    print("\n--- Starting Test Data Generation ---")
    process_directory(test_dir, 'test')
    
    print("\nExplicit Dataset generation complete!")

create_dataset_explicit(
    train_dir="../data/training",
    val_dir="../data/validation",
    test_dir="../data/test",
    output_dir="../data/yolo_dataset"
)
