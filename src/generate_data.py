import os
import cv2
import json
import numpy as np
import pandas as pd
import random
import shutil
from pathlib import Path
from scipy.spatial.distance import pdist, squareform

def calculate_dynamic_params(csv_path):
    df_count = pd.read_csv(csv_path, header=None)
    count = len(df_count)
    
    t_sparse, t_dense = 60, 500
    
    if count <= t_sparse:
        overlap = 0.8
        # FIX: Keep 80% of patches for sparse/hard pallets
        sampling_rate = 0.8 
    elif count >= t_dense:
        overlap = 0.2
        # Smaller portion of dense patches to prevent them from dominating the training set.
        sampling_rate = 0.3 
    else:
        ratio = (count - t_sparse) / (t_dense - t_sparse)
        overlap = 0.8 - (ratio * (0.8 - 0.2))
        # Linear interpolation for sampling
        sampling_rate = 0.8 - (ratio * 0.7) 
    
    return overlap, sampling_rate

def get_exact_pallet_roi(image_path, metadata_path):
    img = cv2.imread(str(image_path))
    with open(metadata_path, 'r') as f:
        M_flat = json.load(f)
    M = np.array(M_flat).reshape(3, 3)
    
    # Affine transformation matrix mapping from pallet capture image coordinates to canonical pallet coordinates
    if abs(M[0, 0]) < 1000:
        M = np.linalg.inv(M)
        
    canonical_corners = np.array([[0,0,1], [1,0,1], [1,1,1], [0,1,1]]).T
    pixel_corners = M @ canonical_corners
    x_coords = pixel_corners[0, :] / pixel_corners[2, :]
    y_coords = pixel_corners[1, :] / pixel_corners[2, :]
    
    h, w = img.shape[:2]
    x_min, x_max = max(0, int(np.min(x_coords))), min(w, int(np.max(x_coords)))
    y_min, y_max = max(0, int(np.min(y_coords))), min(h, int(np.max(y_coords)))
    
    cropped_img = img[y_min:y_max, x_min:x_max]
    return cropped_img, (x_min, y_min, x_max, y_max), M

def process_pallet_for_yolo(pallet_dir, split_name, output_dir, patch_size=2048):
    img_path = pallet_dir / "pallet_capture.jpg"
    meta_path = pallet_dir / "pallet_metadata.json"
    csv_path = pallet_dir / "locations.csv"
    
    if not img_path.exists(): return
    
    overlap, sampling_rate = calculate_dynamic_params(csv_path)
    print(f"[{pallet_dir.name}] Overlap: {overlap:.2f} | Sampling Rate: {sampling_rate:.2%}")

    cropped_img, (crop_x_min, crop_y_min, _, _), M = get_exact_pallet_roi(img_path, meta_path)
    
    df = pd.read_csv(csv_path, header=None, names=['x_c', 'y_c', 'stock_index'])
    ones = np.ones(len(df))
    canonical_coords = np.column_stack((df['x_c'], df['y_c'], ones))
    
    pixel_coords = (M @ canonical_coords.T).T
    df['x_pix'] = (pixel_coords[:, 0] / pixel_coords[:, 2]) - crop_x_min
    df['y_pix'] = (pixel_coords[:, 1] / pixel_coords[:, 2]) - crop_y_min
    
    # Dynamic BBox Size capped between physical limits
    # df['bbox_size'] = 60.0
    # for stock_idx, group in df.groupby('stock_index'):
    #     if len(group) > 1:
    #         pts = group[['x_pix', 'y_pix']].values
    #         dist_matrix = squareform(pdist(pts))
    #         np.fill_diagonal(dist_matrix, np.inf)
    #         median_pitch = np.median(dist_matrix.min(axis=1))
    #         safe_size = np.clip(median_pitch * 0.9, a_min=80.0, a_max=1600.0)
    #         df.loc[group.index, 'bbox_size'] = safe_size

    df['bbox_size'] = 60.0 # Safe default
    for stock_idx, group in df.groupby('stock_index'):
        if len(group) > 1:
            pts = group[['x_pix', 'y_pix']].values
            dist_matrix = squareform(pdist(pts))
            np.fill_diagonal(dist_matrix, np.inf)
            median_pitch = np.median(dist_matrix.min(axis=1))
            
            # Multiply by 0.5 so the box only reaches halfway to the neighbor (NO OVERLAPS)
            # Cap the max size at 300px so larger components (i.e CPUs) don't swallow nearby small components
            safe_size = np.clip(median_pitch * 0.5, a_min=40.0, a_max=300.0)
            df.loc[group.index, 'bbox_size'] = safe_size
            
    h_img, w_img = cropped_img.shape[:2]
    stride = int(patch_size * (1 - overlap))
    patch_id = 0
    
    for y_start in range(0, h_img, stride):
        for x_start in range(0, w_img, stride):
            # Stochastic Downsampling to keep dataset balanced
            if random.random() > sampling_rate:
                continue
                
            x_end = min(x_start + patch_size, w_img)
            y_end = min(y_start + patch_size, h_img)
            
            mask = (df['x_pix'] >= x_start) & (df['x_pix'] < x_end) & \
                   (df['y_pix'] >= y_start) & (df['y_pix'] < y_end)
            patch_labels = df[mask].copy()
            
            if len(patch_labels) == 0: continue 
            
            patch = cropped_img[y_start:y_end, x_start:x_end]
            actual_patch_w, actual_patch_h = x_end - x_start, y_end - y_start
            
            if actual_patch_w < 100 or actual_patch_h < 100: continue 
            
            patch_labels['yolo_x'] = (patch_labels['x_pix'] - x_start) / actual_patch_w
            patch_labels['yolo_y'] = (patch_labels['y_pix'] - y_start) / actual_patch_h
            patch_labels['yolo_w'] = (patch_labels['bbox_size'] / actual_patch_w).clip(upper=1.0)
            patch_labels['yolo_h'] = (patch_labels['bbox_size'] / actual_patch_h).clip(upper=1.0)
            
            base_name = f"{pallet_dir.name}_patch_{patch_id}"
            cv2.imwrite(str(output_dir / "images" / split_name / f"{base_name}.jpg"), patch)
            
            with open(output_dir / "labels" / split_name / f"{base_name}.txt", 'w') as f:
                for _, row in patch_labels.iterrows():
                    f.write(f"0 {row['yolo_x']:.6f} {row['yolo_y']:.6f} {row['yolo_w']:.6f} {row['yolo_h']:.6f}\n")
            
            patch_id += 1
            
    print(f"[{split_name.upper()}] Processed {pallet_dir.name}: {patch_id} patches.")

def create_dataset_hybrid(train_dir="../data/training", 
                            val_dir="../data/validation", 
                            test_dir="../data/test", 
                            output_dir="../data/yolo_dataset",
                            leak_ratio=0.20): # 20% of training patches go to val/test - data leakage so all component types are represented
    
    out_path = Path(output_dir)
    
    if out_path.exists():
        shutil.rmtree(out_path)
        
    for split in ['train', 'val', 'test']:
        (out_path / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_path / "labels" / split).mkdir(parents=True, exist_ok=True)
        
    print("--- Processing Base Directories ---")
    def process_directory(source_dir, split_name):
        src_path = Path(source_dir)
        if not src_path.exists(): return
        pallets = sorted([d for d in src_path.iterdir() if d.is_dir()])
        for p in pallets:
            process_pallet_for_yolo(p, split_name, out_path)

    process_directory(train_dir, 'train')
    process_directory(val_dir, 'val')
    process_directory(test_dir, 'test')
    
    # Implement the Hybrid Split (Leaking training patches)
    print(f"\n--- Seeding Val/Test with {leak_ratio*100}% of Training Data ---")
    train_images_dir = out_path / "images" / "train"
    train_labels_dir = out_path / "labels" / "train"
    
    all_train_images = list(train_images_dir.glob("*.jpg"))
    random.shuffle(all_train_images)
    
    num_to_leak = int(len(all_train_images) * leak_ratio)
    val_leak_count = num_to_leak // 2
    test_leak_count = num_to_leak - val_leak_count
    
    for i, img_path in enumerate(all_train_images[:num_to_leak]):
        label_path = train_labels_dir / f"{img_path.stem}.txt"
        
        target_split = 'val' if i < val_leak_count else 'test'
        
        # Move the files instead of copying to prevent duplicates in train
        shutil.move(str(img_path), str(out_path / "images" / target_split / img_path.name))
        if label_path.exists():
            shutil.move(str(label_path), str(out_path / "labels" / target_split / label_path.name))

    print("\nHybrid Dataset generation complete!")

# create_dataset_hybrid()