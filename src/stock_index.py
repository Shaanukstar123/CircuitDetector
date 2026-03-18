import cv2
import numpy as np
from detector import get_global_predictions
from ultralytics import YOLO
from generate_data import get_exact_pallet_roi
from pathlib import Path

def crop_stock_to_pallet(img, scale=0.25):
    """
    Detects the inner edge of the checkerboard frame on all sides and returns a cropped image.
    """
    # scale down for faster processing
    small = cv2.resize(img, (0, 0), fx=scale, fy=scale)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    
    # Use Otsu's Threshold to handle lighting
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    h_s, w_s = gray.shape
    left_squares, right_squares, top_squares, bottom_squares = [], [], [], []
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 50 < area < 5000:
            x, y, w, h = cv2.boundingRect(cnt)
            # Extent = Area / (w*h). Squares are near 1.0, Circles are ~0.78
            extent = area / float(w * h)
            aspect_ratio = w / float(h)
            # Ignore circular blobs and focus on square-like contours - checkerboard holes
            if 0.7 < aspect_ratio < 1.3 and extent > 0.8:
                cx, cy = x + w/2, y + h/2
                # Wider buckets (35%) to ensure we catch squares even if shifted
                if cx < w_s * 0.35: left_squares.append(x + w) 
                if cx > w_s * 0.65: right_squares.append(x)    
                if cy < h_s * 0.35: top_squares.append(y + h)  
                if cy > h_s * 0.65: bottom_squares.append(y)   
    
    # print(f"Checkerboard detection - Left: {len(left_squares)}, Right: {len(right_squares)}")
    
    # Use median to ignore random noise/labels inside the pallet
    inner_left = int(np.median(left_squares)) if left_squares else 0
    inner_right = int(np.median(right_squares)) if right_squares else w_s
    inner_top = int(np.median(top_squares)) if top_squares else 0
    inner_bottom = int(np.median(bottom_squares)) if bottom_squares else h_s
    
    # Safe Margin
    margin = int(20 * scale)
    inner_left = min(w_s//3, inner_left + margin)
    inner_right = max(2*w_s//3, inner_right - margin)
    
    # Scale boundaries back to full-res
    full_x, full_y = int(inner_left / scale), int(inner_top / scale)
    full_r, full_b = int(inner_right / scale), int(inner_bottom / scale)
    
    return img[full_y:full_b, full_x:full_r], full_x, full_y

def get_master_homography(last_stock_img, cropped_capture_img):
    """
    Maps pallet image pixels to stock image pixels with precise resolution scaling.
    """
    # Crop the stock image
    cropped_stock, off_x, off_y = crop_stock_to_pallet(last_stock_img)
    
    # Rescale both to a common width (2000px) for AKAZE
    tw = 2000
    s_stock = tw / cropped_stock.shape[1]
    s_cap = tw / cropped_capture_img.shape[1]
    
    small_stock = cv2.resize(cropped_stock, (0, 0), fx=s_stock, fy=s_stock)
    small_cap = cv2.resize(cropped_capture_img, (0, 0), fx=s_cap, fy=s_cap)
    
    # 3. AKAZE Feature Matching
    akaze = cv2.AKAZE_create()
    kp1, des1 = akaze.detectAndCompute(small_stock, None)
    kp2, des2 = akaze.detectAndCompute(small_cap, None)
    
    matches = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(des2, des1, k=2) # Pallet -> Stock
    good = [m for m, n in matches if m.distance < 0.7 * n.distance]
    
    src_pts = np.float32([kp2[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    
    # 4. The Matrix Chain
    # M_small maps: (Small Capture) -> (Small Stock Crop)
    M_small, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)
    
    # Scale Matrices
    S_cap = np.diag([s_cap, s_cap, 1])
    S_stock_inv = np.diag([1/s_stock, 1/s_stock, 1])
    
    # Translation Matrix (Correcting the Crop)
    T_off = np.array([[1, 0, off_x], [0, 1, off_y], [0, 0, 1]], dtype=np.float64)
    
    # Master Sequence: FullPallet -> SmallPallet -> SmallStock -> FullStockCrop -> FullStockOriginal
    M_master = T_off @ S_stock_inv @ M_small @ S_cap
    
    return M_master


def align_image_to_capture_visualiser(stock_img, capture_img, scale=0.25):
    """
    Visualizes the AKAZE matches and the final full-resolution blend.
    """
    print("Running AKAZE + Dynamic Crop Debugger...")
    cropped_stock, offset_x, offset_y = crop_stock_to_pallet(stock_img, scale)
    
    small_stock = cv2.resize(cropped_stock, (0, 0), fx=scale, fy=scale)
    small_cap = cv2.resize(capture_img, (0, 0), fx=scale, fy=scale)
    
    akaze = cv2.AKAZE_create()
    kp1, des1 = akaze.detectAndCompute(small_stock, None)
    kp2, des2 = akaze.detectAndCompute(small_cap, None)
    
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)
    
    # --- THE FIX: Cleanly extract good matches ---
    good_matches = []
    for match_pair in matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)
            
    # 1. Visualize the Matches on the clean, cropped images
    match_img = cv2.drawMatches(
        small_stock, kp1, small_cap, kp2, good_matches, None, 
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    
    h_m, w_m = match_img.shape[:2]
    s_m = 1600 / w_m if w_m > 1600 else 1.0
    cv2.imshow("AKAZE Matches (Press Key)", cv2.resize(match_img, (0, 0), fx=s_m, fy=s_m))
    cv2.waitKey(0)
    
    # 2. Generate the Full Translation Matrix
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    M_small, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    
    S = np.array([[scale, 0, 0], [0, scale, 0], [0, 0, 1]])
    S_inv = np.array([[1/scale, 0, 0], [0, 1/scale, 0], [0, 0, 1]])
    M_full_crop = S_inv @ M_small @ S
    
    T = np.array([[1, 0, offset_x], [0, 1, offset_y], [0, 0, 1]], dtype=np.float64)
    M_master = T @ M_full_crop
    
    # 3. The Final Blend 
    M_inv_vis = np.linalg.inv(M_master) 
    
    h_c, w_c = capture_img.shape[:2]
    aligned_img = cv2.warpPerspective(stock_img, M_inv_vis, (w_c, h_c))
    blended = cv2.addWeighted(capture_img, 0.5, aligned_img, 0.5, 0)
    
    h_b, w_b = blended.shape[:2]
    s_b = 1000 / max(h_b, w_b)
    cv2.imshow("Master Matrix Blend (Press any key)", cv2.resize(blended, (0, 0), fx=s_b, fy=s_b))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return aligned_img

def assign_stock_indices(yolo_centers, pallet_dir, visual=False):
    """
    Computes stock image index assignment. 
    If debug=True, visualizes the HSV masking process frame-by-frame.
    """
    cropped_cap, _, _ = get_exact_pallet_roi(pallet_dir / "pallet_capture.jpg", pallet_dir / "pallet_metadata.json")
    pallet_path = Path(pallet_dir)
    stock_files = sorted(pallet_path.glob("stock_*.jpg"), key=lambda x: int(x.stem.split('_')[1]))

    # HSV Thresholds for empty pocket detection
    lower_yellow = np.array([20, 60, 180])
    upper_yellow = np.array([30, 255, 255])

    histories_yellow = {i: [] for i in range(len(yolo_centers))}
    yellow_max_ratio = 0.15 # If more than 15% yellow, it's considered empty (no component)

    for sf in stock_files:
        img = cv2.imread(str(sf))
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        if visual:
            vis_img = img.copy()
            print(f"Debugging {sf.name}...")
        
        # Calculate Homography matrix
        M = get_master_homography(img, cropped_cap)
        
        if M is None:
            print(f"Warning: Homography failed on {sf.name}. Skipping visualization.")
            for i in range(len(yolo_centers)): histories_yellow[i].append(0.0)
            continue

        for i, center in enumerate(yolo_centers):
            # Project the coordinate
            pt = M @ np.array([center['x'], center['y'], 1.0])
            px, py = int(pt[0]/pt[2]), int(pt[1]/pt[2])
            
            # Extract Patch & Calculate Ratio
            patch_radius = 10
            yellow_ratio = 0.0
            
            if patch_radius <= px < hsv_img.shape[1] - patch_radius and patch_radius <= py < hsv_img.shape[0] - patch_radius:
                patch = hsv_img[py-patch_radius : py+patch_radius+1, px-patch_radius : px+patch_radius+1]
                mask = cv2.inRange(patch, lower_yellow, upper_yellow)
                yellow_ratio = np.sum(mask > 0) / mask.size
                
            histories_yellow[i].append(yellow_ratio)

            # Visualisation
            if visual:
                cv2.line(vis_img, (px-10, py), (px+10, py), (0, 0, 255), 2)
                cv2.line(vis_img, (px, py-10), (px, py+10), (0, 0, 255), 2)
                
                if yellow_ratio < yellow_max_ratio:
                    # Component is Present (Yellow covered) -> Draw Green Circle
                    cv2.circle(vis_img, (px, py), 15, (0, 255, 0), 2)
                else:
                    # Yellow base is Visible (Empty) -> Draw Thick Yellow Circle
                    cv2.circle(vis_img, (px, py), 15, (0, 255, 255), 2)

        if visual:
            cv2.imshow(f"{sf.name} - Yellow=Empty, Green=Part Present", cv2.resize(vis_img, (1000, 800)))
            print(f"Image {sf.name}: Press any key to see next...")
            cv2.waitKey(0)

    if visual:
        cv2.destroyAllWindows()

    # Find the exact moment the Yellow disappears
    final_results = []
    for i, center in enumerate(yolo_centers):
        y_hist = np.array(histories_yellow[i])
        stock_idx = len(stock_files) - 1 
        
        for idx, y_ratio in enumerate(y_hist):
            if y_ratio < yellow_max_ratio: 
                stock_idx = idx
                break
                
        final_results.append({
            'x': center['x'],
            'y': center['y'],
            'stock_index': stock_idx
        })

    return final_results