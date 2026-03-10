import cv2
import numpy as np
import json

def get_exact_pallet_roi(image_path, metadata_path):
    """
    Uses the Affine Matrix to project the [0,1] canonical bounds
    onto the image pixel space, extracting the exact target pallet.
    """
    img = cv2.imread(str(image_path))
    
    with open(metadata_path, 'r') as f:
        M_flat = json.load(f)
    M = np.array(M_flat).reshape(3, 3)
    
    # Ensure it maps Canonical -> Pixel
    if abs(M[0, 0]) < 1000:
        M = np.linalg.inv(M)
        
    # 2. Define the 4 corners of the pallet in Canonical [0, 1] space
    canonical_corners = np.array([
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0]
    ]).T  # Transpose for matrix multiplication
    
    # 3. Project to High-Res Pixel Space
    pixel_corners = M @ canonical_corners
    
    # Normalize homogeneous coordinates
    x_coords = pixel_corners[0, :] / pixel_corners[2, :]
    y_coords = pixel_corners[1, :] / pixel_corners[2, :]
    
    # 4. Get the Bounding Box of these projected corners
    x_min, x_max = int(np.min(x_coords)), int(np.max(x_coords))
    y_min, y_max = int(np.min(y_coords)), int(np.max(y_coords))
    
    pad = 50
    h, w = img.shape[:2]
    x_min = max(0, x_min - pad)
    y_min = max(0, y_min - pad)
    x_max = min(w, x_max + pad)
    y_max = min(h, y_max + pad)
    
    img_with_box = img.copy()
    cv2.rectangle(img_with_box, (x_min, y_min), (x_max, y_max), (0, 0, 255), 10)
    
    cropped_img = img[y_min:y_max, x_min:x_max]
    
    return cropped_img, img_with_box, (x_min, y_min, x_max, y_max)

# Test:   
img_path = "data/pallet_24774/pallet_capture.jpg"
meta_path = "data/pallet_24774/pallet_metadata.json"

cropped, boxed, bbox = get_exact_pallet_roi(img_path, meta_path)

scale = 0.3
disp_w = int(boxed.shape[1] * scale)
disp_h = int(boxed.shape[0] * scale)
preview = cv2.resize(boxed, (disp_w, disp_h))
cv2.namedWindow("ROI", cv2.WINDOW_NORMAL)
cv2.imshow("ROI", preview)
cv2.waitKey(0)
cv2.destroyAllWindows()
