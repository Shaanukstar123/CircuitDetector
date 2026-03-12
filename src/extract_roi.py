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
    
    img_with_box = img.copy()
    cv2.rectangle(img_with_box, (x_min, y_min), (x_max, y_max), (0, 0, 255), 10)
    
    cropped_img = img[y_min:y_max, x_min:x_max]
    
    return cropped_img, img_with_box, (x_min, y_min, x_max, y_max)

# Test:   
img_path = "../data/pallet_24774/pallet_capture.jpg"
meta_path = "../data/pallet_24774/pallet_metadata.json"

cropped, boxed, bbox = get_exact_pallet_roi(img_path, meta_path)

scale = 0.3
disp_w = int(boxed.shape[1] * scale)
disp_h = int(boxed.shape[0] * scale)
preview = cv2.resize(boxed, (disp_w, disp_h))
cv2.namedWindow("ROI", cv2.WINDOW_NORMAL)
cv2.imshow("ROI", preview)
cv2.waitKey(0)
cv2.destroyAllWindows()
