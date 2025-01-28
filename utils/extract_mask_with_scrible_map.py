import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import rembg
import torch
class ExtractMaskFromScribbleMap:
    @staticmethod
    def detect_shapes_bbox(image, padding=10):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY_INV)
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.dilate(thresh, kernel, iterations=1)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Get bounding boxes and center coordinates
        bounding_boxes = []
        centers = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 50:  # Filter small contours
                x, y, w, h = cv2.boundingRect(contour)
                bounding_boxes.append((x, y, w, h))
                center = (x + w / 2, y + h / 2)
                centers.append(center)

        if len(centers) < 2:
            return bounding_boxes

        x = np.array(centers)
        
        # Modified KMeans clustering with error handling
        try:
            # Limit number of clusters to number of samples - 1
            max_clusters = min(8, len(centers) - 1)
            best_score = -1
            best_k = 2
            
            for i in range(2, max_clusters + 1):
                kmeans = KMeans(n_clusters=i, random_state=10)
                labels = kmeans.fit_predict(x)
                
                # Check if we have at least 2 samples per cluster
                unique_labels, counts = np.unique(labels, return_counts=True)
                if np.all(counts >= 2):  # Only calculate score if all clusters have at least 2 samples
                    silhouette_avg = silhouette_score(x, labels)
                    if silhouette_avg > best_score:
                        best_score = silhouette_avg
                        best_k = i

            kmeans = KMeans(n_clusters=best_k, random_state=10)
            kmeans.fit(x)

            clustered_bboxes = {i: [] for i in range(best_k)}
            for idx, label in enumerate(kmeans.labels_):
                clustered_bboxes[label].append(bounding_boxes[idx])

            cluster_bboxes = []
            for cluster_idx, bboxes in clustered_bboxes.items():
                min_x = min(b[0] for b in bboxes)
                min_y = min(b[1] for b in bboxes)
                max_x = max(b[0] + b[2] for b in bboxes)
                max_y = max(b[1] + b[3] for b in bboxes)

                min_x = max(0, min_x - padding)
                min_y = max(0, min_y - padding)
                max_x += padding
                max_y += padding

                cluster_bboxes.append((min_x, min_y, max_x - min_x, max_y - min_y))

            return cluster_bboxes
            
        except Exception as e:
            print(f"Clustering failed, returning original bounding boxes: {str(e)}")
            return bounding_boxes

    @staticmethod
    def get_map(original_image, scribble_image):
        """
        Extracts masks using the main function logic
        Args:
            original_image: The source/reference image
            scribble_image: The scribble map image
        Returns:
            Binary mask
        """
        try:
            # Get image dimensions first
            height, width = original_image.shape[:2]
            
            # Detect bounding boxes from scribble image
            boxes = ExtractMaskFromScribbleMap.detect_shapes_bbox(scribble_image)
            
            # Initialize mask
            final_mask = np.zeros((height, width), dtype=np.uint8)
            
            # Process each bounding box
            for i, bbox in enumerate(boxes):
                x, y, w, h = bbox
                # Ensure coordinates are within image bounds
                x = max(0, min(x, width - 1))
                y = max(0, min(y, height - 1))
                w = min(w, width - x)
                h = min(h, height - y)
                
                if w <= 0 or h <= 0:
                    continue
                    
                roi = original_image[y:y+h, x:x+w]
                
                # Extract region and remove background
                extracted_with_alpha = ExtractMaskFromScribbleMap.removeBackground(roi)
                
                # Ensure the ROI and mask have the same dimensions
                if extracted_with_alpha.shape[:2] != (h, w):
                    extracted_with_alpha = cv2.resize(extracted_with_alpha, (w, h))
                
                # Add the alpha channel to the mask
                final_mask[y:y+h, x:x+w] = np.maximum(
                    final_mask[y:y+h, x:x+w],
                    extracted_with_alpha[:, :, 3]
                )
            
            # Convert to tensor
            tensor_mask = torch.tensor(final_mask, dtype=torch.float32).unsqueeze(0) / 255.0
            
            return tensor_mask

        except Exception as e:
            print(f"Error in get_map: {str(e)}")
            # Now height and width are defined before they might be used in the error case
            return torch.zeros((1, height, width), dtype=torch.float32)
import numpy as np
import torch
from PIL import Image

def save_tensor_mask(tensor_mask, output_path):
    """
    Save the tensor mask as an image.

    Parameters:
    - tensor_mask: A PyTorch tensor representing the mask.
    - output_path: Path to save the output image.
    """
    # Ensure the tensor is in the range [0, 255] for saving as an image
    mask = tensor_mask.squeeze(0).cpu().numpy() * 255  # Squeeze the batch dimension, convert to numpy and scale to [0, 255]
    
    # Clip to make sure all values are in the valid range [0, 255]
    mask = np.clip(mask, 0, 255).astype(np.uint8)
    
    # Convert the mask to an image and save it
    image = Image.fromarray(mask)
    image.save(output_path)
    print(f"Mask saved to {output_path}")
