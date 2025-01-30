import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import rembg

class ExtractMaskFromScribbleMap:
    @staticmethod
    def detect_shapes_bbox(image, padding=10):
        # Implementation from your original code
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
            if area > 50:
                x, y, w, h = cv2.boundingRect(contour)
                bounding_boxes.append((x, y, w, h))
                center = (x + w / 2, y + h / 2)
                centers.append(center)

        # Handle cases with few centers
        if len(centers) < 2:
            return bounding_boxes
            
        x = np.array(centers)
        
        # Modified clustering logic to handle edge cases
        if len(centers) < 4:  # If we have 2 or 3 centers
            kmeans = KMeans(n_clusters=2, random_state=10)
            kmeans.fit(x)
            labels = kmeans.labels_
        else:
            # KMeans clustering with silhouette score
            best_score = -1
            best_k = 2
            best_labels = None
            
            # Only try clustering up to the number of points minus 1
            max_clusters = min(10, len(centers))
            
            for i in range(2, max_clusters):
                kmeans = KMeans(n_clusters=i, random_state=10)
                labels = kmeans.fit_predict(x)
                
                # Only calculate silhouette score if we have enough points
                if len(np.unique(labels)) > 1:
                    try:
                        silhouette_avg = silhouette_score(x, labels)
                        if silhouette_avg > best_score:
                            best_score = silhouette_avg
                            best_k = i
                            best_labels = labels
                    except:
                        continue
            
            if best_labels is None:
                best_k = 2
                kmeans = KMeans(n_clusters=best_k, random_state=10)
                labels = kmeans.fit_predict(x)
            else:
                labels = best_labels

        # Group bounding boxes by cluster
        clustered_bboxes = {i: [] for i in range(len(np.unique(labels)))}
        for idx, label in enumerate(labels):
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

    @staticmethod
    def removeBackground(input_image):
        rgb_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        input_array = np.array(rgb_image)
        output_array = rembg.remove(input_array)
        output_bgra = cv2.cvtColor(output_array, cv2.COLOR_RGBA2BGRA)
        return output_bgra

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
            # Get image dimensions
            height, width = original_image.shape[:2]
            
            # Detect bounding boxes from scribble image
            boxes = ExtractMaskFromScribbleMap.detect_shapes_bbox(scribble_image)
            
            # Initialize mask
            final_mask = np.zeros((height, width), dtype=np.uint8)
            
            # Process each bounding box
            for i, bbox in enumerate(boxes):
                x, y, w, h = bbox
                roi = original_image[y:y+h, x:x+w]
                
                # Extract region and remove background
                extracted_with_alpha = ExtractMaskFromScribbleMap.removeBackground(roi)
                
                # Add the alpha channel to the mask
                final_mask[y:y+h, x:x+w] = np.maximum(
                    final_mask[y:y+h, x:x+w],
                    extracted_with_alpha[:, :, 3]
                )
            
            return torch.from_numpy(final_mask).float().unsqueeze(0) / 255.0

        except Exception as e:
            print(f"Error in get_map: {str(e)}")
            return np.zeros((height, width), dtype=np.uint8)