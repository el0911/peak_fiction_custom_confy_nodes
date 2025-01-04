import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import rembg

class ExtractMaskFromScribbleMap:
    def detect_shapes_bbox(image, padding=20):
        import cv2
        import numpy as np
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian Blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Thresholding
        _, thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY_INV)

        # Dilate the image
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.dilate(thresh, kernel, iterations=1)

        # Ensure dtype is uint8
        thresh = np.uint8(thresh)

        # Debugging: Check thresh properties
        print(f"thresh dtype: {thresh.dtype}, shape: {thresh.shape}")

        # Find contours
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

        if len(centers) < 2:
            return bounding_boxes

        # KMeans clustering
        x = np.array(centers)
        best_score = -1
        best_k = 2
        for i in range(2, min(10, len(centers) + 1)):
            kmeans = KMeans(n_clusters=i, random_state=10)
            kmeans.fit(x)
            silhouette_avg = silhouette_score(x, kmeans.labels_)
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


    @staticmethod
    def removeBackground(input_image):
        rgb_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        input_array = np.array(rgb_image)
        output_array = rembg.remove(input_array)
        output_bgra = cv2.cvtColor(output_array, cv2.COLOR_RGBA2BGRA)
        return output_bgra

    import numpy as np
    import torch

    def get_map(original_image, scribble_image):
        """
        Extracts masks using the main function logic and returns a tensor
        
        Args:
            original_image: The source/reference image
            scribble_image: The scribble map image
        
        Returns:
            Binary mask as a torch tensor of shape (1, H, W)
        """
        height, width = original_image.shape[:2]
        
        try:
            # Detect bounding boxes from scribble image
            boxes = ExtractMaskFromScribbleMap.detect_shapes_bbox(scribble_image)
            
            if not boxes:
                print("No bounding boxes found.")
                return torch.zeros((1, height, width), dtype=torch.float32)
            
            print(f"Found {len(boxes)} bounding boxes.")
            
            # Initialize mask
            final_mask = np.zeros((height, width), dtype=np.uint8)
            
            # Process each bounding box
            for i, bbox in enumerate(boxes):
                x, y, w, h = bbox
                print(f"Processing bounding box {i+1}/{len(boxes)}: {bbox}")
                
                # Ensure the bounding box is within image bounds
                if x + w <= width and y + h <= height:
                    roi = original_image[y:y+h, x:x+w]
                else:
                    print(f"Warning: Bounding box {bbox} is out of image bounds.")
                    continue  # Skip this iteration
                
                # Extract region and remove background
                extracted_with_alpha = ExtractMaskFromScribbleMap.removeBackground(roi)
                print(f"extracted_with_alpha shape: {extracted_with_alpha.shape}")
                
                # Check if extracted_with_alpha has 4 channels (RGBA)
                if extracted_with_alpha.shape[2] >= 4:
                    alpha_channel = extracted_with_alpha[:, :, 3]
                    print(f"Alpha channel extracted, max value: {np.max(alpha_channel)}")
                    final_mask[y:y+h, x:x+w] = np.maximum(
                        final_mask[y:y+h, x:x+w],
                        alpha_channel
                    )
                else:
                    print("Warning: No alpha channel found in extracted_with_alpha.")
            
            # Convert numpy array to torch tensor
            # Add batch dimension and ensure float32 dtype
            tensor_mask = torch.from_numpy(final_mask).float().unsqueeze(0)
            
            print("Final mask converted to tensor.")
            return tensor_mask

        except Exception as e:
            print(f"Error processing image: {str(e)}")
            return torch.zeros((1, height, width), dtype=torch.float32)