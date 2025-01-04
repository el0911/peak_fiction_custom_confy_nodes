from utils.extract_mask_with_scrible_map import ExtractMaskFromScribbleMap
import numpy as np
import cv2

class Extract_mask_with_scrible_map:
    """
    A node that extracts a mask from a scribble map using the original image as reference.
    Returns a MASK type that can be used with other mask-compatible nodes.
    """
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "original_image": ("IMAGE",),
                "scribble_image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "get_map"
    CATEGORY = "peakfiction/custom"

    def get_map(self, original_image, scribble_image):
        # Convert tensor images to numpy arrays if needed
        if not isinstance(original_image, np.ndarray):
            original_image = original_image.numpy()
        if not isinstance(scribble_image, np.ndarray):
            scribble_image = scribble_image.numpy()
            
        # Ensure images are in BGR format for OpenCV
        if original_image.shape[-1] == 4:  # RGBA
            original_image = cv2.cvtColor(original_image, cv2.COLOR_RGBA2BGR)
        elif original_image.shape[-1] == 3:  # RGB
            original_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)
            
        if scribble_image.shape[-1] == 4:  # RGBA
            scribble_image = cv2.cvtColor(scribble_image, cv2.COLOR_RGBA2BGR)
        elif scribble_image.shape[-1] == 3:  # RGB
            scribble_image = cv2.cvtColor(scribble_image, cv2.COLOR_RGB2BGR)

        mask = ExtractMaskFromScribbleMap.get_map(original_image, scribble_image)
        return (mask,)