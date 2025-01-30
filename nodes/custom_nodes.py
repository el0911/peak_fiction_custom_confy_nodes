from ..utils.extract_mask_with_scrible_map import ExtractMaskFromScribbleMap
import numpy as np
import cv2
import logging
import torch  # If using PyTorch
# import tensorflow as tf  # Uncomment if using TensorFlow

def tensor_to_numpy(tensor):
    # Convert the tensor to a numpy array
    numpy_image = tensor.numpy()

    # Convert the numpy array to a cv2 image
    cv2_image = np.transpose(numpy_image, (1, 2, 0))
    cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    return cv2_image


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

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "get_map"
    CATEGORY = "peakfiction/custom"

   # Set up logging configuration
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

    
    def get_map(self, original_image, scribble_image):
        try:
            # If input is already a NumPy array, skip conversion
            if not isinstance(original_image, np.ndarray):
                original_image = np.clip(
                    255.0 * original_image.cpu().numpy().squeeze(), 0, 255
                ).astype(np.uint8)

            if not isinstance(scribble_image, np.ndarray):
                scribble_image = np.clip(
                    255.0 * scribble_image.cpu().numpy().squeeze(), 0, 255
                ).astype(np.uint8)

            # Debugging: Check the converted array shapes
            logging.debug(f"Original Image Shape: {original_image.shape}")
            logging.debug(f"Scribble Image Shape: {scribble_image.shape}")

            # Your further processing logic here...
            # For example: compute some map or apply transformations

            # Ensure images are in BGR format for OpenCV
            if original_image.shape[-1] == 4:  # RGBA
                original_image = cv2.cvtColor(original_image, cv2.COLOR_RGBA2BGR)
                logging.debug("Converted original image from RGBA to BGR.")
            elif original_image.shape[-1] == 3:  # RGB
                original_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)
                logging.debug("Converted original image from RGB to BGR.")

            if scribble_image.shape[-1] == 4:  # RGBA
                scribble_image = cv2.cvtColor(scribble_image, cv2.COLOR_RGBA2BGR)
                logging.debug("Converted scribble image from RGBA to BGR.")
            elif scribble_image.shape[-1] == 3:  # RGB
                scribble_image = cv2.cvtColor(scribble_image, cv2.COLOR_RGB2BGR)
                logging.debug("Converted scribble image from RGB to BGR.")

            # Optionally use ExtractMaskFromScribbleMap
            mask = ExtractMaskFromScribbleMap.get_map(original_image, scribble_image)
            
            # print(type(original_image_))
            print(type(mask))
            print(type(mask))
            print(type(mask))
            print(type(mask))
            print(mask)
            
            # Assuming 'mask' is the NumPy array you got from the ExtractMaskFromScribbleMap method
            # # Convert the NumPy array to a tensor
            # mask_tensor = torch.from_numpy(mask)

            # # Reshape to add batch and channel dimensions (if needed)
            # mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)

            # # Optionally, convert to float type for image processing or neural networks
            # mask_tensor = mask_tensor.float()
 


            # Check if mask is returned
            # if mask is None:
            #     logging.warning("No mask found, returning original image.")
            #     return original_image  # Return original image if no mask is found
            
            print("Mask successfully created.")
            return (mask,)
        
        except Exception as e:
            # Log the exception details
            logging.error(f"An error occurred while processing images: {str(e)}")
            # In case of an error, return the original image
            return (original_image,)