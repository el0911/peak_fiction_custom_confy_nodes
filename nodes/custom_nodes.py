from ..utils.extract_mask_with_scrible_map import ExtractMaskFromScribbleMap
import numpy as np
import cv2
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

    RETURN_TYPES = ("IMAGE","IMAGE",)
    FUNCTION = "get_map"
    CATEGORY = "peakfiction/custom"

    def get_map(self, original_image, scribble_image):
        # Convert tensors to NumPy arrays
        # if not isinstance(original_image, np.ndarray):
        original_image_ = tensor_to_numpy(original_image)
        # if not isinstance(scribble_image, np.ndarray):
        scribble_image_ = tensor_to_numpy(scribble_image)

        # # Debugging: Check the converted array shapes
        # print("Original Image Shape:", original_image.shape)
        # print("Scribble Image Shape:", scribble_image.shape)

        # # Ensure images are in BGR format for OpenCV
        # if original_image.shape[-1] == 4:  # RGBA
        #     original_image = cv2.cvtColor(original_image, cv2.COLOR_RGBA2BGR)
        # elif original_image.shape[-1] == 3:  # RGB
        #     original_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)

        # if scribble_image.shape[-1] == 4:  # RGBA
        #     scribble_image = cv2.cvtColor(scribble_image, cv2.COLOR_RGBA2BGR)
        # elif scribble_image.shape[-1] == 3:  # RGB
        #     scribble_image = cv2.cvtColor(scribble_image, cv2.COLOR_RGB2BGR)

        # Optionally use ExtractMaskFromScribbleMap
        # mask = ExtractMaskFromScribbleMap.get_map(original_image, scribble_image)
        # return (original_image, scribble_image, mask)



        return (original_image, scribble_image)
