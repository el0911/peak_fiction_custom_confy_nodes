from ..utils.extract_mask_with_scrible_map import ExtractMaskFromScribbleMap
from ..utils.panorama_to_cube_map import PanoramaToCubemap
import numpy as np
import cv2
import logging
import os
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


# node to turn a panoram into a cubemap
class Cube_map_From_Panorama:
    """
    A node that converts a panorama image into a cubemap.
    Returns a list of 6 images representing the cubemap faces.
    """
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "panorama_image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE")
    FUNCTION = "convert_to_cubemap"
    CATEGORY = "peakfiction/custom"

    def convert_to_cubemap(self, panorama_image):
        # Placeholder for actual cubemap conversion logic
        # For now, just returning the input image six times
        #  make sure input is NDArray: Numpy array with shape [H, W, C].
        
        if not isinstance(panorama_image, np.ndarray):
            panorama_image = np.clip(
                255.0 * panorama_image.cpu().numpy().squeeze(), 0, 255
            ).astype(np.uint8)

        
        panorama_images = PanoramaToCubemap.panorama_to_cubemap(panorama_image, cube_size=512)
        print(f" length of images: {len(panorama_images)}")
        return (
            torch.from_numpy(panorama_images[0]).float().unsqueeze(0) / 255.0,
            torch.from_numpy(panorama_images[1]).float().unsqueeze(0) / 255.0,  
            torch.from_numpy(panorama_images[2]).float().unsqueeze(0) / 255.0,
            torch.from_numpy(panorama_images[3]).float().unsqueeze(0) / 255.0,
            torch.from_numpy(panorama_images[4]).float().unsqueeze(0) / 255.0,
            torch.from_numpy(panorama_images[5]).float().unsqueeze(0) / 255.0,
        )

    
# class to load a model from memory must be in the models/vggt folder loads all models in specified folder load a list of files from a folder and show it as options to the user its a select option as input not string
class Load_model_from_memory:
    """
    A node that loads a model from memory.
    Returns the loaded model.
    """
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        # load all files in the models/vggt folder and show them as options
        model_folder = os.path.join(os.path.dirname(__file__), "../../models/vggt")
        # Create the directory if it does not exist
        if not os.path.exists(model_folder):
            os.makedirs(model_folder)
        model_files = [f for f in os.listdir(model_folder) if f.endswith('.pt') or f.endswith('.pth')]
        return {
            "required": {
                "model_name": (model_files,),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_model"
    CATEGORY = "peakfiction/custom"

    def load_model(self, model_name):
        # Placeholder for actual model loading logic
        # For now, just returning a dummy model
        print(f"Loading model: {model_name}")
        return (torch.nn.Module(),)  # Replace with actual model loading logic